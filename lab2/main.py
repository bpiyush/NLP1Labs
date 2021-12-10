"""Entrypoint to training and evaluation."""
import os
from collections import defaultdict
from copy import deepcopy
from posixpath import basename
import numpy as np
import torch
from torch import optim

from data.example import Example, examplereader
from data.vocabulary import OrderedDict, Vocabulary
from utils.io import load_txt, save_json
from utils.plot import plot_single_sequence
from utils.text import tokens_from_treestring, transitions_from_treestring

import models

from train import fix_seed, train_model, custom_evaluate


def collect_all_subtrees(data, use_head_label=True):
    data_subexamples = []
    data_idtracker = []

    for i, example in enumerate(data):

        subexamples = []
        idtracker = []
        subtrees = [subtree for subtree in example.tree.subtrees()]

        for subtree in subtrees:
            subtree_string = ' '.join(str(subtree).split())

            if use_head_label:
                label = example.label
            else:
                label = int(subtree_string[1])

            subexample = Example(
                tokens=tokens_from_treestring(subtree_string),
                transitions=transitions_from_treestring(subtree_string),
                label=label,
                tree=subtree,
            )
            subexamples.append(subexample)
            idtracker.append(i)

        data_subexamples.extend(subexamples)
        data_idtracker.extend(idtracker)
    
    return data_subexamples, data_idtracker


def setup_data(
    use_pretrained_embeddings,
    pretrained_embeddings_path,
    train_path="trees/train.txt",
    dev_path="trees/dev.txt",
    test_path="trees/test.txt",
    lower=False,
    use_subtrees=False,
):
    train_data = list(examplereader(train_path, lower=lower))
    if use_subtrees:
        print(".... Using supervision from subtrees ....")
        print(f"Training data size (before): {len(train_data)}")
        train_data, _ = collect_all_subtrees(train_data, use_head_label=False)
        print(f"Training data size (after): {len(train_data)}")

    dev_data = list(examplereader(dev_path, lower=lower))
    test_data = list(examplereader(test_path, lower=lower))

    print("::: Configuring data :::")
    print("Train: \t", len(train_data))
    print("Dev: \t", len(dev_data))
    print("Test:\t", len(test_data))

    # Now let's map the sentiment labels 0-4 to a more readable form
    i2t = ["very negative", "negative", "neutral", "positive", "very positive"]
    t2i = OrderedDict({p : i for p, i in zip(i2t, range(len(i2t)))})

    # 2. create vocabularies
    if not use_pretrained_embeddings:
        print("::: Using training dataset to create vocabulary. :::")
        # use the standard vocabularies constructed from the training data
        v = Vocabulary()
        for data_set in (train_data,):
            for ex in data_set:
                for token in ex.tokens:
                    v.count_token(token)

        v.build()
        print("Vocabulary size:", len(v.w2i))
    else:
        print("::: Using word-embeddings to create vocabulary. :::")
        word_embeddings_txt = load_txt(pretrained_embeddings_path)
        v = Vocabulary()
        vectors = []
        for line in word_embeddings_txt[:-1]:
            token, vector = line.split(" ")[0], line.split(" ")[1:]
            vector = np.array([float(y) for y in vector])
            vectors.append(vector)
            v.count_token(token)

        v.build()
        vectors = np.stack(vectors, axis=0)

        # add zero-vectors for <unk> and <pad>
        vectors = np.concatenate([np.zeros((2, 300)), vectors], axis=0)

        print("Vocabulary size:", len(v.w2i), "\t Vectors shape: ", vectors.shape)
    
    return train_data, dev_data, test_data, v, t2i, i2t, vectors


def setup_model(model_name, model_args, v):
    add_model_args = {
        "vocab_size": len(v.w2i),
        "vocab": v,
    }
    model = models.__dict__[model_name](**model_args, **add_model_args)
    return model


def run_experiment(
        model_name,
        model_args,
        optim_args,
        train_args,
        seed=42,
        use_pretrained_embeddings=False,
        pretrained_embeddings_path="./googlenews.word2vec.300d.txt",
        plot_losses=True,
        plot_accuracies=True,
        verbose=True,
        expt_name="",
        use_subtrees=False,
    ):
    """Runs a single experiment for given model on the SST dataset."""
    configs = {
        "model_args": deepcopy(model_args),
        "train_args": deepcopy(train_args),
        "optim_args": deepcopy(optim_args),
        "seed": seed,
        "use_pretrained_embeddings": use_pretrained_embeddings,
        "pretrained_embeddings_path": pretrained_embeddings_path,
    }
    
    # 0. fix seed and select device
    fix_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # # 1. load data
    # LOWER = False  # we will keep the original casing
    # train_data = list(examplereader("trees/train.txt", lower=LOWER))
    # dev_data = list(examplereader("trees/dev.txt", lower=LOWER))
    # test_data = list(examplereader("trees/test.txt", lower=LOWER))

    # print("::: Configuring data :::")
    # print("Train: \t", len(train_data))
    # print("Dev: \t", len(dev_data))
    # print("Test:\t", len(test_data))

    # # Now let's map the sentiment labels 0-4 to a more readable form
    # i2t = ["very negative", "negative", "neutral", "positive", "very positive"]
    # t2i = OrderedDict({p : i for p, i in zip(i2t, range(len(i2t)))})

    # # 2. create vocabularies
    # if not use_pretrained_embeddings:
    #     print("::: Using training dataset to create vocabulary. :::")
    #     # use the standard vocabularies constructed from the training data
    #     v = Vocabulary()
    #     for data_set in (train_data,):
    #         for ex in data_set:
    #             for token in ex.tokens:
    #                 v.count_token(token)

    #     v.build()
    #     print("Vocabulary size:", len(v.w2i))
    # else:
    #     print("::: Using word-embeddings to create vocabulary. :::")
    #     word_embeddings_txt = load_txt(pretrained_embeddings_path)
    #     v = Vocabulary()
    #     vectors = []
    #     for line in word_embeddings_txt[:-1]:
    #         token, vector = line.split(" ")[0], line.split(" ")[1:]
    #         vector = np.array([float(y) for y in vector])
    #         vectors.append(vector)
    #         v.count_token(token)

    #     v.build()
    #     vectors = np.stack(vectors, axis=0)

    #     # add zero-vectors for <unk> and <pad>
    #     vectors = np.concatenate([np.zeros((2, 300)), vectors], axis=0)

    #     print("Vocabulary size:", len(v.w2i), "\t Vectors shape: ", vectors.shape)

    train_data, dev_data, test_data, v, t2i, i2t, vectors = setup_data(
        use_pretrained_embeddings=use_pretrained_embeddings,
        pretrained_embeddings_path=pretrained_embeddings_path,
        use_subtrees=use_subtrees,
    )    
    
    print("::: Configuring model :::")
    # add_model_args = {
    #     "vocab_size": len(v.w2i),
    #     "vocab": v,
    # }
    # model = models.__dict__[model_name](**model_args, **add_model_args)
    model = setup_model(model_name, model_args, v)
    
    if use_pretrained_embeddings:
        assert hasattr(model, "init_embedding_weights")
        model.init_embedding_weights(vectors)

    model = model.to(device)
    print(model)
    models.print_parameters(model)
    
    print("::: Configuring optimizer :::")
    optimizer = optim.Adam(model.parameters(), **optim_args)
    
    print("::: Configuring training :::")
    add_train_args = {
        "model": model,
        "optimizer": optimizer,
        "train_data": train_data,
        "dev_data": dev_data,
        "test_data": test_data,
        "verbose": verbose,
    }
    losses, accuracies, best_model_acc, ckpt = train_model(**train_args, **add_train_args)
    
    print("::: Configuring plotting :::")
    if plot_losses:
        plot_single_sequence(
            range(len(accuracies)), accuracies,
            plot_label=model_name, y_label="Accuracy", marker="-",
            title=f"{model_name} (Best test accuracy: {best_model_acc['test']:.4f})",
        )

    if plot_accuracies:
        plot_single_sequence(
            range(len(losses)), losses,
            plot_label=model_name, y_label="Loss", marker="-",
            title=f"{model_name} (Best test accuracy: {best_model_acc['test']:.4f})",
        )
    
    print("::: Saving checkpoint and logs :::")
    if not len(expt_name):
        expt_name = f"{model_name}_seed_{seed}"

    ckpt_path = os.path.join("checkpoints", f"{expt_name}-best_model.ckpt")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(ckpt, ckpt_path)
    
    logs_path = os.path.join("logs", f"{expt_name}-train_logs.json")
    # remove problematic keys from configs
    if "batch_fn" in configs["train_args"]:
        del configs["train_args"]["batch_fn"]
    if "eval_fn" in configs["train_args"]:
        del configs["train_args"]["eval_fn"]
    if "prep_fn" in configs["train_args"]:
        del configs["train_args"]["prep_fn"]
    logs = {
        "model_name": model_name,
        "losses": losses,
        "accuracies": accuracies,
        "best_model_acc": best_model_acc,
        "configs": configs,
    }
    os.makedirs(os.path.dirname(logs_path), exist_ok=True)
    save_json(logs, logs_path)

    return losses, accuracies, best_model_acc


def run_multiple_seed_experiments(expt_args, seeds=[0, 42, 420]):
    """Runs multiple experiments with different seeds."""
    best_agg_acc = defaultdict(list)
    expt_name_prefix = expt_args["expt_name"]
    for seed in seeds:
        print(f":::::::::::::::::::::::: Seed : {seed} ::::::::::::::::::::::::")
        expt_args.update(
            {
                "seed": seed,
                "expt_name": f"{expt_name_prefix}-{expt_args['model_name']}_seed_{seed}",
            },
        )
        _, _, best_model_acc = run_experiment(**expt_args)
        for phase in best_model_acc:
            best_agg_acc[phase].append(best_model_acc[phase])
        print(f":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")

    for phase in best_agg_acc:
        print(f"{phase}\t: Mean\t: {np.mean(best_agg_acc[phase]):.4f} Std\t: {np.std(best_agg_acc[phase]):.4f}")
    
    seeds = [str(s) for s in seeds]
    logs_path = os.path.join("logs", f"{expt_args['model_name']}-seeds_{'_'.join(seeds)}.json")
    save_json(best_agg_acc, logs_path)

    return best_agg_acc


def eval_experiment(
        model_name,
        model_args,
        eval_args,
        ckpt_path,
        seed=42,
        use_pretrained_embeddings=False,
        pretrained_embeddings_path="./googlenews.word2vec.300d.txt",
        expt_name="",  
    ):
    """Evaluates a trained model on a test set."""
    fix_seed(seed)

    # load data
    train_data, dev_data, test_data, v, t2i, i2t = setup_data(
        use_pretrained_embeddings=use_pretrained_embeddings,
        pretrained_embeddings_path=pretrained_embeddings_path,
    )

    # load model
    model = setup_model(model_name, model_args, v=v)
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt["state_dict"])
    
    # run forward pass
    results = custom_evaluate(
        model, test_data, **eval_args,
    )
    
    # save results
    if not len(expt_name):
        expt_name = f"{os.path.basename(ckpt_path).split('.ckpt')[0]}"
    results_path = os.path.join("results", f"{expt_name}-results.pt")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    torch.save(results, results_path)
    print(f"::: Saved results at {results_path} :::")
    