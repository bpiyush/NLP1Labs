"""Entrypoint to training and evaluation."""
import os
from collections import defaultdict
import numpy as np
import torch
from torch import optim

from data.example import Example, examplereader
from data.vocabulary import OrderedDict, Vocabulary
from utils.io import load_txt, save_json
from utils.plot import plot_single_sequence

import models

from train import fix_seed, train_model


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
    ):
    """Runs a single experiment for given model on the SST dataset."""
    configs = locals()
    
    # 0. fix seed and select device
    fix_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. load data
    LOWER = False  # we will keep the original casing
    train_data = list(examplereader("trees/train.txt", lower=LOWER))
    dev_data = list(examplereader("trees/dev.txt", lower=LOWER))
    test_data = list(examplereader("trees/test.txt", lower=LOWER))

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
    
    
    print("::: Configuring model :::")
    add_model_args = {
        "vocab_size": len(v.w2i),
        "vocab": v,
    }
    model = models.__dict__[model_name](**model_args, **add_model_args)
    
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
    for seed in seeds:
        print(f":::::::::::::::::::::::: Seed : {seed} ::::::::::::::::::::::::")
        expt_args.update({"seed": seed})
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