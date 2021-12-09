"""Defines the training and evaluation routines."""
import time
import random
import numpy as np
import torch
from torch import nn
from torch import optim


# Select the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def fix_seed(seed=42):
    # Seed manually to make runs reproducible
    # You need to set this again if you do multiple runs of the same model
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # When running on the CuDNN backend two further options must be set for reproducibility
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_examples(data, shuffle=True, verbose=True, **kwargs):
    """Shuffle data set and return 1 example at a time (until nothing left)"""
    if shuffle:
        if verbose:
            print("Shuffling training data")
        random.shuffle(data)  # shuffle training data each epoch
    for example in data:
        yield example

def prepare_example(example, vocab):
    """
    Map tokens to their IDs for a single example
    """

    # vocab returns 0 if the word is not there (i2w[0] = <unk>)
    x = [vocab.w2i.get(t, 0) for t in example.tokens]

    x = torch.LongTensor([x])
    x = x.to(device)

    y = torch.LongTensor([example.label])
    y = y.to(device)

    return x, y


def simple_evaluate(model, data, prep_fn=prepare_example, **kwargs):
    """Accuracy of a model on given data set."""
    correct = 0
    total = 0
    model.eval()  # disable dropout (explained later)

    for example in data:

        # convert the example input and label to PyTorch tensors
        x, target = prep_fn(example, model.vocab)

        # forward pass without backpropagation (no_grad)
        # get the output from the neural network for input x
        with torch.no_grad():
            logits = model(x)

        # get the prediction
        prediction = logits.argmax(dim=-1)

        # add the number of correct predictions to the total correct
        correct += (prediction == target).sum().item()
        total += 1

    return correct, total, correct / float(total)


def get_minibatch(data, batch_size=25, shuffle=True, verbose=True):
    """Return minibatches, optional shuffling"""
  
    if shuffle:
        if verbose:
            print("Shuffling training data")
        random.shuffle(data)  # shuffle training data each epoch
  
    batch = []
  
    # yield minibatches
    for example in data:
        batch.append(example)
    
        if len(batch) == batch_size:
            yield batch
            batch = []
      
    # in case there is something left
    if len(batch) > 0:
        yield batch


def pad(tokens, length, pad_value=1):
    """add padding 1s to a sequence to that it has the desired length"""
    return tokens + [pad_value] * (length - len(tokens))


def prepare_minibatch(mb, vocab):
    """
    Minibatch is a list of examples.
    This function converts words to IDs and returns
    torch tensors to be used as input/targets.
    """
    batch_size = len(mb)
    maxlen = max([len(ex.tokens) for ex in mb])

    # vocab returns 0 if the word is not there
    x = [pad([vocab.w2i.get(t, 0) for t in ex.tokens], maxlen) for ex in mb]

    x = torch.LongTensor(x)
    x = x.to(device)

    y = [ex.label for ex in mb]
    y = torch.LongTensor(y)
    y = y.to(device)

    return x, y


def evaluate(model, data, 
             batch_fn=get_minibatch, prep_fn=prepare_minibatch,
             batch_size=16):
    """Accuracy of a model on given data set (using mini-batches)"""
    correct = 0
    total = 0
    model.eval()  # disable dropout

    for mb in batch_fn(data, batch_size=batch_size, shuffle=False):
        x, targets = prep_fn(mb, model.vocab)
        with torch.no_grad():
            logits = model(x)
      
        predictions = logits.argmax(dim=-1).view(-1)
    
        # add the number of correct predictions to the total correct
        correct += (predictions == targets.view(-1)).sum().item()
        total += targets.size(0)

    return correct, total, correct / float(total)


def prepare_treelstm_minibatch(mb, vocab):
    """
    Returns sentences reversed (last word first)
    Returns transitions together with the sentences.  
    """
    batch_size = len(mb)
    maxlen = max([len(ex.tokens) for ex in mb])

    # vocab returns 0 if the word is not there
    # NOTE: reversed sequence!
    x = [pad([vocab.w2i.get(t, 0) for t in ex.tokens], maxlen)[::-1] for ex in mb]

    x = torch.LongTensor(x)
    x = x.to(device)

    y = [ex.label for ex in mb]
    y = torch.LongTensor(y)
    y = y.to(device)

    maxlen_t = max([len(ex.transitions) for ex in mb])
    transitions = [pad(ex.transitions, maxlen_t, pad_value=2) for ex in mb]
    transitions = np.array(transitions)
    transitions = transitions.T  # time-major

    return (x, transitions), y


def train_model(
        model,
        optimizer,
        train_data,
        dev_data,
        test_data,
        num_iterations=10000, 
        print_every=1000,
        eval_every=1000,
        batch_fn=get_examples,
        prep_fn=prepare_example,
        eval_fn=simple_evaluate,
        batch_size=1,
        eval_batch_size=None,
        verbose=True,
    ):
    """Train a model."""  
    iter_i = 0
    train_loss = 0.
    print_num = 0
    start = time.time()
    criterion = nn.CrossEntropyLoss() # loss function
    best_eval = 0.
    best_iter = 0

    # store train loss and validation accuracy during training
    # so we can plot them afterwards
    losses = []
    accuracies = []  
  
    if eval_batch_size is None:
        eval_batch_size = batch_size
  
    while True:  # when we run out of examples, shuffle and continue
        for batch in batch_fn(train_data, batch_size=batch_size, verbose=verbose):

            # forward pass
            model.train()
            x, targets = prep_fn(batch, model.vocab)
            logits = model(x)

            B = targets.size(0)  # later we will use B examples per update

            # compute cross-entropy loss (our criterion)
            # note that the cross entropy loss function computes the softmax for us
            loss = criterion(logits.view([B, -1]), targets.view(-1))
            train_loss += loss.item()

            # backward pass (tip: check the Introduction to PyTorch notebook)

            # erase previous gradients
            optimizer.zero_grad()

            # compute gradients
            loss.backward()

            # update weights - take a small step in the opposite dir of the gradient
            optimizer.step()

            print_num += 1
            iter_i += 1

            # print info
            if iter_i % print_every == 0:
                if verbose:
                    print("Iter %r: loss=%.4f, time=%.2fs" % (iter_i, train_loss, time.time()-start))
                losses.append(train_loss)
                print_num = 0        
                train_loss = 0.

            # evaluate
            if iter_i % eval_every == 0:
                _, _, accuracy = eval_fn(model, dev_data, batch_size=eval_batch_size, batch_fn=batch_fn, prep_fn=prep_fn)
                accuracies.append(accuracy)
                if verbose:
                    print("iter %r: dev acc=%.4f" % (iter_i, accuracy))

                # save best model parameters
                if accuracy > best_eval:
                    if verbose:
                        print("new highscore")
                    best_eval = accuracy
                    best_iter = iter_i
                    path = "{}.pt".format(model.__class__.__name__)
                    ckpt = {
                        "state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_eval": best_eval,
                        "best_iter": best_iter
                    }
                    torch.save(ckpt, path)

            # done training
            if iter_i == num_iterations:
                print("Done training")

                # evaluate on train, dev, and test with best model
                print("Loading best model")
                path = "{}.pt".format(model.__class__.__name__)        
                ckpt = torch.load(path)
                model.load_state_dict(ckpt["state_dict"])

                _, _, train_acc = eval_fn(
                    model, train_data, batch_size=eval_batch_size, 
                    batch_fn=batch_fn, prep_fn=prep_fn)
                _, _, dev_acc = eval_fn(
                    model, dev_data, batch_size=eval_batch_size,
                    batch_fn=batch_fn, prep_fn=prep_fn)
                _, _, test_acc = eval_fn(
                    model, test_data, batch_size=eval_batch_size, 
                    batch_fn=batch_fn, prep_fn=prep_fn)
                
                best_model_acc = {
                    "train": train_acc,
                    "dev": dev_acc,
                    "test": test_acc,
                }

                print("best model iter {:d}: "
                      "train acc={:.4f}, dev acc={:.4f}, test acc={:.4f}".format(
                          best_iter, train_acc, dev_acc, test_acc))

                return losses, accuracies, best_model_acc, ckpt