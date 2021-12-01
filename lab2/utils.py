"""Helper functions"""
import os
import numpy as np
import matplotlib.pyplot as plt


def plot_single_categorical_histogram(
        values=None,
        x=None,
        counts=None,
        ax=None,
        figsize=(6, 5),
        title="Categorical distribution",
        xlabel="Classes",
        ylabel="Counts",
        bar_label=None,
        show=True,
    ):
    """Plots bar-plot."""

    if values is not None:
        assert x is None and counts is None
        x, counts = np.unique(values, return_counts=True)
    else:
        assert x is not None and counts is not None

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)

    ax.bar(x=x, height=counts, width=0.5, label=bar_label, align="center")
    ax.grid()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if bar_label is not None:
        ax.legend()
    
    if show and ax is None:
        plt.show()

    
def plot_multiple_categorical_histogram(
        lists_of_values,
        labels,
        width=0.6,
        figsize=(6, 5),
        title="Categorical distribution",
        xlabel="Classes",
        ylabel="Counts",
    ):
    """Plots bar-plot."""
    assert len(lists_of_values) == len(labels)
    num_distributions = len(labels)
    width_per_dist = width / num_distributions
    starting_points = np.linspace(-width_per_dist, width_per_dist, num_distributions)
    offset = width_per_dist / num_distributions
    center_points = [x + ((x < 0) * offset) + (-1 * (x > 0) * offset) for x in starting_points]

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for i, values in enumerate(lists_of_values):
        x, counts = np.unique(values, return_counts=True)
        ax.bar(x=x - center_points[i], height=counts, label=labels[i], width=width_per_dist, align="center")

    ax.grid()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

    plt.show()


def plot_single_sequence(
        x, y, plot_label, x_label="Iterations", y_label="Loss", title=None,
        save=True, save_path="./results/sample.png", show=True,
    ):
    """Plots sequences y1 and y1 vs x."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    
    if y_label == "Loss":
        optima_fn = lambda x: np.min(y)
    elif y_label == "Accuracy":
        optima_fn = lambda x: np.max(y)
    else:
        raise NotImplementedError

    plot_label = f"{plot_label} (Best: {optima_fn(y):.4f})"
    ax.plot(x, y, "--o", label=plot_label)

    ax.grid()
    if title is None:
        title = f"{y_label.capitalize()} vs {x_label.capitalize()}"
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.legend()

    if save:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    
    if show:
        plt.show()


def plot_two_sequences(
        x, y1, y2, y1_label, y2_label, x_label, y_label, title,
        save=True, save_path="./results/sample.png", show=True,
    ):
    """Plots sequences y1 and y1 vs x."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    if len(y1):
        ax.plot(x, y1, "--o", label=y1_label)
    if len(y2):
        ax.plot(x, y2, "--o", label=y2_label)

    ax.grid()
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.legend()

    if save:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    
    if show:
        plt.show()


def load_txt(path: str):
    """Loads data from text file."""
    with open(path, "r") as f:
        data = f.read()
    return data.split("\n")