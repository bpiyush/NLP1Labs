"""Helper functions"""
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
