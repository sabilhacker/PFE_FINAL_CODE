from matplotlib import pyplot as plt
import numpy as np


def plot_average_loss_with_variance(losses, N=10):
    """
    Plots the average and variance of training loss over every N steps.

    Args:
        losses (float): loss values to plot.
        N (int): Number of steps to average over.

    Returns:
        None
    """

    # Calculate the means and variances over every N steps
    means = []
    variances = []

    for i in range(0, len(losses), N):
        # Handle the last data point if it's smaller than N
        end = i + N
        if end > len(losses) and len(losses[i:]) < N:
            chunk = np.concatenate([losses[i - (N - len(losses[i:])) : i], losses[i:]])
        else:
            chunk = losses[i:end]

        means.append(np.mean(chunk))
        variances.append(np.var(chunk))

    # Create an x-axis for the plot
    x = np.arange(0, len(losses), N)[: len(means)]

    # Plot the means and variances
    plt.plot(x, means, "-o", label="Mean Loss")
    plt.fill_between(
        x,
        np.array(means) - np.array(variances),
        np.array(means) + np.array(variances),
        color="gray",
        alpha=0.2,
        label="Variance",
    )

    # Set the title and labels for the plot
    plt.title("Training Loss over Time")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()

    # Display the plot
    plt.show()
