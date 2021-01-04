import matplotlib.pyplot as plt
import numpy as np


def paint_bars(
    ax,
    data,
    labels: list,
    weights,
    bins: int,
    range: tuple,
    title: str = None,
    x_label: str = None,
    y_label: str = None,
    x_unit: str = None,
    x_scale: float = None,
    density: bool = False,
    use_error: bool = False,
    color: str = "black",
    fmt: str = ".k",
) -> None:
    """Plot with vertical bar, can be used for data display.

        Note:
        According to ROOT:
        "The error per bin will be computed as sqrt(sum of squares of weight) for each bin."

    """
    plt.ioff()
    # Check input
    data_1dim = np.array([])
    weight_1dim = np.array([])
    if isinstance(data, np.ndarray):
        data = [data]
        weights = [weights]
    for datum, weight in zip(data, weights):
        assert isinstance(datum, np.ndarray), "data element should be numpy array."
        assert isinstance(weight, np.ndarray), "weights element should be numpy array."
        assert (
            datum.shape == weight.shape
        ), "Input weights should be None or have same type as arrays."
        if len(data_1dim) == 0:
            data_1dim = datum
            weight_1dim = weight
        else:
            data_1dim = np.concatenate((data_1dim, datum))
            weight_1dim = np.concatenate((weight_1dim, weight))

    # Scale x axis
    if x_scale is not None:
        data_1dim = data_1dim * x_scale
    # Make bar plot
    # get bin error and edges
    plot_ys, _ = np.histogram(
        data_1dim, bins=bins, range=range, weights=weight_1dim, density=density
    )
    sum_weight_squares, bin_edges = np.histogram(
        data_1dim, bins=bins, range=range, weights=np.power(weight_1dim, 2)
    )
    if density:
        error_scale = 1 / (np.sum(weight_1dim) * (range[1] - range[0]) / bins)
        errors = np.sqrt(sum_weight_squares) * error_scale
    else:
        errors = np.sqrt(sum_weight_squares)
    # Only plot ratio when bin is not 0.
    bin_centers = np.array([])
    bin_ys = np.array([])
    bin_yerrs = np.array([])
    for i, y1 in enumerate(plot_ys):
        if y1 != 0:
            ele_center = np.array([0.5 * (bin_edges[i] + bin_edges[i + 1])])
            bin_centers = np.concatenate((bin_centers, ele_center))
            ele_y = np.array([y1])
            bin_ys = np.concatenate((bin_ys, ele_y))
            ele_yerr = np.array([errors[i]])
            bin_yerrs = np.concatenate((bin_yerrs, ele_yerr))
    # plot bar
    bin_size = bin_edges[1] - bin_edges[0]
    if use_error:
        ax.errorbar(
            bin_centers,
            bin_ys,
            xerr=bin_size / 2.0,
            yerr=bin_yerrs,
            fmt=fmt,
            label=labels,
            color=color,
            markerfacecolor=color,
            markeredgecolor=color,
        )
    else:
        ax.errorbar(
            bin_centers,
            bin_ys,
            xerr=bin_size / 2.0,
            yerr=None,
            fmt=fmt,
            label=labels,
            color=color,
            markerfacecolor=color,
            markeredgecolor=color,
        )
    # Config
    if title is not None:
        ax.set_title(title)
    if x_label is not None:
        if x_unit is not None:
            ax.set_xlabel(x_label + "/" + x_unit)
        else:
            ax.set_xlabel(x_label)
    else:
        if x_unit is not None:
            ax.set_xlabel(x_unit)
    if y_label is not None:
        ax.set_ylabel(y_label)
    if range is not None:
        ax.axis(xmin=range[0], xmax=range[1])
    ax.legend(loc="upper right")
