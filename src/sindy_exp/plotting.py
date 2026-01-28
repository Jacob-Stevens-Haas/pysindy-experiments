from dataclasses import dataclass
from typing import Annotated, Optional, Sequence
from warnings import warn

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns
import sympy as sp
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.typing import ColorType

PAL = sns.color_palette("Set1")
PLOT_KWS = {"alpha": 0.7, "linewidth": 3}


@dataclass
class _ColorConstants:
    color_sequence: list[ColorType]

    def set_sequence(self, color_sequence: list[ColorType]):
        self.color_sequence = color_sequence

    @property
    def TRUE(self):
        return self.color_sequence[0]

    @property
    def MEAS(self):
        return self.color_sequence[1]

    @property
    def EST(self):
        return self.color_sequence[2]

    @property
    def SIM(self):
        return self.color_sequence[3]

    @property
    def TRAIN(self):
        return self.color_sequence[4]

    @property
    def TEST(self):
        return self.color_sequence[5]


COLOR = _ColorConstants(mpl.color_sequences["tab10"])


def plot_coefficients(
    coefficients: Annotated[np.ndarray, "(n_coord, n_features)"],
    input_features: Sequence[str],
    feature_names: Sequence[str],
    ax: Axes,
    **heatmap_kws,
) -> None:
    """Plot a set of dynamical system coefficients in a heatmap.

    Args:
        coefficients: A 2D array holding the coefficients of different
            library functions.  System dimension is rows, function index
            is columns
        input_features: system coordinate names, e.g. "x","y","z" or "u","v"
        feature_names: the names of the functions in the library.
        ax: the matplotlib axis to plot on
        **heatmap_kws: additional kwargs to seaborn's styling
    """

    def detex(input: str) -> str:
        if input[0] == "$":
            input = input[1:]
        if input[-1] == "$":
            input = input[:-1]
        return input

    if input_features is None:
        input_features = [r"$\dot x_" + f"{k}$" for k in range(coefficients.shape[0])]
    else:
        input_features = [r"$\dot " + f"{detex(fi)}$" for fi in input_features]

    if feature_names is None:
        feature_names = [f"f{k}" for k in range(coefficients.shape[1])]

    with sns.axes_style(style="white", rc={"axes.facecolor": (0, 0, 0, 0)}):
        heatmap_args = {
            "xticklabels": input_features,
            "yticklabels": feature_names,
            "center": 0.0,
            "cmap": sns.color_palette("vlag", n_colors=20, as_cmap=True),
            "ax": ax,
            "linewidths": 0.1,
            "linecolor": "whitesmoke",
        }
        heatmap_args.update(**heatmap_kws)
        coefficients = np.where(
            coefficients == 0, np.nan * np.empty_like(coefficients), coefficients
        )
        sns.heatmap(coefficients.T, **heatmap_args)

        ax.tick_params(axis="y", rotation=0)

    return ax


def _coeff_dicts_to_matrix(
    coeffs: Sequence[dict[sp.Expr, float]],
) -> tuple[np.ndarray, list[str]]:
    """Convert a list of coefficient dictionaries to a dense matrix.

    The input is a list of dictionaries mapping feature identifiers (
    SymPy expressions) to coefficients. All dictionaries are assumed to
    correspond to different coordinates of the same system. This helper builds
    a consistent feature ordering across coordinates and returns a numeric
    matrix along with the stringified feature names.
    """

    if not coeffs:
        raise ValueError("No coefficient dictionaries provided.")

    features: list[sp.Expr] = sorted({key for d in coeffs for key in d.keys()}, key=str)

    mat = np.zeros((len(coeffs), len(features)), dtype=float)
    for row, d in enumerate(coeffs):
        for col, feat in enumerate(features):
            mat[row, col] = d[feat]

    feature_names = [str(f).replace("**", "^") for f in features]
    return mat, feature_names


def _compare_coefficient_plots_impl(
    coefficients_est: Annotated[np.ndarray, "(n_coord, n_feat)"],
    coefficients_true: Annotated[np.ndarray, "(n_coord, n_feat)"],
    input_features: Sequence[str],
    feature_names: Sequence[str],
    scaling: bool = True,
    axs: Optional[Sequence[Axes]] = None,
) -> None:
    """Internal implementation for coefficient comparison heatmaps."""
    n_cols = len(coefficients_est)

    # helps boost the color of small coefficients.  Maybe log is better?
    all_vals = np.hstack((coefficients_est.flatten(), coefficients_true.flatten()))
    nzs = all_vals[all_vals.nonzero()]
    max_val = np.max(np.abs(nzs), initial=0.0)
    min_val = np.min(np.abs(nzs), initial=np.inf)
    if scaling and np.isfinite(min_val) and max_val / min_val > 10:
        pwr_ratio = 1.0 / np.log10(max_val / min_val)
    else:
        pwr_ratio = 1

    def signed_root(x):
        return np.sign(x) * np.power(np.abs(x), pwr_ratio)

    with sns.axes_style(style="white", rc={"axes.facecolor": (0, 0, 0, 0)}):
        if axs is None:
            fig, axs = plt.subplots(
                1, 2, figsize=(1.9 * n_cols, 8), sharey=True, sharex=True
            )
            fig.tight_layout()

        vmax = signed_root(max_val)

        plot_coefficients(
            signed_root(coefficients_true),
            input_features=input_features,
            feature_names=feature_names,
            ax=axs[0],
            cbar=False,
            vmax=vmax,
            vmin=-vmax,
        )

        plot_coefficients(
            signed_root(coefficients_est),
            input_features=input_features,
            feature_names=feature_names,
            ax=axs[1],
            cbar=False,
            vmax=vmax,
            vmin=-vmax,
        )

        axs[0].set_title("True Coefficients", rotation=45)
        axs[1].set_title("Est. Coefficients", rotation=45)


def compare_coefficient_plots(
    coefficients_est: Annotated[np.ndarray, "(n_coord, n_feat)"],
    coefficients_true: Annotated[np.ndarray, "(n_coord, n_feat)"],
    input_features: Sequence[str],
    feature_names: Sequence[str],
    scaling: bool = True,
    axs: Optional[Sequence[Axes]] = None,
) -> None:
    """Create plots of true and estimated coefficients.

    Deprecated:
        Use :func:`compare_coefficient_plots_from_dicts` with coefficient
        dictionaries instead. This function will be removed in a future
        release.
    """

    warn(
        "compare_coefficient_plots is deprecated; use "
        "compare_coefficient_plots_from_dicts with coefficient dictionaries instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    _compare_coefficient_plots_impl(
        coefficients_est,
        coefficients_true,
        input_features=input_features,
        feature_names=feature_names,
        scaling=scaling,
        axs=axs,
    )


def compare_coefficient_plots_from_dicts(
    coefficients_est: Sequence[dict[sp.Expr, float]],
    coefficients_true: Sequence[dict[sp.Expr, float]],
    input_features: Sequence[str],
    feature_names: Sequence[str] | None = None,
    scaling: bool = True,
    axs: Optional[Sequence[Axes]] = None,
):
    """Wrapper to compare coefficients given as dictionaries.

    Converts aligned coefficient dictionaries into dense matrices and then
    delegates to :func:`compare_coefficient_plots` for plotting.

    This assumes that the coefficient dictionaries are aligned, i.e., that they
    contain the same keys across all coordinates, as produced by
    ``unionize_coeff_dicts()``.
    """

    true_mat, inferred_feature_names = _coeff_dicts_to_matrix(coefficients_true)
    est_mat, est_feature_names = _coeff_dicts_to_matrix(coefficients_est)

    if true_mat.shape != est_mat.shape:
        raise ValueError("True and estimated coefficient shapes do not match")

    if inferred_feature_names != est_feature_names:
        raise ValueError(
            "Feature names inferred from true and estimated coefficients do not match"
        )

    _compare_coefficient_plots_impl(
        est_mat,
        true_mat,
        input_features=input_features,
        feature_names=inferred_feature_names,
        scaling=scaling,
        axs=axs,
    )


def _plot_training_trajectory(
    ax: Axes,
    x_train: np.ndarray,
    x_true: np.ndarray,
    x_smooth: np.ndarray | None,
    labels: bool = True,
) -> None:
    """Plot a single training trajectory

    If x_smooth is provided, it is only plotted if sufficiently different
    from x_train.
    """
    if x_train.shape[1] == 2:
        ax.plot(
            x_true[:, 0], x_true[:, 1], ".", label="True", color=COLOR.TRUE, **PLOT_KWS
        )
        ax.plot(
            x_train[:, 0],
            x_train[:, 1],
            ".",
            label="Measured",
            color=COLOR.MEAS,
            **PLOT_KWS,
        )
        if (
            x_smooth is not None
            and np.linalg.norm(x_smooth - x_train) / np.linalg.norm(x_train) > 1e-12
        ):
            ax.plot(
                x_smooth[:, 0],
                x_smooth[:, 1],
                ".",
                label="Smoothed",
                color=COLOR.EST,
                **PLOT_KWS,
            )
        if labels:
            ax.set(xlabel="$x_0$", ylabel="$x_1$")
        else:
            ax.set(xticks=[], yticks=[])
    elif x_train.shape[1] == 3:
        ax.plot(
            x_true[:, 0],
            x_true[:, 1],
            x_true[:, 2],
            color=COLOR.TRUE,
            label="True values",
            **PLOT_KWS,
        )

        ax.plot(
            x_train[:, 0],
            x_train[:, 1],
            x_train[:, 2],
            ".",
            color=COLOR.MEAS,
            label="Measured values",
            alpha=0.3,
        )
        if (
            x_smooth is not None
            and np.linalg.norm(x_smooth - x_train) / x_smooth.size > 1e-12
        ):
            ax.plot(
                x_smooth[:, 0],
                x_smooth[:, 1],
                x_smooth[:, 2],
                ".",
                color=COLOR.EST,
                label="Smoothed values",
                alpha=0.3,
            )
        if labels:
            ax.set(xlabel="$x$", ylabel="$y$", zlabel="$z$")
        else:
            ax.set(xticks=[], yticks=[], zticks=[])
    else:
        raise ValueError("Can only plot 2d or 3d data.")


def plot_training_data(
    t_train: np.ndarray,
    x_train: np.ndarray,
    x_true: np.ndarray,
    x_smooth: np.ndarray | None = None,
    coord_names: Optional[Sequence[str]] = None,
) -> tuple[Figure, Figure]:
    """Plot training data (and smoothed training data, if different)."""
    if coord_names is None:
        coord_names = [f"$x_{i}$" for i in range(x_true.shape[1])]

    fig_composite = plt.figure(figsize=(12, 6))
    if x_train.shape[-1] == 2:
        ax_traj = fig_composite.add_subplot(1, 2, 1)
    elif x_train.shape[-1] == 3:
        ax_traj = fig_composite.add_subplot(1, 2, 1, projection="3d")
    else:
        raise ValueError("Too many or too few coordinates to plot")
    _plot_training_trajectory(ax_traj, x_train, x_true, x_smooth)
    ax_traj.legend()
    ax_traj.set(title="Trajectory Plot")
    ax_psd = fig_composite.add_subplot(1, 2, 2)
    _plot_data_psd(ax_psd, x_train, coord_names, traj_type="train")
    _plot_data_psd(ax_psd, x_true, coord_names, traj_type="true")
    ax_psd.set(title="Absolute Spectral Density")

    n_coord = x_true.shape[-1]
    fig_by_coord_1d = plt.figure(figsize=(n_coord * 4, 6))
    for coord_ind, cname in enumerate(coord_names):
        ax = fig_by_coord_1d.add_subplot(n_coord, 1, coord_ind + 1)
        plot_training_1d(ax, coord_ind, t_train, x_train, x_true, x_smooth, cname)

    fig_by_coord_1d.axes[-1].legend()

    return fig_composite, fig_by_coord_1d


def _plot_data_psd(
    ax: Axes,
    x: np.ndarray,
    coord_names: Sequence[str],
    traj_type: str = "train",
):
    """Plot the power spectral density of training data."""
    if traj_type == "train":
        color = COLOR.MEAS
    elif traj_type == "sim":
        color = COLOR.SIM
    elif traj_type == "true":
        color = COLOR.TRUE
    elif traj_type == "smooth":
        color = COLOR.EST
    else:
        raise ValueError(f"Unknown traj_type '{traj_type}'")
    coord_names = [name + f" {traj_type}" for name in coord_names]
    for coord, series in zip(coord_names, x.T):
        ax.loglog(
            np.abs(scipy.fft.rfft(series)) / np.sqrt(len(series)),
            color=color,
            label=coord,
        )
    ax.legend()
    ax.set(xlabel="Wavenumber")
    ax.set(ylabel="Magnitude")


def plot_training_1d(
    ax: Axes,
    coord_ind: int,
    t_train: np.ndarray,
    x_train: np.ndarray,
    x_true: np.ndarray,
    x_smooth: Optional[np.ndarray],
    coord_name: str,
):
    ax.plot(t_train, x_train[..., coord_ind], ".", color=COLOR.MEAS, label="measured")
    ax.plot(t_train, x_true[..., coord_ind], "-", color=COLOR.TRUE, label="true")
    if x_smooth is not None:
        ax.plot(t_train, x_smooth[..., coord_ind], color=COLOR.EST, label="smoothed")
    ax.set(xlabel="t", ylabel=coord_name)


def plot_pde_training_data(last_train, last_train_true, smoothed_last_train):
    """Plot training data (and smoothed training data, if different)."""
    # 1D:
    if len(last_train.shape) == 3:
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        axs[0].imshow(last_train_true, vmin=0, vmax=last_train_true.max())
        axs[0].set(title="True Data")
        axs[1].imshow(last_train, vmin=0, vmax=last_train_true.max())
        axs[1].set(title="Noisy Data")
        axs[2].imshow(smoothed_last_train, vmin=0, vmax=last_train_true.max())
        axs[2].set(title="Smoothed Data")
        return plt.show()


def plot_test_sim_data_1d_panel(
    axs: Sequence[Axes],
    x_true: Optional[np.ndarray],
    x_sim: np.ndarray,
    t_test: np.ndarray,
    t_sim: np.ndarray,
    coord_names: Sequence[str],
) -> None:
    for ordinate, ax in enumerate(axs):
        if x_true is not None:
            ax.plot(t_test, x_true[:, ordinate], color=COLOR.TRUE, label="True")
        axs[ordinate].plot(
            t_sim, x_sim[:, ordinate], "--", color=COLOR.SIM, label="Simulation"
        )
        axs[ordinate].set(xlabel="t", ylabel=coord_names[ordinate])


def _plot_test_sim_data_2d(
    ax: Axes,
    x_true: Optional[np.ndarray],
    x_sim: np.ndarray,
    labels: bool,
    coord_names: Sequence[str],
) -> None:
    if x_true is not None:
        ax.plot(x_true[:, 0], x_true[:, 1], color=COLOR.TRUE, label="True Values")
    ax.plot(x_sim[:, 0], x_sim[:, 1], "--", color=COLOR.SIM, label="Simulation")
    if labels:
        ax.set(xlabel=coord_names[0], ylabel=coord_names[1])
    else:
        ax.set(xticks=[], yticks=[])


def _plot_test_sim_data_3d(
    ax: Axes, x_vals: np.ndarray, label: Optional[str], coord_names: Sequence[str]
):
    if label == "True":
        color = COLOR.TRUE
    elif label == "Simulation":
        color = COLOR.SIM
    else:
        color = None
    ax.plot(x_vals[:, 0], x_vals[:, 1], x_vals[:, 2], color=color, label=label)
    if label:
        ax.set(xlabel=coord_names[0], ylabel=coord_names[1], zlabel=coord_names[2])
    else:
        ax.set(xticks=[], yticks=[], zticks=[])


def plot_test_trajectory(
    x_true: np.ndarray,
    x_sim: np.ndarray,
    t_test: np.ndarray,
    t_sim: np.ndarray,
    figs: Optional[tuple[Figure, Figure]] = None,
    coord_names: Optional[Sequence[str]] = None,
) -> tuple[Figure, Figure]:
    """Plot a test trajectory

    Args:
        last_test: a single trajectory of the system
        model: a trained model to simulate and compare to test data
        dt: the time interval in test data

    Returns:
        The sequence of axes used for the single-dimension time-series plots.
        If ``axs`` is provided, the same sequence is returned.
    """
    if coord_names is None:
        coord_names = [f"$x_{i}$" for i in range(x_true.shape[1])]
    if not figs:
        fig_by_coord_1d, axs_by_coord = plt.subplots(
            x_true.shape[1], 1, sharex=True, figsize=(7, 9)
        )
        if x_true.shape[1] == 2:
            fig_composite, axs_composite = plt.subplots(1, 2, figsize=(10, 4.5))
        elif x_true.shape[1] == 3:
            fig_composite, axs_composite = plt.subplots(
                1, 2, figsize=(10, 4.5), subplot_kw={"projection": "3d"}
            )
        else:
            raise ValueError("Can only plot 2d or 3d data.")
    else:
        fig_composite, fig_by_coord_1d = figs
        axs_composite = fig_composite.axes
        axs_by_coord = fig_by_coord_1d.axes

    assert isinstance(axs_composite, list)
    assert isinstance(axs_by_coord, list)
    plot_test_sim_data_1d_panel(axs_by_coord, None, x_sim, t_test, t_sim, coord_names)
    axs_by_coord[-1].legend()
    if x_true.shape[1] == 2:
        _plot_test_sim_data_2d(
            axs_composite[0], None, x_sim, labels=True, coord_names=coord_names
        )
    elif x_true.shape[1] == 3:
        _plot_test_sim_data_3d(axs_composite[0], x_sim, "Simulation", coord_names)
    axs_composite[0].legend()
    _plot_data_psd(axs_composite[1], x_sim, coord_names, traj_type="sim")
    if not figs:
        fig_by_coord_1d.suptitle("Test Trajectories by Dimension")
        fig_composite.suptitle("Full Test Trajectories")
        axs_composite[0].set(title="true trajectory")
        axs_composite[0].set(title="model simulation")
    return fig_composite, fig_by_coord_1d
