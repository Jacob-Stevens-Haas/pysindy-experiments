from copy import copy
from functools import partial
from logging import getLogger
from pprint import pformat
from typing import Annotated, Callable, Iterable, Optional, Sequence, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import sindy_exp
from numpy.typing import DTypeLike, NDArray
from scipy.stats import kstest
from sindy_exp import config
from sindy_exp.odes import plot_ode_panel
from sindy_exp.utils import (
    GridsearchResult,
    GridsearchResultDetails,
    NestedDict,
    SavedData,
    SeriesDef,
    SeriesList,
    TrialData,
    _amax_to_full_inds,
    _argopt,
    _grid_locator_match,
    _PlotPrefs,
    simulate_test_data,
)

pformat = partial(pformat, indent=4, sort_dicts=True)
logger = getLogger(__name__)
name = "gridsearch"
lookup_dict = vars(config)

OtherSliceDef = tuple[int | Callable]
SkinnySpecs = Optional[tuple[tuple[str, ...], tuple[OtherSliceDef, ...]]]


def run(
    seed: int,
    group: str,
    grid_params: list[str],
    grid_vals: list[Sequence],
    grid_decisions: Sequence[str],
    other_params: dict,
    series_params: Optional[SeriesList] = None,
    metrics: Optional[Sequence[str]] = None,
    plot_prefs: _PlotPrefs = _PlotPrefs(True, False, ()),
    skinny_specs: SkinnySpecs = None,
) -> GridsearchResultDetails:
    """Run a grid-search wrapper of an experiment.

    Arguments:
        group: an experiment registered in gen_experiments.  It must
            have a name and a metric_ordering attribute
        grid_params: kwarg names to grid and pass to
            experiment
        grid_vals: kwarg values to grid.  Indices match grid_params
        grid_decisions: What to do with each grid param, e.g.
            {"plot", "best"}.  Indices match grid_params.
        other_params: a dict of other kwargs to pass to experiment
        metrics: names of metrics to record from each wrapped experiment
        plot_prefs: whether to plot results, and if so, a function to
            intercept and modify plot data.  Use this for applying any
            scaling or conversions.
        skinny_specs: Allow only conducting some of the grid search,
            where axes are all searched, but not all combinations are
            searched.  The first element is a sequence of grid_names to
            skinnify.  The second is the thin_slices criteria (see
            docstring for _ndindex_skinny).  By default, all plot axes
            are made skinny with respect to each other.
    """
    other_params = NestedDict(**other_params)
    base_ex, base_group = sindy_exp.experiments[group]
    if series_params is None:
        series_params = SeriesList(None, None, [SeriesDef(group, {}, [], [])])
        legends = False
    else:
        legends = True
    n_metrics = len(metrics)
    metric_ordering = [base_ex.metric_ordering[metric] for metric in metrics]
    n_plotparams = len([decide for decide in grid_decisions if decide == "plot"])
    series_searches: list[tuple[list[GridsearchResult], list[GridsearchResult]]] = []
    intermediate_data: list[SavedData] = []
    plot_data: list[SavedData] = []
    if base_group is not None:
        other_params["group"] = base_group
    for s_counter, series_data in enumerate(series_params.series_list):
        curr_other_params = copy(other_params)
        if series_params.param_name is not None:
            curr_other_params[series_params.param_name] = series_data.static_param
        new_grid_vals: list = grid_vals + series_data.grid_vals
        new_grid_params = grid_params + series_data.grid_params
        new_grid_decisions = grid_decisions + len(series_data.grid_params) * ["best"]
        if skinny_specs is not None:
            ind_skinny, where_others = _curr_skinny_specs(skinny_specs, new_grid_params)
        else:
            ind_skinny = [
                ind for ind, decide in enumerate(new_grid_decisions) if decide == "plot"
            ]
            where_others = None
        full_results_shape = (len(metrics), *(len(grid) for grid in new_grid_vals))
        full_results = np.full(full_results_shape, np.nan)
        gridpoint_selector = _ndindex_skinny(
            full_results_shape[1:], ind_skinny, where_others
        )
        rng = np.random.default_rng(seed)
        for ind_counter, ind in enumerate(gridpoint_selector):
            print(f"Calculating series {s_counter}, gridpoint{ind_counter}", end="\r")
            new_seed = rng.integers(1000)
            param_updates = {}
            for axis_ind, key, val_list in zip(ind, new_grid_params, new_grid_vals):
                param_updates[key] = val_list[axis_ind]
                curr_other_params.update(param_updates)
            curr_results, grid_data = base_ex.run(
                new_seed, **curr_other_params, display=False, return_all=True
            )
            grid_data: TrialData
            intermediate_data.append(
                {"params": curr_other_params.flatten(), "pind": ind, "data": grid_data}
            )
            full_results[(slice(None), *ind)] = [
                curr_results[metric] for metric in metrics
            ]
        grid_optima, grid_ind = _marginalize_grid_views(
            new_grid_decisions, full_results, metric_ordering
        )
        series_searches.append((grid_optima, grid_ind))

    if plot_prefs:
        full_m_inds = _amax_to_full_inds(
            plot_prefs.grid_ind_match, [s[1] for s in series_searches]
        )
        for int_data in intermediate_data:
            logger.debug(
                f"Checking whether to save/plot :\n{pformat(int_data['params'])}\n"
                f"\tat location {pformat(int_data['pind'])}\n"
                f"\tagainst spec: {pformat(plot_prefs.grid_params_match)}\n"
                f"\twith allowed locations {pformat(full_m_inds)}"
            )
            if _grid_locator_match(
                int_data["params"],
                int_data["pind"],
                plot_prefs.grid_params_match,
                full_m_inds,
            ) and int_data["params"] not in [saved["params"] for saved in plot_data]:
                grid_data = int_data["data"]
                print("Results for params: ", int_data["params"], flush=True)
                grid_data |= simulate_test_data(
                    grid_data["model"], grid_data["dt"], grid_data["x_test"]
                )
                logger.info("Found match, simulating and plotting")
                plot_ode_panel(grid_data)
                plot_data.append(int_data)
        if plot_prefs.rel_noise:
            grid_vals, grid_params = plot_prefs.rel_noise(
                grid_vals, grid_params, grid_data
            )
        fig, subplots = plt.subplots(
            n_metrics,
            n_plotparams,
            sharey="row",
            sharex="col",
            squeeze=False,
            figsize=(n_plotparams * 3, 0.5 + n_metrics * 2.25),
        )
        for series_data, series_name in zip(
            series_searches, (ser.name for ser in series_params.series_list)
        ):
            plot(
                fig,
                subplots,
                metrics,
                grid_params,
                grid_vals,
                series_data[0],
                series_name,
                legends,
            )
        if series_params.print_name is not None:
            title = f"Grid Search on {series_params.print_name} in {group}"
        else:
            title = f"Grid Search in {group}"
        fig.suptitle(title)
        fig.tight_layout()

    main_metric_ind = metrics.index("main") if "main" in metrics else 0
    return {
        "system": group,
        "plot_data": plot_data,
        "series_data": {
            name: data
            for data, name in zip(
                [list(zip(metrics, argopts)) for metrics, argopts in series_searches],
                [ser.name for ser in series_params.series_list],
            )
        },
        "metrics": metrics,
        "grid_params": grid_params,
        "grid_vals": grid_vals,
        "main": max(
            grid[main_metric_ind].max()
            for metrics, _ in series_searches
            for grid in metrics
        ),
    }


def plot(
    fig: plt.Figure,
    subplots: Sequence[plt.Axes],
    metrics: Sequence[str],
    grid_params: Sequence[str],
    grid_vals: Sequence[Sequence[float] | np.ndarray],
    grid_searches: Sequence[GridsearchResult],
    name: str,
    legends: bool,
):
    for m_ind_row, m_name in enumerate(metrics):
        for col, (param_name, x_ticks, param_search) in enumerate(
            zip(grid_params, grid_vals, grid_searches)
        ):
            ax = subplots[m_ind_row, col]
            ax.plot(x_ticks, param_search[m_ind_row], label=name)
            x_ticks = np.array(x_ticks)
            if m_name in ("coeff_mse", "coeff_mae"):
                ax.set_yscale("log")
            x_ticks_normalized = (x_ticks - x_ticks.min()) / (
                x_ticks.max() - x_ticks.min()
            )
            x_ticks_lognormalized = (np.log(x_ticks) - np.log(x_ticks).min()) / (
                np.log(x_ticks.max()) - np.log(x_ticks).min()
            )
            ax = subplots[m_ind_row, col]
            if kstest(x_ticks_normalized, "uniform") < kstest(
                x_ticks_lognormalized, "uniform"
            ):
                ax.set_xscale("log")
            if m_ind_row == 0:
                ax.set_title(f"{param_name}")
            if col == 0:
                ax.set_ylabel(f"{m_name}")
    if legends:
        ax.legend()


T = TypeVar("T", bound=np.generic)


def _marginalize_grid_views(
    grid_decisions: Iterable[str],
    results: Annotated[NDArray[T], "shape (n_metrics, *n_gridsearch_values)"],
    max_or_min: Sequence[str] = None,
) -> tuple[list[GridsearchResult[T]], list[GridsearchResult]]:
    """Marginalize unnecessary dimensions by taking max across axes.

    Ignores NaN values
    Args:
        grid_decisions: list of how to treat each non-metric gridsearch
            axis.  An array of metrics for each "plot" grid decision
            will be returned, along with an array of the the index
            of collapsed dimensions that returns that metric
        results: An array of shape (n_metrics, *n_gridsearch_values)
        max_or_min: either "max" or "min" for each row of results
    Returns:
        a list of the metric optima for each plottable grid decision, and
        a list of the flattened argoptima.
    """
    arg_dtype: DTypeLike = ",".join(results.ndim * "i")
    plot_param_inds = [ind for ind, val in enumerate(grid_decisions) if val == "plot"]
    grid_searches = []
    args_maxes = []
    optfuns = [np.nanmax if opt == "max" else np.nanmin for opt in max_or_min]
    for param_ind in plot_param_inds:
        reduce_axes = tuple(set(range(results.ndim - 1)) - {param_ind})
        selection_results = np.array(
            [opt(result, axis=reduce_axes) for opt, result in zip(optfuns, results)]
        )
        sub_arrs = []
        for m_ind, (result, opt) in enumerate(zip(results, max_or_min)):
            pad_m_ind = np.vectorize(lambda tp: np.void((m_ind, *tp), dtype=arg_dtype))
            arg_max = pad_m_ind(_argopt(result, reduce_axes, opt))
            sub_arrs.append(arg_max)

        args_max = np.stack(sub_arrs)
        grid_searches.append(selection_results)
        args_maxes.append(args_max)
    return grid_searches, args_maxes


def _ndindex_skinny(
    shape: tuple[int],
    thin_axes: Optional[Sequence[int]] = None,
    thin_slices: Optional[Sequence[OtherSliceDef]] = None,
):
    """
    Return an iterator like ndindex, but only traverse thin_axes once

    This is useful for grid searches with multiple plot axes, where
    searching across all combinations of plot axes is undesirable.
    Slow for big arrays! (But still probably trivial compared to the
    gridsearch operation :))

    Args:
        shape: array shape
        thin_axes: axes for which you don't want the product of all
            indexes
        thin_slices: the indexes for other thin axes when traversing
            a particular thin axis. Defaults to 0th index

    Example:

    >>> set(_ndindex_skinny((2,2), (0,1), ((0,), (lambda x: x,))))

    {(0, 0), (0, 1), (1, 1)}
    """
    if thin_axes is None and thin_slices is None:
        thin_axes = ()
        thin_slices = ()
    elif thin_axes is None:
        raise ValueError("Must pass thin_axes if thin_slices is not None")
    elif thin_slices is None:  # slice other thin axes at 0th index
        n_thin = len(thin_axes)
        thin_slices = n_thin * ((n_thin - 1) * (0,),)
    full_indexes = np.ndindex(shape)

    def ind_checker(multi_index):
        """Check if a multi_index meets thin index criteria"""
        matches = []
        # check whether multi_index matches criteria of any thin_axis
        for ax1, where_others in zip(thin_axes, thin_slices, strict=True):
            other_axes = list(thin_axes)
            other_axes.remove(ax1)
            match = True
            # check whether multi_index meets criteria of a particular thin_axis
            for ax2, slice_ind in zip(other_axes, where_others, strict=True):
                if callable(slice_ind):
                    slice_ind = slice_ind(multi_index[ax1])
                # would check: "== slice_ind", but must allow slice_ind = -1
                match *= multi_index[ax2] == range(shape[ax2])[slice_ind]
            matches.append(match)
        return any(matches)

    while True:
        try:
            ind = next(full_indexes)
        except StopIteration:
            break
        if ind_checker(ind):
            yield ind


def _curr_skinny_specs(
    skinny_specs: SkinnySpecs, grid_params: list[str]
) -> tuple[Sequence[int], Sequence[OtherSliceDef]]:
    """Calculate which skinny specs apply to current parameters"""
    skinny_param_inds = [
        grid_params.index(pname) for pname in skinny_specs[0] if pname in grid_params
    ]
    missing_sk_inds = [
        skinny_specs[0].index(pname)
        for pname in skinny_specs[0]
        if pname not in grid_params
    ]
    where_others = []
    for orig_sk_ind, match_criteria in zip(
        range(len(skinny_specs[0])), skinny_specs[1], strict=True
    ):
        if orig_sk_ind in missing_sk_inds:
            continue
        missing_criterion_inds = tuple(
            sk_ind if sk_ind < orig_sk_ind else sk_ind - 1 for sk_ind in missing_sk_inds
        )
        new_criteria = tuple(
            match_criterion
            for cr_ind, match_criterion in enumerate(match_criteria)
            if cr_ind not in missing_criterion_inds
        )
        where_others.append(new_criteria)
    return skinny_param_inds, tuple(where_others)
