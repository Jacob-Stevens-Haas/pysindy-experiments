import pysindy as ps

from .utils import (
    gen_data,
    compare_coefficient_plots,
    coeff_metrics,
    integration_metrics,
    unionize_coeff_matrices,
    _make_model
)

name = "LORENZ"


def run(
    seed: float,
    /,
    sim_params: dict,
    diff_params: dict,
    feat_params: dict,
    opt_params: dict,
    display: bool = True,
) -> dict:
    dt, t_train, x_train, x_test, x_dot_test = gen_data(
        ps.utils.lorenz, 3, seed, **sim_params
    )
    input_features = ["x", "y", "z"]
    model = _make_model(input_features, dt, diff_params, feat_params, opt_params)

    model.fit(x_train, quiet=True, multiple_trajectories=True)
    coeff_true = [
        {"x": -10, "y": 10},
        {"x": 28, "y": -1, "x z": -1},
        {"z": -8 / 3, "x y": 1},
    ]
    coeff_true, coefficients, feature_names = unionize_coeff_matrices(model, coeff_true)

    # make the plots
    if display:
        model.print()
        compare_coefficient_plots(
            coefficients,
            coeff_true,
            input_features=input_features,
            feature_names=feature_names,
        )

    # calculate metrics
    metrics = coeff_metrics(coefficients, coeff_true)
    metrics.update(integration_metrics(model, x_test, t_train, x_dot_test))
    return metrics


if __name__ == "__main__":
    run(seed=1, diff_params={"kind": "FiniteDifference"}, opt_params={"kind": "stlsq"})

sim_params = {"test": {"n_trajectories": 2}}
diff_params = {"test": {"kind": "FiniteDifference"}}
feat_params = {"test": {"kind": "Polynomial"}, "test2": {"kind": "Fourier"}}
opt_params = {"test": {"kind": "STLSQ"}}
