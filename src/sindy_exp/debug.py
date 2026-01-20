from sindy_exp.data import gen_data

data_dict = gen_data(
    "hopf",
    seed=50,
    n_trajectories=1,
    ic_stdev=3,
    noise_rel=0.1,
    display=True,
    array_namespace="jax",
)
print(data_dict["data"].input_features)
