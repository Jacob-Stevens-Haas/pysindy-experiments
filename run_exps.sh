    # --param other_params=lor-ross-cubic \
nohup python -m gen_experiments gridsearch --seed 19 \
    --param ex_name=lorenz \
    --param metrics=all \
    --param other_params=lor-ross-cubic \
    --param grid_params=duration-absnoise \
    --param grid_vals=duration-absnoise \
    --param grid_decisions=plot2 \
    --param series_params=kalman-auto3 \
    --param plot_prefs=test-absrel5 \
    --param skinny_specs=duration-noise &> lorenz.log &

nohup python -m gen_experiments gridsearch --seed 19 \
    --param ex_name=cubic_ho \
    --param metrics=all \
    --param other_params=4nonzero-cubic \
    --param grid_params=duration-absnoise \
    --param grid_vals=duration-absnoise \
    --param grid_decisions=plot2 \
    --param series_params=kalman-auto3 \
    --param plot_prefs=test-absrel5 \
    --param skinny_specs=duration-noise &> cubic_ho.log &

nohup python -m gen_experiments gridsearch --seed 19 \
    --param ex_name=sho \
    --param metrics=all \
    --param other_params=4nonzero-cubic \
    --param grid_params=duration-absnoise \
    --param grid_vals=duration-absnoise \
    --param grid_decisions=plot2 \
    --param series_params=kalman-auto3 \
    --param plot_prefs=test-absrel5 \
    --param skinny_specs=duration-noise &> sho.log &

nohup python -m gen_experiments gridsearch --seed 19 \
    --param ex_name=vdp \
    --param metrics=all \
    --param other_params=4nonzero-cubic \
    --param grid_params=duration-absnoise \
    --param grid_vals=duration-absnoise \
    --param grid_decisions=plot2 \
    --param series_params=kalman-auto3 \
    --param plot_prefs=test-absrel5 \
    --param skinny_specs=duration-noise &> vdp.log &

nohup python -m gen_experiments gridsearch --seed 19 \
    --param ex_name=lv \
    --param metrics=all \
    --param other_params=4nonzero-cubic \
    --param grid_params=duration-absnoise \
    --param grid_vals=duration-absnoise \
    --param grid_decisions=plot2 \
    --param series_params=kalman-auto3 \
    --param plot_prefs=test-absrel5 \
    --param skinny_specs=duration-noise &> lv.log &

nohup python -m gen_experiments gridsearch --seed 19 \
    --param ex_name=duff \
    --param metrics=all \
    --param other_params=4nonzero-cubic \
    --param grid_params=duration-absnoise \
    --param grid_vals=duration-absnoise \
    --param grid_decisions=plot2 \
    --param series_params=kalman-auto3 \
    --param plot_prefs=test-absrel5 \
    --param skinny_specs=duration-noise &> duff.log &

nohup python -m gen_experiments gridsearch --seed 19 \
    --param ex_name=hopf \
    --param metrics=all \
    --param other_params=hopf-cubic \
    --param grid_params=duration-absnoise \
    --param grid_vals=duration-absnoise \
    --param grid_decisions=plot2 \
    --param series_params=kalman-auto3 \
    --param plot_prefs=test-absrel5 \
    --param skinny_specs=duration-noise &> hopf.log &

nohup python -m gen_experiments gridsearch --seed 19 \
    --param ex_name=ross \
    --param metrics=all \
    --param other_params=lor-ross-cubic \
    --param grid_params=duration-absnoise \
    --param grid_vals=duration-absnoise \
    --param grid_decisions=plot2 \
    --param series_params=kalman-auto3 \
    --param plot_prefs=test-absrel5 \
    --param skinny_specs=duration-noise &> ross.log &

# nohup python -m gen_experiments gridsearch --seed 19 --debug \
#     --param ex_name=ross \
#     --param metrics=all \
#     --param other_params=debug \
#     --param grid_params=duration-absnoise \
#     --param grid_vals=debug \
#     --param grid_decisions=plot2 \
#     --param series_params=kalman-auto3 \
#     --param plot_prefs=test-absrel5 \
#     &> ross.log &
