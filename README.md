
This is the code release for [Calibrated Multi-Level Quantile Forecasting](https://arxiv.org/abs/2512.23671)

## Quick start

To run MultiQT on your own data, use the `multiQT()` function in `example.py`. 

## Code overview

* `example.py`: Start here if you want to apply MultiQT to your own data, as this file contains a self-contained implementation of the method. 
* `data/`: Data for the COVID experiments (data for the energy experiments is accessed directly via AWS).
* Everything else is code for reproducing the paper results.

## Reproducing paper results

1. Set up virtual environment

```
conda create --name multilevelqt
conda activate multilevelqt
conda install --yes --file requirements.txt
```

2. Run the following code

To generate the MultiQT-corrected forecasts, save them to `cache/`, and compute metrics:

```
python run_experiments.py --dataset covid
python run_experiments.py --dataset energy
python compute_metrics.py
```
With no parallelization, the COVID experiments (15 forecasters x 50 states x 4 horizons for >50 time steps at 23 quantile levels) take ~10 minutes to run. The energy experiments (2 targets x 490 sites x 6 hours for 365 time steps at 99 quantile levels) take ~140 minutes. 

To process the cached results and generate most of the paper's figures (saved to `figs/`):

```
python generate_figures.py
python plot_all_case_studies.py
```

To generate the individual case study plots (Figs 1 and 8 of the paper), run the notebook `case_studies.ipynb`. 


To generate the Quantile Tracker crossing fraction plot in the Appendix:

```
python QT_crossing_frac.py
```

## Citation

If you find this code useful, please use this citation:

```
@article{ding2025calibrated,
  title={Calibrated Multi-Level Quantile Forecasting},
  author={Ding, Tiffany and Gibbs, Isaac and Tibshirani, Ryan J.},
  journal={arXiv preprint arXiv:2512.23671},
  year={2025}
}
```

