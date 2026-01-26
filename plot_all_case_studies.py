import numpy as np
import matplotlib.pyplot as plt
import os

from utils import plot_forecasts_shaded, plot_calibration
from energy_utils import get_energy_data, levels as energy_levels, ISO_to_sites
from covid_utils import get_covid_data, good_forecasters, levels as covid_levels

fig_folder = 'figs/case_studies/all'

def create_axes_with_square_right_columns(nrows, ncols, fig_width, fig_height):

    # make the two rightmost columns square (in inches)
    num_square_cols = 2
    desired_side = fig_height / nrows  # height per row => desired square side in inches
    total_square_width = num_square_cols * desired_side
    remaining = fig_width - total_square_width

    if remaining <= 0:
        # fallback to equal-width columns if figure too narrow
        width_ratios = [1] * ncols
    else:
        first_cols_width = remaining / (ncols - num_square_cols)
        width_ratios = [first_cols_width] * (ncols - num_square_cols) + [desired_side] * num_square_cols

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), 
                            gridspec_kw={'width_ratios': width_ratios,
                                         'hspace': 0.2 # subplot spacing
                                         })

    # enforce square display for the rightmost columns
    for i in range(nrows):
        for j in range(ncols - num_square_cols, ncols):
            axes[i, j].set_box_aspect(1)

    return fig, axes

# Dataset-agnostic core plotting function (renamed and accepts out_path)
def plot_multiple_forecasters(forecasters_chunk,          
                              data_loader,         # fn(forecaster) -> (Y, Yhat_base)
                              calib_loader,        # fn(forecaster) -> Yhat_calib
                              start_date_fn=None,  # optional fn(forecaster) -> start_date_str
                              levels=None,
                              ylabel='',
                              xlabel='Week',
                              xlims=None,
                              ylims=None,
                              fig_size=(12.5, 16),
                              save_path=None):
    nrows, ncols = len(forecasters_chunk), 4
    fig_width, fig_height = fig_size
    fig, axes = create_axes_with_square_right_columns(nrows, ncols, fig_width, fig_height)
    axes = np.atleast_2d(axes)  # ensure 2D indexing even if nrows == 1

    for i, forecaster in enumerate(forecasters_chunk):
        # start_date = start_date_fn(forecaster) if start_date_fn is not None else None

        # Load base forecasts
        Y, Yhat_base = data_loader(forecaster)

        # Load calibrated forecasts
        Yhat_calib = calib_loader(forecaster)

        # 1) Plot raw forecasts in first column
        ax = axes[i, 0]
        plot_forecasts_shaded(Y, Yhat_base, levels, ax=ax, title=None, plot_Y=True,
            linewidth=0.8, base_color="tab:red", alpha_min=0.05, alpha_max=0.4,
            forecast_label='Raw forecasts', xlabel='', ylabel=ylabel, 
            legend_loc='upper left' if i == 0 else None)
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.xaxis.grid(False)
        ax.set_xticks([])

        # 2) Plot calibrated forecasts in second column
        ax = axes[i, 1]
        plot_forecasts_shaded(Y, Yhat_calib, levels, ax=ax, title=None, plot_Y=True,
            linewidth=0.8, base_color="tab:blue", alpha_min=0.05, alpha_max=0.4,
            forecast_label='MultiQT forecasts', xlabel='', ylabel='', legend_loc='upper left' if i == 0 else None)
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.xaxis.grid(False)
        ax.set_xticks([])
        ax.set_yticklabels([])

        # Set raw forecast ylims to be the same as calibrated forecast ylims
        if ylims is None:
            axes[i, 0].set_ylim(axes[i,1].get_ylim())

        # 3) Plot raw calibration in third column
        ax = axes[i, 2]
        plot_calibration(Y, Yhat_base, ax=ax, color='tab:red', title=None,
                        single_state=True, plot_yequalsx=True, label='', alpha=1, levels=levels)

        # 4) Plot calibrated calibration in fourth column
        ax = axes[i, 3]
        plot_calibration(Y, Yhat_calib, ax=ax, color='tab:blue', title=None,
                            single_state=True, plot_yequalsx=True, label='', alpha=1, levels=levels)

        # Remove axis labels for calibration plots
        axes[i, 2].set_xlabel('')
        axes[i, 2].set_ylabel('')
        axes[i, 3].set_xlabel('')
        axes[i, 3].set_ylabel('')

        # Put forecaster name as a vertical label at left of the row
        axes[i, 0].text(-0.3, 0.5, forecaster, transform=axes[i, 0].transAxes,
                        rotation=90, va='center', fontsize=10)

    # Set x-labels on last row only
    axes[-1, 0].set_xlabel(xlabel)
    axes[-1, 1].set_xlabel(xlabel)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)


## ========================= All COVID forecast plots =========================
def plot_covid_case_studies(state,
                            horizon,
                            forecasters=good_forecasters,
                            dataset_name='covid',
                            Yhat_type='quantile-specific',
                            levels=covid_levels,
                            fig_width=12.5,
                            fig_height=16):
    # make fig folder for this dataset/state/horizon
    fig_prefix_local = f'{fig_folder}/{dataset_name}/{state}_h={horizon}'
    os.makedirs(fig_prefix_local, exist_ok=True)

    # data loader bound to state & horizon
    def data_loader(forecaster):
        return get_covid_data(forecaster, state, horizon, Yhat_type=Yhat_type, levels=levels)

    # calibration loader bound to state & horizon and naming convention
    def calib_loader(forecaster):
        path = f"cache/{dataset_name}/{forecaster}_{state}_Yhat={Yhat_type}_h={horizon}.npy"
        return np.load(path)

    # optional start date function (kept for possible date-axis labeling)
    def start_date_fn(forecaster):
        return 0

    # Split and plot in two chunks to avoid overly tall single figure
    total = len(forecasters)
    half = (total + 1) // 2
    first_half = forecasters[:half]
    second_half = forecasters[half:]

    plot_multiple_forecasters(first_half, 
                              data_loader, calib_loader, start_date_fn,
                              levels=levels, ylabel='Deaths',
                              xlabel='Week', fig_size=(fig_width, fig_height),
                              save_path=f'{fig_prefix_local}/all_forecasters_{state}_h={horizon}_ALL_plots_pt1.pdf')

    if second_half:
        plot_multiple_forecasters(second_half, 
                                  data_loader, calib_loader, start_date_fn,
                                  levels=levels, ylabel='Deaths',
                                  xlabel='Week', fig_size=(fig_width, fig_height),
                                  save_path=f'{fig_prefix_local}/all_forecasters_{state}_h={horizon}_ALL_plots_pt2.pdf')


## ========================= All energy forecast plots =========================

# Sample 8 sites from ERCOT Wind and Solar and make plots
def sample_sites_for_target(ISO, target_variable, n=8, seed=0):
    """Return a reproducible random sample of up to `n` sites for the given ISO and target."""
    sites = ISO_to_sites.get(ISO, {}).get(target_variable, [])
    if len(sites) == 0:
        return []
    rng = np.random.default_rng(seed)
    if len(sites) <= n:
        return list(sites)
    return list(rng.choice(sites, size=n, replace=False))


def plot_energy_sample_sites(ISO='ERCOT', target_variable='Wind', hour=16, n_sites=8,
                             fig_root=fig_folder, levels=energy_levels, seed=0):
    """Sample sites and produce forecast + calibration plots for each site x hour.

    For each sampled site and each hour in `hours` this saves:
      - raw forecast shaded plot
      - calibrated forecast shaded plot (if cache file exists)
      - calibration plot for raw forecasts
      - calibration plot for calibrated forecasts (if available)

    Plots are saved under `{fig_root}/energy/{target_variable}/{site}/`.
    """
    np.random.seed(seed)
    sampled_sites = np.random.choice(list(ISO_to_sites[ISO][target_variable]), size=n_sites, replace=False)
    print(f'Sampled sites for {ISO} {target_variable} at hour {hour}: {sampled_sites}')

    # Create data loaders
    data_loader = lambda site: get_energy_data(ISO, site, hour, 
                                               target_variable=target_variable,
                                               Yhat_type='quantile-specific')
    calib_loader = lambda site: np.load(
        f"cache/energy/{ISO}_{site}_{target_variable}_Yhat=quantile-specific_hour={hour}.npy")

    fig_prefix_local = f'{fig_root}/energy/hour={hour}'
    os.makedirs(fig_prefix_local, exist_ok=True)

    plot_multiple_forecasters(
            sampled_sites,  # single site as forecaster
            data_loader, calib_loader, 
            levels=levels, ylabel=f'Power (megawatts)',
            xlabel='Day', fig_size=(12.5, 16),
            xlims=(243, 303), # Sept 1 - Oct 31
            save_path=f'{fig_prefix_local}/{target_variable}_hour={hour}_{n_sites}SAMPLES_plots.pdf'
        )
       
# # PLOT ALL CASE STUDIES FOR COVID
# if __name__ == "__main__":

#     from covid_utils import states

#     ## 1) COVID
#     dataset = 'covid'

#     for state in states:
#         for horizon in [1, 2,3, 4]:
            
#             # Make fig folder
#             fig_prefix = f'{fig_folder}/{dataset}/{state}_h={horizon}'
#             os.makedirs(fig_prefix, exist_ok=True)

#             plot_covid_case_studies(state, horizon)
#             print(f'Done plotting all COVID case studies for {state} with horizon {horizon}.')


# ORIGINAL
if __name__ == "__main__":

    ## 1) COVID
    # ------- COVID plot settings -------
    states = ['ca', 'vt']
    horizon = 1
    # -----------------------------------

    for state in states:

        dataset = 'covid'
        # Make fig folder
        fig_prefix = f'{fig_folder}/{dataset}/{state}_h={horizon}'
        os.makedirs(fig_prefix, exist_ok=True)

        plot_covid_case_studies(state, horizon)
        print(f'Done plotting all COVID case studies for {state} with horizon {horizon}.')

    ## 2) Energy
    # Run sampling & plotting for ERCOT Wind and Solar (8 sites each)
    print('Sampling 8 sites for ERCOT Wind and Solar and creating plots...')
    plot_energy_sample_sites(ISO='ERCOT', target_variable='Wind', hour=16, n_sites=8, seed=2)
    plot_energy_sample_sites(ISO='ERCOT', target_variable='Solar', hour=16, n_sites=8, seed=0)
    print(f'Done with energy forecast plotting for random sample.')


