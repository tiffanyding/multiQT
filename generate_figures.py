import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from matplotlib.patches import FancyArrowPatch

import pdb

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

def despine(ax, mode='all'):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if mode == 'all': 
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

def load_covid_metrics():
    """Load COVID metrics from CSV file."""
    metrics_path = Path("metrics/covid_metrics.csv")
    if not metrics_path.exists():
        print(f"Error: {metrics_path} not found!")
        return None
    
    df = pd.read_csv(metrics_path)
    print(f"Loaded COVID metrics: {len(df)} rows")
    return df

def load_energy_metrics():
    """Load Energy metrics from CSV file."""
    metrics_path = Path("metrics/energy_metrics.csv")
    if not metrics_path.exists():
        print(f"Error: {metrics_path} not found!")
        return None
    
    df = pd.read_csv(metrics_path)
    print(f"Loaded Energy metrics: {len(df)} rows")
    return df

# Helper function
def get_metric_names_and_labels(y_metric, x_metric):
     # Determine y-axis columns and label
    if y_metric == 'pit_entropy':
        raw_y_col = 'raw_pit_entropy'
        cal_y_col = 'cal_pit_entropy'
        y_label = 'PIT entropy'
    elif y_metric == 'avg_coverage_gap':
        raw_y_col = 'raw_avg_coverage_gap'
        cal_y_col = 'cal_avg_coverage_gap'
        y_label = 'Calibration error'
    else:
        raise ValueError(f"Unknown y_metric: {y_metric}")
    
    # Determine x-axis columns
    if x_metric == 'quantile_score':
        raw_x_col = 'raw_avg_quantile_score'
        cal_x_col = 'cal_avg_quantile_score'
        x_label = 'Quantile loss'
    elif x_metric == 'log_score':
        raw_x_col = 'raw_avg_log_score'
        cal_x_col = 'cal_avg_log_score'
        x_label = 'Average log score'
    else:
        raise ValueError(f"Unknown x_metric: {x_metric}")
    
    return raw_y_col, cal_y_col, y_label, raw_x_col, cal_x_col, x_label


def create_covid_arrow_plot_averaged(df, horizon, fig_folder, alpha=0.8, 
                                     ax=None, save=True, y_metric='pit_entropy',
                                     x_metric='quantile_score'):
    """Create arrow plot with metrics averaged across states for each forecaster."""
    # Filter data for this horizon
    horizon_data = df[df['horizon'] == horizon].copy()
    
    if len(horizon_data) == 0:
        print(f"No data for horizon {horizon}")
        return
    
    raw_y_col, cal_y_col, y_label, raw_x_col, cal_x_col, x_label = get_metric_names_and_labels(y_metric, x_metric)

     # Group by forecaster and average across states
    agg_dict = {
        raw_y_col: 'mean',
        cal_y_col: 'mean',
        raw_x_col: 'mean',
        cal_x_col: 'mean',
    }
    
    forecaster_metrics = horizon_data.groupby('forecaster').agg(agg_dict).reset_index()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    n_forecasters = len(forecaster_metrics)
    colors = plt.cm.tab20(np.linspace(0, 0.9, n_forecasters)) # can also replace tab20 with hsv
    
    for i, row in forecaster_metrics.iterrows():
        forecaster = row['forecaster']
        raw_qs = row[raw_x_col]
        raw_y = row[raw_y_col]
        cal_qs = row[cal_x_col]
        cal_y = row[cal_y_col]
        
        x1, y1 = raw_qs, raw_y
        x2, y2 = cal_qs, cal_y

        arrow = FancyArrowPatch(
                posA=(x1, y1),
                posB=(x2, y2),
                arrowstyle="Simple, tail_width=0.01, head_width=0.2, head_length=0.1",
                color=colors[i],
                alpha=alpha,
                mutation_scale=20,
                linewidth=1.5,
                label=forecaster
            )
        ax.add_patch(arrow)
    
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    
    # Set axis limits based on y_metric
    if y_metric == 'pit_entropy' and x_metric == 'quantile_score':
        ax.set_ylim(0.75, 1) # Standardize across horizons
        ax.set_xlim(10, 55)
    else:  # avg_coverage_gap
        ax.autoscale_view()

    ax.set_title(f'$h$={horizon}', fontsize=14)
    ax.grid(True, alpha=0.3)
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    # ax.legend(fontsize=8)
    
    if save:
        despine(ax)
        plt.tight_layout()
        plt.savefig(f'{fig_folder}/covid_arrow_averaged_horizon={horizon}_{y_metric}_{x_metric}.pdf', bbox_inches='tight')
        plt.close()


def create_arrow_plot(df, filter_col, filter_val, fig_folder, alpha=0.2, 
                      title_prefix='', filename_prefix='', dataset=None, 
                      y_metric='pit_entropy', x_metric='quantile_score',
                      ax=None, save=True):
    """Create arrow plot (NOT differences)."""
    # Filter data for this condition
    filtered_data = df[df[filter_col] == filter_val].copy()
    
    if len(filtered_data) == 0:
        print(f"No data for {filter_col} {filter_val}")
        return
    
    raw_y_col, cal_y_col, y_label, raw_x_col, cal_x_col, x_label = get_metric_names_and_labels(y_metric, x_metric)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    
    # Plot arrows from raw metrics to calibrated metrics
    for _, row in filtered_data.iterrows():
        x1, y1 = row[raw_x_col], row[raw_y_col]
        x2, y2 = row[cal_x_col], row[cal_y_col]
        arrow = FancyArrowPatch(
                posA=(x1, y1),
                posB=(x2, y2),
                arrowstyle="Simple, tail_width=0.01, head_width=0.2, head_length=0.1",
                color='tab:purple',
                alpha=alpha,
                mutation_scale=20,
                linewidth=1.5
            )
        ax.add_patch(arrow)

    ax.autoscale_view()  # ensure arrows are within view
    if dataset == 'energy' and filter_val == 'Wind' and y_metric == 'pit_entropy':
        ax.set_ylim(0.8, 1.005)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(f'{title_prefix}', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    if save:
        # remove 
        despine(ax, mode='TR')
        plt.tight_layout()
        plt.savefig(f'{fig_folder}/{filename_prefix}_arrow_{filter_col}={filter_val}_{y_metric}_{x_metric}.pdf', bbox_inches='tight')
        plt.close()


def create_arrow_plot_differences(df, filter_col, filter_val, fig_folder, alpha=0.2, 
                                 title_prefix='', filename_prefix='', dataset=None, 
                                 y_metric='pit_entropy', x_metric='quantile_score'):
    """Create arrow plot showing differences from origin (0,0)."""
    # Filter data for this condition
    filtered_data = df[df[filter_col] == filter_val].copy()
    
    if len(filtered_data) == 0:
        print(f"No data for {filter_col} {filter_val}")
        return
    
    raw_y_col, cal_y_col, y_label, raw_x_col, cal_x_col, x_label = get_metric_names_and_labels(y_metric, x_metric)

    
    # Calculate differences
    filtered_data['x_diff'] = filtered_data[cal_x_col] - filtered_data[raw_x_col]
    filtered_data['y_diff'] = filtered_data[cal_y_col] - filtered_data[raw_y_col]
    
    _, ax = plt.subplots(figsize=(5, 5))
    
    # Plot arrows from origin to differences
    for _, row in filtered_data.iterrows():
        x1, y1 = 0, 0
        x2, y2 = row['x_diff'], row['y_diff']
        arrow = FancyArrowPatch(
                posA=(x1, y1),
                posB=(x2, y2),
                arrowstyle="Simple, tail_width=0.01, head_width=0.2, head_length=0.1",
                color='tab:purple',
                alpha=alpha,
                mutation_scale=20,
                linewidth=1.5
            )
        ax.add_patch(arrow)
    
    # # Add reference lines at origin
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.5, linewidth=1)

    # remove plot edges
    despine(ax)
    if dataset == 'covid':
        ax.set_xlim(-41, 10)
        if y_metric == 'pit_entropy' and x_metric == 'quantile_score':
            ax.set_ylim(-0.05, 0.58)
        else:  # avg_coverage_gap
            ax.autoscale_view()
    else:
        ax.autoscale_view()  # ensure arrows are within view
    ax.set_xlabel('$\\Delta$Score (Calibrated - Raw)', fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(f'{title_prefix}={filter_val}', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{fig_folder}/{filename_prefix}_arrow_differences_{filter_col}={filter_val}_{y_metric}_{x_metric}.pdf', bbox_inches='tight')
    plt.close()

def create_covid_arrow_plot_differences(df, horizon, fig_folder, alpha=0.2, y_metric='pit_entropy', x_metric='quantile_score'):
    """Create arrow plot showing differences from origin (0,0)."""
    return create_arrow_plot_differences(df, 'horizon', horizon, fig_folder, alpha, '$h$', 'covid', dataset='covid', 
                                         y_metric=y_metric, x_metric=x_metric)

def create_calibration_plot(df, filter_col, filter_val, fig_folder, forecast_type='raw', 
                           title_prefix='', filename_prefix='', alpha=0.1, figsize=(2.6, 2.6)):
    """Create calibration plot for forecasts."""

    # Specify plotting color
    if forecast_type == 'raw':
        color = 'tab:red'
    else:
        color = 'tab:blue'

    # Filter data for this condition
    filtered_data = df[df[filter_col] == filter_val].copy()
    
    if len(filtered_data) == 0:
        print(f"No data for {filter_col} {filter_val}")
        return
    
    _, ax = plt.subplots(figsize=figsize)
    
    # Extract coverage columns for the specified forecast type
    coverage_cols = [col for col in df.columns if col.startswith(f'{forecast_type}_coverage_')]
    levels = [float(col.split('_')[-1]) / 100 for col in coverage_cols]
    levels = sorted(levels)
    
    # Plot calibration for each row in filtered data
    for _, row in filtered_data.iterrows():
        coverage_values = []
        for level in levels:
            col_name = f'{forecast_type}_coverage_{int(np.round(level*100, 0))}'
            if col_name in row:
                coverage_values.append(row[col_name])
        
        if len(coverage_values) == len(levels):
            ax.plot(levels, coverage_values, alpha=alpha, linewidth=1, color=color)
    
    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2)
    
    ax.set_xlabel('Desired coverage')
    ax.set_ylabel('Actual coverage')
    # ax.set_title(f'{filter_val}')
    # Set ticks 
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_xticklabels(['0', '','','', '1'])
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.set_yticklabels(['0', '','','', '1'])
    # ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # if figsize == (2.2, 2.2):
    #     ax.legend(fontsize=6, loc='upper left')
    # else:
    #     ax.legend(fontsize=7, loc='upper left')

    ax.grid(True, alpha=0.2)

    # despine top and right edges
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{fig_folder}/{filename_prefix}_calibration_{forecast_type}_{filter_col}={filter_val}.pdf', bbox_inches='tight')
    plt.close()

def create_covid_calibration_plot(df, horizon, fig_folder, forecast_type='raw'):
    """Create calibration plot for COVID forecasts."""
    return create_calibration_plot(df, 'horizon', horizon, fig_folder, forecast_type, '$h$', 'covid')

# Energy wrapper functions for the abstracted plot types
def create_energy_arrow_plot_differences(df, target_type, fig_folder, alpha=0.2,
                                         y_metric='pit_entropy', x_metric='quantile_score'):
    """Create Energy arrow plot showing differences from origin."""
    # Filter for ERCOT data first
    ercot_data = df[df['ISO'] == 'ERCOT']
    # Get hour from data to include in filename
    if len(ercot_data) > 0 and 'hour' in ercot_data.columns:
        hour = ercot_data['hour'].iloc[0]
        filename_prefix = f'energy_hour_{hour}'
    else:
        filename_prefix = 'energy'
    return create_arrow_plot_differences(ercot_data, 'target_variable', target_type, fig_folder, alpha, f'Target', filename_prefix, dataset='energy',
                                         y_metric=y_metric, x_metric=x_metric)

def create_energy_arrow_plot(df, target_type, fig_folder, alpha=0.2,
                             title_prefix='Target', filename_prefix='energy',
                             y_metric='pit_entropy', x_metric='quantile_score',
                             ax=None, save=True):
     """Create Energy arrow plot."""
     # Filter for ERCOT data first
     ercot_data = df[df['ISO'] == 'ERCOT']
     # Get hour from data to include in filename if not already specified
     if filename_prefix == 'energy' and len(ercot_data) > 0 and 'hour' in ercot_data.columns:
         hour = ercot_data['hour'].iloc[0]
         filename_prefix = f'energy_hour_{hour}'
     return create_arrow_plot(ercot_data, 'target_variable', target_type, fig_folder, alpha, title_prefix, filename_prefix, dataset='energy',
                              y_metric=y_metric, x_metric=x_metric, ax=ax, save=save)

def create_energy_calibration_plot(df, target_type, fig_folder, forecast_type='raw'):
    """Create Energy calibration plot."""
    # Filter for ERCOT data first
    ercot_data = df[df['ISO'] == 'ERCOT']
    # Get hour from data to include in filename
    if len(ercot_data) > 0 and 'hour' in ercot_data.columns:
        hour = ercot_data['hour'].iloc[0]
        filename_prefix = f'energy_hour_{hour}'
    else:
        filename_prefix = 'energy'
    return create_calibration_plot(ercot_data, 'target_variable', target_type, fig_folder, forecast_type, 
                                   '', filename_prefix, alpha=0.2, figsize=(2.2, 2.2))

def create_covid_figures(fig_folder):
    """Generate all COVID-related figures."""
    print("Generating COVID figures...")
    
    # Load COVID metrics
    covid_df = load_covid_metrics()
    if covid_df is None:
        return
    
    # Create output directory
    covid_fig_folder = Path(f"{fig_folder}/covid")
    covid_fig_folder.mkdir(parents=True, exist_ok=True)
    
    horizons = [1, 2, 3, 4]
    
    for horizon in horizons:
        print(f"  Processing horizon {horizon}...")
        
        # Generate plots for both y-metrics
        for x_metric in ['quantile_score', 'log_score']:
            for y_metric in ['pit_entropy', 'avg_coverage_gap']:
                print(f"    Y-metric: {y_metric}")
                
                # 1. Arrow plot with averaging across states
                create_covid_arrow_plot_averaged(covid_df, horizon, covid_fig_folder, 
                                                 y_metric=y_metric, x_metric=x_metric)

                # 2. Arrow plot showing differences from origin
                create_covid_arrow_plot_differences(covid_df, horizon, covid_fig_folder, 
                                                     y_metric=y_metric, x_metric=x_metric)

        # 3. Calibration plot for raw forecasts (only one version needed)
        create_covid_calibration_plot(covid_df, horizon, covid_fig_folder, 'raw')

        # 4. Calibration plot for calibrated forecasts (only one version needed)
        create_covid_calibration_plot(covid_df, horizon, covid_fig_folder, 'cal')

    # Also create version of #1 with all horizons as subplots for both metrics
    for x_metric in ['quantile_score', 'log_score']:
        for y_metric in ['pit_entropy', 'avg_coverage_gap']:
            if x_metric == 'log_score' and y_metric == 'avg_coverage_gap':
                continue  # skip 

            print(f"Creating combined arrow plot for all horizons - {y_metric}...")
            _, axes = plt.subplots(1,4,figsize=(10, 3), sharex=True, sharey=True)
            for horizon in horizons:
                create_covid_arrow_plot_averaged(covid_df, horizon, covid_fig_folder, ax=axes[horizon-1], save=False, 
                                                 y_metric=y_metric, x_metric=x_metric)
            for i, ax in enumerate(axes):
                # despine top and right edges
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

                # Remove y-axis labels for all but the first plot
                if i > 0:
                    ax.set_ylabel('')

            plt.tight_layout()
            plt.savefig(covid_fig_folder / f'covid_arrow_averaged_all_horizons_{y_metric}_{x_metric}.pdf', bbox_inches='tight')
            plt.close()

    print(f"COVID figures saved to {covid_fig_folder}")

def create_energy_figures(fig_folder):
    """Generate all Energy-related figures."""
    print("Generating Energy figures...")
    
    # Load Energy metrics
    energy_df = load_energy_metrics()
    if energy_df is None:
        return
    
    # Create output directory
    energy_fig_folder = Path(f"{fig_folder}/energy")
    energy_fig_folder.mkdir(parents=True, exist_ok=True)
    
    target_types = ['Wind', 'Solar']
    hours = [0, 4, 8, 12, 16, 20]
    
    for target_type in target_types:
        print(f"  Processing {target_type}...")
        
        # Create separate subfolder for this target type
        target_fig_folder = energy_fig_folder / target_type
        target_fig_folder.mkdir(parents=True, exist_ok=True)
        
        # Generate individual plots for each target_var x hour combination
        for hour in hours:
            print(f"    Hour: {hour}")
            energy_df_hour = energy_df[energy_df['hour'] == hour]

            for y_metric in ['pit_entropy', 'avg_coverage_gap']:
                print(f"      Y-metric: {y_metric}")

                # 1. Arrow plot showing differences from origin for this hour
                create_energy_arrow_plot_differences(energy_df_hour, target_type, target_fig_folder, y_metric=y_metric)

                # 2. Raw arrow plot for this hour
                create_energy_arrow_plot(energy_df_hour, target_type, target_fig_folder, alpha=0.2,
                                       title_prefix=f'Hour {hour}', filename_prefix=f'energy_hour_{hour}', y_metric=y_metric)

            # 3. Calibration plots for this target_var x hour combination
            create_energy_calibration_plot(energy_df_hour, target_type, target_fig_folder, 'raw')
            create_energy_calibration_plot(energy_df_hour, target_type, target_fig_folder, 'cal')
        
        # Create 6-subplot figures with arrow plots for different hours
        for y_metric in ['pit_entropy', 'avg_coverage_gap']:
            print(f"    Creating combined arrow plot for all hours - {y_metric}...")
            _, axes = plt.subplots(2, 3, figsize=(7, 4.7), sharex=True, sharey=True)
            axes = axes.flatten()
            
            # Determine y_label once for this metric
            if y_metric == 'pit_entropy':
                y_label = 'PIT Entropy'
            else:
                y_label = 'Calibration Error'

            # Map UTC hours to CST string (e.g., "12:00 p.m.") for title labeling
            cst_hour_labels = {0: '6:00 p.m.', 4: '10:00 p.m.', 8: '2:00 a.m.',
                               12: '6:00 a.m.', 16: '10:00 a.m.', 20: '2:00 p.m.'}

            for i, hour in enumerate([8, 12, 16, 20, 0, 4]): # Order hours starting from midnight CST
                energy_df_hour = energy_df[energy_df['hour'] == hour]
                
                # Use the refactored create_energy_arrow_plot function
                create_energy_arrow_plot(energy_df_hour, target_type, target_fig_folder, 
                                        alpha=0.2, title_prefix=cst_hour_labels[hour], 
                                        filename_prefix='', y_metric=y_metric,
                                        ax=axes[i], save=False)
                
                # Remove y-axis labels for all but the first plot in each row
                if i % 3 != 0:
                    axes[i].set_ylabel('')
                # Remove x-axis labels for all but the bottom row
                if i < 3:
                    axes[i].set_xlabel('')
                # despine top and right edges
                axes[i].spines['top'].set_visible(False)
                axes[i].spines['right'].set_visible(False)
            
            # plt.suptitle(f'{target_type} - {y_label}', fontsize=16)
            plt.tight_layout()
            plt.savefig(target_fig_folder / f'{target_type.lower()}_arrow_all_hours_{y_metric}.pdf', bbox_inches='tight')
            plt.close()
    
    print(f"Energy figures saved to {energy_fig_folder}")

if __name__ == "__main__":
    print("Starting figure generation...")

    fig_folder = 'figs'

    # Create figures directory
    Path(fig_folder).mkdir(exist_ok=True)
    
    # Generate COVID figures
    create_covid_figures(fig_folder)
    
    # Generate Energy figures  
    create_energy_figures(fig_folder)
    
    print("Figure generation complete!")