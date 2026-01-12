import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

from covid_utils import good_forecasters, states, levels, get_covid_data
from utils import QT

horizon = 1
lr = 'adaptive+'

def compute_crossing_frac(quantiles):
    """Compute the fraction of times quantiles cross (are not monotonic)."""
    # Transpose if shape[1] > shape[0]
    if quantiles.shape[1] > quantiles.shape[0]:
        quantiles = quantiles.T
    T = len(quantiles)
    crossed = np.zeros((T,))
    for t in range(T):
        crossed[t] = ~np.all(np.sort(quantiles[t,:]) == quantiles[t,:])

    return np.mean(crossed)

def run_qt_crossing_analysis():
    """Run quantile tracker analysis and compute crossing fractions for all COVID forecasters."""
    cache_file = f"cache/QT_covid_h={horizon}_lr={lr}_crossing_frac.npy"
    
    # Check if results already exist
    if os.path.exists(cache_file):
        print(f"Loading existing results from {cache_file}")
        crossing_fractions = np.load(cache_file)
        print(f"Loaded {len(crossing_fractions)} crossing fraction values")
    else:
        print("Computing crossing fractions for all COVID forecasters and states...")
        
        Yhat_type = 'quantile-specific'
        crossing_fractions = []
        
        total_experiments = len(good_forecasters) * len(states)
        experiment_count = 0
        
        for forecaster in good_forecasters:
            for state in states:
                experiment_count += 1
                print(f"Processing {experiment_count}/{total_experiments}: {forecaster}, {state}")
                
                try:
                    # Get data
                    Y, Yhat = get_covid_data(forecaster, state, horizon, 
                                           Yhat_type=Yhat_type, 
                                           levels=levels)
                    
                    # Run quantile tracker
                    Y_forecast = QT(Y, levels, Yhat=Yhat, lr=lr, lr_window=50, q0=0)
                    
                    # Compute crossing fraction
                    crossing_frac = compute_crossing_frac(Y_forecast)
                    crossing_fractions.append(crossing_frac)
                    
                    print(f"  Crossing fraction: {crossing_frac:.4f}")
                    
                except Exception as e:
                    print(f"  Error processing {forecaster}, {state}: {e}")
                    continue
        
        # Convert to numpy array and save
        crossing_fractions = np.array(crossing_fractions)
        
        # Create cache directory if it doesn't exist
        os.makedirs("cache", exist_ok=True)
        
        # Save results
        np.save(cache_file, crossing_fractions)
        print(f"Saved {len(crossing_fractions)} crossing fractions to {cache_file}")
    
    return crossing_fractions

def create_crossing_fraction_histogram(crossing_fractions):
    """Create a publication-quality histogram of crossing fractions."""
    # Set style for publication quality
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Create histogram
    n_bins = 30
    counts, bins, patches = ax.hist(crossing_fractions, bins=n_bins, 
                                   alpha=0.7, density=False, 
                                   color='steelblue', edgecolor='black', linewidth=0.5)
    
    # Add statistics
    mean_val = np.mean(crossing_fractions)
    median_val = np.median(crossing_fractions)
    std_val = np.std(crossing_fractions)
    
    # Add vertical lines for mean and median
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_val:.3f}')
    ax.axvline(median_val, color='orange', linestyle='--', linewidth=2, 
               label=f'Median: {median_val:.3f}')
    
    # Formatting
    ax.set_xlabel('Fraction of time steps with crossed quantiles', fontsize=14)
    ax.set_ylabel('Number of time series', fontsize=14)
    # ax.set_title('Distribution of Quantile Crossing Fractions\n(COVID Forecasters, Horizon=4)', 
    #             fontsize=16, pad=20)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend(fontsize=12)
    
    # # Add text box with statistics
    # stats_text = f'N = {len(crossing_fractions)}\n'
    # stats_text += f'Mean = {mean_val:.3f}\n'
    # stats_text += f'Std = {std_val:.3f}\n'
    # stats_text += f'Min = {np.min(crossing_fractions):.3f}\n'
    # stats_text += f'Max = {np.max(crossing_fractions):.3f}'
    
    # ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
    #         verticalalignment='top', fontsize=11,
    #         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Adjust layout
    plt.tight_layout()
    
    # Create output directory and save
    output_dir = Path("figs/QT")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"QT_covid_h={horizon}_lr={lr}_crossing_frac_histogram.pdf"
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Histogram saved to {output_file}")
    
    # Print summary statistics
    print("\nCrossing Fraction Statistics:")
    print(f"Number of series: {len(crossing_fractions)}")
    print(f"Mean: {mean_val:.4f}")
    print(f"Median: {median_val:.4f}")
    print(f"Standard deviation: {std_val:.4f}")
    print(f"Min: {np.min(crossing_fractions):.4f}")
    print(f"Max: {np.max(crossing_fractions):.4f}")
    print(f"25th percentile: {np.percentile(crossing_fractions, 25):.4f}")
    print(f"75th percentile: {np.percentile(crossing_fractions, 75):.4f}")

if __name__ == "__main__":
    print("Starting Quantile Tracker crossing fraction analysis...")
    
    # Run analysis and get crossing fractions
    crossing_fractions = run_qt_crossing_analysis()
    
    # Create histogram
    create_crossing_fraction_histogram(crossing_fractions)
    
    print("Analysis complete!")
