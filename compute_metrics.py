
import numpy as np
import pandas as pd
import os

from covid_utils import good_forecasters, states, levels as covid_levels, get_covid_data
from energy_utils import get_energy_data, ISO_to_sites, levels as energy_levels

from utils import quantile_loss, compute_coverage

import pdb

'''
In this file, we generate csv's of metrics

For COVID, the columns are 
- state
- forecaster 
- horizon
- Number of time steps in that series
- For Raw and Calibrated forecasts:
    * coverage at level {0.01, 0.025, ..., 0.99}
    * Average coverage gap (avg of |desired - actual coverage|)
    * PIT entropy
    * Average quantile score

For Energy, the columns are
- ISO
- site
- target variable (Wind or Solar)
- Number of time steps in that series
- For Raw and Calibrated forecasts:
    * coverage at level {0.01, 0.025, ..., 0.99}
    * Average coverage gap (avg of |desired - actual coverage|)
    * PIT entropy
    * Average quantile score
'''

# ----------------------------------------------------------------
#                           PIT Entropy
# ----------------------------------------------------------------


def approximate_density_at_y(y, levels, values, min_density=1e-16):
    """
    Given quantile levels and corresponding values, compute the density at y.
    """

    # We will use the PIT_score function to find F(y+eps) and F(y-eps)
    eps = 1e-6
    F_y_plus = PIT_score(y + eps, values, levels)
    F_y_minus = PIT_score(y - eps, values, levels)

    # The density is approximated as the difference quotient
    density = (F_y_plus - F_y_minus) / (2 * eps)

    # Ensure density is nonzero
    density = max(density, min_density)
    return density

def PIT_score(Y, values, levels):
    """
    Compute PIT = F(Y) from quantile forecasts (levels, values), without
    constructing a distribution object.

    Assumptions & behavior:
    - `levels` are quantile levels in ascending order in (0,1).
    - `values` are the corresponding (nondecreasing) quantile forecasts (ties allowed).
    - Interior CDF is piecewise-linear between (values, levels).
    - Exponential tails:
        left:  F(x) = mass_left * exp(lam_left * (x - v0))
        right: F(x) = 1 - mass_right * exp(-lam_right * (x - v_end))
      with lam chosen so that the boundary density matches the density from the
      nearest non-tied interior bin. If all interior bins are tied, falls back to lam=1.0.
    - Ties are rounded up: if multiple quantile levels map to the same value v
      and Y==v, return the *largest* level among those tied (e.g., p50=p60=10 and Y=10 -> 0.6).

    Supports scalar or array Y.
    """
    levels = np.asarray(levels, dtype=float)
    values = np.asarray(values, dtype=float)
    if levels.ndim != 1 or values.ndim != 1 or len(levels) != len(values):
        raise ValueError("levels and values must be 1D arrays of the same length")
    if not np.all(np.isfinite(values)):
        raise ValueError("values must be finite")
    if not np.all(np.diff(levels) > 0):
        raise ValueError("levels must be strictly increasing")
    if not np.all(np.diff(values) >= 0):
        pdb.set_trace()
        raise ValueError("values must be nondecreasing (ties allowed)")

    Y = np.asarray(Y, dtype=float)
    scalar_input = (Y.ndim == 0)
    if scalar_input:
        Y = Y[None]

    v0, v_end = values[0], values[-1]
    mass_left  = levels[0]
    mass_right = 1.0 - levels[-1]

    out = np.empty_like(Y, dtype=float)

    # Find a usable interior density near the left boundary (first non-tied bin)
    lam_left = 1.0  # safe fallback
    # density = dq/dx on first bin with positive width
    for i in range(len(values) - 1):
        dx = values[i+1] - values[i]
        if dx > 0:
            dq = levels[i+1] - levels[i]
            dens_left = dq / dx
            if mass_left > 0:
                lam_left = dens_left / mass_left
            break

    # Find a usable interior density near the right boundary (last non-tied bin)
    lam_right = 1.0  # safe fallback
    for i in range(len(values) - 2, -1, -1):
        dx = values[i+1] - values[i]
        if dx > 0:
            dq = levels[i+1] - levels[i]
            dens_right = dq / dx
            if mass_right > 0:
                lam_right = dens_right / mass_right
            break

    # Left tail: x <= v0
    mask_l = Y <= v0
    if np.any(mask_l):
        out[mask_l] = mass_left * np.exp(lam_left * (Y[mask_l] - v0))

    # Right tail: x >= v_end
    mask_r = Y >= v_end
    if np.any(mask_r):
        out[mask_r] = 1.0 - mass_right * np.exp(-lam_right * (Y[mask_r] - v_end))

    # Interior: v0 < x < v_end
    mask_i = (~mask_l) & (~mask_r)
    if np.any(mask_i):
        Yi = Y[mask_i]

        # For each Yi, find k = first index with values[k] >= Yi  (searchsorted right= 'left')
        # We'll then use:
        #   - If values[k] == Yi and there may be ties: advance k to the LAST index in the tied block
        #     and set F(Yi) = levels[k]  (round-up rule).
        #   - Else (values[k-1] < Yi < values[k]): linear interpolate between (values[k-1], levels[k-1])
        #     and (values[k], levels[k]).
        k = np.searchsorted(values, Yi, side='left')

        Fi = np.empty_like(Yi, dtype=float)
        for j in range(len(Yi)):
            kk = k[j]

            # Safety for pathological cases (shouldn't happen due to masks)
            kk = min(max(1, kk), len(values)-1)

            if values[kk] == Yi[j]:
                # Round up across ties: move to the last index where value == Yi[j]
                kk2 = kk
                while kk2 + 1 < len(values) and values[kk2 + 1] == Yi[j]:
                    kk2 += 1
                Fi[j] = levels[kk2]
            else:
                # Strict interior interpolation between kk-1 and kk
                x0, x1 = values[kk-1], values[kk]
                q0, q1 = levels[kk-1], levels[kk]
                # Since x0 < Yi < x1 by construction, x1-x0 > 0 (ties would have been caught above)
                t = (Yi[j] - x0) / (x1 - x0)
                Fi[j] = q0 + t * (q1 - q0)

        out[mask_i] = Fi

    return out[0] if scalar_input else out

def compute_binned_entropy(samples, bins=10, bin_edges=None, base=None):
    """
    Compute the entropy of continuous samples by binning.

    Parameters
    ----------
    samples : array-like
        1D array of continuous observations.
    bins : int, optional
        Number of equal-width bins to use if bin_edges not provided.
    bin_edges : array-like, optional
        Explicit bin edges. If provided, `bins` is ignored.
    base : float or None, optional
        Logarithm base for entropy. If None, use natural log.

    Returns
    -------
    entropy : float
        The entropy of the binned distribution: -sum(p_i * log(p_i)).
    probs : ndarray
        The probability mass in each bin.
    edges : ndarray
        The bin edges used.
    """
    samples = np.asarray(samples)

    # Determine bin edges
    if bin_edges is not None:
        edges = np.asarray(bin_edges)
    else:
        edges = np.linspace(0, 1, bins + 1)

    # Digitize samples into bins
    counts, _ = np.histogram(samples, bins=edges)
    probs = counts / counts.sum()

    # Filter out zero-probability bins
    nonzero = probs > 0
    probs_nz = probs[nonzero]

    # Compute entropy
    if base is None:
        log_probs = np.log(probs_nz)
    else:
        log_probs = np.log(probs_nz) / np.log(base)

    entropy = -np.sum(probs_nz * log_probs)

    # Normalize by dividing by the maximum possible entropy given the number of bins
    entropy /= np.log(bins)

    # return entropy, probs, edges
    return entropy 

def compute_entropy_of_PIT_scores(Ys, forecasts_over_time, levels):
    """
    Compute the PIT scores for a set of quantiles and observed values over time.
    """
    # Much more efficient: compute PIT scores for each time step individually
    # since each time step may have different quantile values
    try:
        num_times = forecasts_over_time.shape[1]
        PIT_scores = np.zeros(num_times)

        # Vectorized computation for each time step
        for t in range(num_times):
            PIT_scores[t] = PIT_score(Ys[t], forecasts_over_time[:, t], levels)
    
        # Compute the entropy of the PIT scores
        entropy = compute_binned_entropy(PIT_scores)
        return entropy

    except Exception as e:
        return np.nan
    

# ----------------------------------------------------------------
#                           Quantile Score
# ----------------------------------------------------------------

def compute_avg_quantile_score(Ys, forecasts, levels, min0=True, over_time=False):
    """
    Compute the average quantile score across all time steps and quantile levels.
    
    y: np.array, shape (n,)
    forecast_arr: np.array, shape (len(levels), n)
    levels: list of floats, quantile levels
    over_time: boolean, whether to return running average over time
    """
    if min0:
        # Ensure that all forecasts are non-negative
        forecasts = np.maximum(forecasts, 0)
        
    n = len(Ys)
    scores = np.zeros((n,)) # score at each time
    for i, y in enumerate(Ys):
        time_forecasts = forecasts[:, i]  # Get forecasts for time i
        scores[i] = np.mean([quantile_loss(level, y - forecast) for forecast, level in zip(time_forecasts, levels)])

    # Compute running average of quantile score over time
    if over_time:
        scores = np.cumsum(scores) / (np.arange(n) + 1)
        return scores
    
    return np.mean(scores)

# ----------------------------------------------------------------
#                           Log Score
# ----------------------------------------------------------------

def log_score(y, forecast_arr, levels):
    """
    Log score = - log density at true y

    y: scalar
    forecast_arr: np.array, shape (len(levels),)
    levels: list of floats, quantile levels
    """
        
    # Approximate log density at true y
    density_at_y = approximate_density_at_y(y, levels, forecast_arr)
    if density_at_y <= 0:
        pdb.set_trace()
        raise ValueError("Density at true value is non-positive... Error in converting quantiles to distribution?")
    return -np.log(density_at_y)

def compute_avg_log_score(Ys, forecasts, levels, min0=True, over_time=False):
    """
    Log score = - log density at true y

    y: np.array, shape (n,)
    forecast_arr: np.array, shape (len(levels), n)
    levels: list of floats, quantile levels
    over_time: boolean, whether to return running average over time
    """
    if min0:
        # Ensure that all forecasts are non-negative
        forecasts = np.maximum(forecasts, 0)
        
    n = len(Ys)
    scores = np.zeros((n,)) # score at each time
    for i, y in enumerate(Ys):
        time_forecasts = forecasts[:, i]  # Get forecasts for time i
        scores[i] = log_score(y, time_forecasts, levels)
        
    # Compute running average of log score over time
    if over_time:
        scores = np.cumsum(scores) / (np.arange(n) + 1)
        return scores
    
    return np.mean(scores)

# ----------------------------------------------------------------
#                           Coverage Functions
# ----------------------------------------------------------------

def compute_all_coverages(Y, forecasts, levels):
    """
    Compute empirical coverage for all quantile levels.
    """
    coverages_array = compute_coverage(Y, forecasts, levels, plot_results=False)
    
    # Convert to dictionary format for compatibility
    coverages = {}
    for i, level in enumerate(levels):
        # Convert level to percentage for key name
        coverages[f'coverage_{int(np.round(level*100, 0))}'] = coverages_array[i]
    
    return coverages

def compute_average_coverage_gap(Y, forecasts, levels):
    """
    Compute average coverage gap across all confidence levels.
    """
    coverages_array = compute_coverage(Y, forecasts, levels, plot_results=False)
    gaps = np.abs(levels - coverages_array)
    return np.mean(gaps)

# ----------------------------------------------------------------
#                           Main Functions
# ----------------------------------------------------------------

# Helper function
def replace_inf_values(Yhat_cal, Y, k=30):
    if np.any(np.isinf(Yhat_cal)):
        print('Forecast contains infinite values,', 
                'replacing -inf with min(smallest observed Y_t in last k time steps, smallest finite current quantile forecast)',
                'and +inf with max(largest observed Y_t in last k time steps, largest finite current quantile forecast)')
        
        # Replace +inf with max observed Y in last k time steps
        m, T = np.shape(Yhat_cal)
        for t in range(T):
            if np.isinf(Yhat_cal[:,t]).any():
                max_Y = np.max(Y[max(0, t-k):t+1])
                min_Y = np.min(Y[max(0, t-k):t+1])
                smallest_finite = np.min(Yhat_cal[np.isfinite(Yhat_cal[:,t]), t])
                largest_finite = np.max(Yhat_cal[np.isfinite(Yhat_cal[:,t]), t])
                Yhat_cal[:,t] = np.where(Yhat_cal[:,t] == float('-inf'), min(min_Y, smallest_finite), Yhat_cal[:,t])
                Yhat_cal[:,t] = np.where(Yhat_cal[:,t] == float('inf'), max(max_Y, largest_finite), Yhat_cal[:,t])

        # pdb.set_trace()
        
        # assert that forecasts are still ordered
        for t in range(T):
            if not np.all(np.diff(Yhat_cal[:,t]) >= 0):
                print("Error: Forecast quantiles are not ordered after replacing inf values.")
                pdb.set_trace()
            # assert np.all(np.diff(Yhat_cal[:,t]) >= 0), "Error: Forecast quantiles are not ordered after replacing inf values."

    return Yhat_cal

def compute_covid_metrics():
    """
    Compute metrics for COVID data and save to CSV.
    """
    results = []
    metrics_folder = 'metrics'
    
    # Create metrics folder if it doesn't exist
    os.makedirs(metrics_folder, exist_ok=True)
    
    print("Computing COVID metrics...")
    
    # Calculate total number of combinations for progress tracking
    total_combinations = len(good_forecasters) * len(states) * 4  # 4 horizons
    current_count = 0


    for forecaster in good_forecasters:
        print(f"Processing forecaster: {forecaster}")
        for state in states:
            for horizon in [1, 2, 3, 4]:

                current_count += 1
                print(f"  [{current_count}/{total_combinations}] {forecaster}_{state}_h={horizon}")
                
                # Load raw forecasts and observed values
                Y, Yhat_raw = get_covid_data(forecaster, state, horizon, 
                                                Yhat_type='quantile-specific', 
                                                levels=covid_levels, 
                                                truncate_negY=True, 
                                                return_df=False)
                
                if Yhat_raw is None or Y is None:
                    continue
                    
                # Load calibrated forecasts
                cache_path = f"cache/covid/{forecaster}_{state}_Yhat=quantile-specific_h={horizon}.npy"
                if os.path.exists(cache_path):
                    Yhat_cal = np.load(cache_path)
                else:
                    print(f"Warning: No calibrated forecasts found for {forecaster}_{state}_h={horizon}")
                    continue

                Yhat_cal = replace_inf_values(Yhat_cal, Y) # Replace infinite values with recent max Y_t
                    
                
                # Basic info
                num_time_steps = len(Y)
                
                # Compute metrics for raw forecasts
                raw_coverages = compute_all_coverages(Y, Yhat_raw, covid_levels)
                raw_avg_gap = compute_average_coverage_gap(Y, Yhat_raw, covid_levels)
                raw_pit_entropy = compute_entropy_of_PIT_scores(Y, Yhat_raw, covid_levels) 
                raw_quantile_score = compute_avg_quantile_score(Y, Yhat_raw, covid_levels)
                raw_log_score = compute_avg_log_score(Y, Yhat_raw, covid_levels)
                
                # Compute metrics for calibrated forecasts  
                cal_coverages = compute_all_coverages(Y, Yhat_cal, covid_levels)
                cal_avg_gap = compute_average_coverage_gap(Y, Yhat_cal, covid_levels)
                cal_pit_entropy = compute_entropy_of_PIT_scores(Y, Yhat_cal, covid_levels)
                cal_quantile_score = compute_avg_quantile_score(Y, Yhat_cal, covid_levels)
                cal_log_score = compute_avg_log_score(Y, Yhat_cal, covid_levels)
                
                # Create result row
                result = {
                    'state': state,
                    'forecaster': forecaster,
                    'horizon': horizon,
                    'num_time_steps': num_time_steps,
                    'raw_avg_coverage_gap': raw_avg_gap,
                    'cal_avg_coverage_gap': cal_avg_gap,
                    'raw_pit_entropy': raw_pit_entropy, 
                    'cal_pit_entropy': cal_pit_entropy,
                    'raw_avg_quantile_score': raw_quantile_score,
                    'cal_avg_quantile_score': cal_quantile_score,
                    'raw_avg_log_score': raw_log_score,
                    'cal_avg_log_score': cal_log_score
                }
                
                # Add individual coverage metrics
                for key, value in raw_coverages.items():
                    result[f'raw_{key}'] = value
                for key, value in cal_coverages.items():
                    result[f'cal_{key}'] = value
                
                results.append(result)
                    
                # except Exception as e:
                #     print(f"Error processing {forecaster}_{state}_h={horizon}: {e}")
                #     continue
    
    # Save to CSV
    df = pd.DataFrame(results)
    output_path = os.path.join(metrics_folder, 'covid_metrics.csv')
    df.to_csv(output_path, index=False)
    print(f"COVID metrics saved to {output_path} ({len(results)} rows)")
    
    return df

def compute_energy_metrics(cache_folder='cache/energy/', save_to='energy_metrics.csv'):
    """
    Compute metrics for Energy data and save to CSV.
    """
    metrics_folder = 'metrics'
    os.makedirs(metrics_folder, exist_ok=True)

    hours = [0, 4, 8, 12, 16, 20]  # in UTC
    
    results = []
    
    print("Computing Energy metrics...")

    ISO_list = ['ERCOT'] 
    
    # Calculate total number of combinations for progress tracking
    total_combinations = 0
    for ISO in ISO_list:
        for target_var in ['Wind', 'Solar']:
            sites = ISO_to_sites[ISO][target_var]
            total_combinations += len(sites) * len(hours)

    current_count = 0
    
    for ISO in ISO_list: 
        print(f"Processing ISO: {ISO}")
        for target_var in ['Wind', 'Solar']:
            sites = ISO_to_sites[ISO][target_var]
            
            for site in sites:
                for hour in hours: 
                    current_count += 1
                    print(f"  [{current_count}/{total_combinations}] {ISO}_{site}_{target_var}_{hour}")

                    # Load raw forecasts and observed values
                    try:
                        Y, Yhat_raw = get_energy_data(ISO, site, hour=hour, target_variable=target_var)
                  
                        # Load calibrated forecasts  
                        cache_path = f"{cache_folder}/{ISO}_{site}_{target_var}_Yhat=quantile-specific_hour={hour}.npy"
                        Yhat_cal = np.load(cache_path)

                        Yhat_cal = replace_inf_values(Yhat_cal, Y) # Replace infinite values with recent max Y_t
                        
                        # Basic info
                        num_time_steps = len(Y)
                        
                        # Compute metrics for raw forecasts
                        raw_coverages = compute_all_coverages(Y, Yhat_raw, energy_levels)
                        raw_avg_gap = compute_average_coverage_gap(Y, Yhat_raw, energy_levels)
                        raw_pit_entropy = compute_entropy_of_PIT_scores(Y, Yhat_raw, energy_levels)
                        raw_quantile_score = compute_avg_quantile_score(Y, Yhat_raw, energy_levels)
                        
                        # Compute metrics for calibrated forecasts
                        cal_coverages = compute_all_coverages(Y, Yhat_cal, energy_levels)
                        cal_avg_gap = compute_average_coverage_gap(Y, Yhat_cal, energy_levels)
                        cal_pit_entropy = compute_entropy_of_PIT_scores(Y, Yhat_cal, energy_levels)
                        cal_quantile_score = compute_avg_quantile_score(Y, Yhat_cal, energy_levels)
                        
                        # Create result row
                        result = {
                            'ISO': ISO,
                            'site': site,
                            'target_variable': target_var,
                            'hour': hour,
                            'num_time_steps': num_time_steps,
                            'raw_avg_coverage_gap': raw_avg_gap,
                            'raw_pit_entropy': raw_pit_entropy,
                            'raw_avg_quantile_score': raw_quantile_score,
                            'cal_avg_coverage_gap': cal_avg_gap,
                            'cal_pit_entropy': cal_pit_entropy,
                            'cal_avg_quantile_score': cal_quantile_score
                        }
                        
                        # Add individual coverage metrics
                        for key, value in raw_coverages.items():
                            result[f'raw_{key}'] = value
                        for key, value in cal_coverages.items():
                            result[f'cal_{key}'] = value
                        
                        results.append(result)
                    except Exception as e:
                        print(f"Error processing {ISO}_{site}_{target_var}: {e}")
                        continue
                    
    
    # Save to CSV
    df = pd.DataFrame(results)
    output_path = os.path.join(metrics_folder, save_to)
    df.to_csv(output_path, index=False)
    print(f"Energy metrics saved to {output_path} ({len(results)} rows)")
    
    return df

if __name__ == "__main__":
    print("Starting metrics computation...")
    
    # Compute COVID metrics
    compute_covid_metrics()
    
    # Compute Energy metrics
    compute_energy_metrics()
    
    print("All metrics computed successfully!")
    

