import matplotlib.pyplot as plt
import numpy as np
import pdb

from sklearn.isotonic import IsotonicRegression 

#==============================================================================
#          General             
#==============================================================================

def quantile_loss(tau, x):
    '''
    Quantile loss:
        l_tau(x) = tau * x if x >= 0 else (1 - tau) * (-x)
    '''
    return np.where(x >= 0, tau * x, (1 - tau) * (-x))

def buffered_isotonic_regression(v, delta):
    '''
    Returns the solution to
        min_{u} ||u - v||^2   s.t.   u[i+1] - u[i] >= delta
    via the shifted‐PAVA trick.
    '''
    v = np.asarray(v, float)
    m = len(v)
    shifts = np.arange(m) * delta

    # shift down
    y = v - shifts

    # run PAVA on y
    ir = IsotonicRegression(increasing=True, out_of_bounds='clip')
    y_hat = ir.fit_transform(np.arange(m), y)

    # un‐shift up
    return y_hat + shifts

#==============================================================================
#          Methods              
#==============================================================================

def get_lr(lr, S=None, level=None, lr_window=None):
    '''
    Get learning rate based on option specified by lr. 
    * If lr is a positive number, return lr 
    * If lr is 'adaptive' or 'adaptive+', compute adaptive learning rate based on 
      past forecast errors in S.
    
    Inputs:
        - lr: positive number or 'adaptive' or 'adaptive+'
        - S: num levels x T array of forecast errors (Y - base forecasts). Only needed if 
            lr is 'adaptive' or 'adaptive+'
        - level: quantile level 
        - lr_window: int, window size for choosing adaptive learning rate. Only needed if lr is 
            'adaptive' or 'adaptive+'
    '''
    
    # If lr is a positive number, use it as a constant learning rate
    if isinstance(lr, (int, float)) and lr > 0:
        return lr
    elif lr == 'adaptive':   
        # Rule from "Conformal PID Control", Section 2.1
        lr = 0.1 * np.max(np.abs(S[max(0,len(S)-lr_window):]))
    elif lr == 'adaptive+': 
        # replace max with 90% quantile to be more robust to outliers
        lr = 0.1 * np.quantile(np.abs(S[max(0,len(S)-lr_window):]), 0.9)
        # To ensure that the learning rate is not 0 (at least 0.1)
        lr = max(lr, 0.1) 
    else:
        raise ValueError("Invalid learning rate option.")

    assert lr > 0, f"Learning rate must be positive, got {lr}"

    return lr

def apply_projection(v, levels, projection):
    '''
    Apply projection to unordered quantiles v

    Inputs:
        - v: (m,) array of quantile values
        - levels: (m,) array of quantile levels
        - projection: how unordered quantiles are mapped to new quantiles. 
            Options: 'none', 'sort', 'isotonic', 'buffered-isotonic'
    '''
    if projection == 'none': # This corresponds to Quantile Tracker
        return v
    elif projection == 'sort':
        return np.sort(v)
    elif projection == 'isotonic':
        ir = IsotonicRegression()
        try:
            ir.fit(levels, v)
        except:
            print('ISOTONIC REGRESSION FAILED')
            pdb.set_trace()
        return ir.predict(levels)
    elif projection == 'buffered-isotonic':
        # Isotonic regression but with at least a buffer of 1 between adjacent quantiles
        buffer = 1 # This is a hyperparameter that can be tuned
        try:
            return buffered_isotonic_regression(v, buffer)
        except Exception:
            print('BUFFERED-ISOTONIC REGRESSION FAILED')
            pdb.set_trace()
    else:
        raise ValueError("Invalid projection.")

def projectedQT(Y, levels, Yhat=None, lr='adaptive++', projection='isotonic', eval_grad_at='played',
                delay=None, lr_window=50, q0=0, return_extra_info=False):
    '''
    Run projected Quantile Tracker for multi-quantile level, which is a meta-algorithm that
    includes MultiQT and vanilla Quantile Tracker as instantiations.

    Inputs:
        - Y: (T,) array of values to track
        - levels: (m,) array of quantile levels to track (e.g., [0.1, 0.5, 0.9])
        - Yhat: (m,T) array of forecasts, where Yhat[i,t] is the levels[i]-quantile forecast for Y at time t
        - lr: learning rate, which is either a positive number or 'adaptive' or 'adaptive+'
        - projection: how unordered quantiles are mapped to ordered quantiles. 
            Options: 'none,'sort','isotonic', 'buffered-isotonic'
        - eval_grad_at: which iterate to evaluate gradient at. Options: 'hidden' (unordered) or 'played' (ordered)
        - delay: list of lists. delay[t] is a list of time steps whose Y values are revealed at time t.
        - lr_window: window size for adaptive learning rate
        - q0: scalar or (m,) array of initial quantile values
        - return_extra_info: Boolean. If True, additionally return dict with learning rates
            and hidden iterates over time.
    '''
    
    T = len(Y)
    num_levels = len(levels)

    if Yhat is None:
        Yhat = np.zeros((num_levels, T))

    S = Y - Yhat # Forecast error, will be num_levels x T

    if delay is None:
        delay = [[t] for t in range(T)] # Each Y[t] is observed at time t

    hidden = np.zeros((num_levels, T)) # hidden sequence
    played = np.zeros(hidden.shape) # played sequence
    lrs = np.zeros((num_levels, T)) # Store learning rates
    gradients = np.zeros(hidden.shape) # Store (negative) gradients 

    # Initialize played and hidden sequences
    hidden[0,:] = q0
    played[0,:] = q0

    observed_idx = [] # keep track of observed indices

    for t in range(T-1):
        observed_idx += delay[t] 

        ## Step 1: Update the hidden vector
        for i, tau in enumerate(levels):

            # Compute and store gradient
            if eval_grad_at == 'hidden':
                eval_point = hidden[i,t] + Yhat[i,t]
            elif eval_grad_at == 'played':
                eval_point = played[i,t]
            else:
                raise ValueError("Invalid eval_grad_at.")
            gradients[i,t] = (Y[t] > eval_point) - (1 - tau)

            # If there are present/past Y values revealed at time t
            hidden[i,t+1] = hidden[i,t] # Start with no update
            if len(delay[t]) > 0:
               
                # Get learning rate
                if len(observed_idx) > 0:
                    lrs[i,t] = get_lr(lr, S[i,observed_idx], tau, lr_window)
                    if lr == 'adaptive+':
                        # Use the same lr for all levels
                        if i == 0: # Pass in past residuals from *all* levels. We will then take the 90% quantile inside get_lr
                            lrs[i,t] = get_lr(lr, S[:,observed_idx], tau, lr_window)
                        else:
                            lrs[i,t] = lrs[0,t]

                # Apply gradient step for each revealed Y value
                for s in delay[t]:
                    assert s <= t, "delay[t] should contain time indices <= t"
                    hidden[i,t+1] += lrs[i,t] * gradients[i,s]
            
        ## Step 2: Obtain the played vector for time t+1
        played[:,t+1] = apply_projection(hidden[:,t+1] + Yhat[:,t+1], levels, projection)
        
    Y_forecast = played
    
    if return_extra_info:
        extra_info = {'lrs': lrs,
                      'hidden': hidden}
        return Y_forecast, extra_info
    
    return Y_forecast

def QT(Y, levels, Yhat, lr, lr_window=50, q0=0, delay=None, return_extra_info=False):
    '''
    Run Quantile Tracker at multiple quantile levels (no quantile ordering enforced)
    
    Inputs:
        - Y: (T,) array of values to track
        - levels: (m,) array of quantile levels to track
        - Yhat: (T,m) array of forecasts, where Yhat[:,i] is the forecast for Y at quantile level levels[i]
        - lr: learning rate
        - lr_window: window size for adaptive learning rate
        - q0: scalar or (m,) array of initial quantile values
    '''
   
    return projectedQT(Y, levels, Yhat, lr, projection='none', eval_grad_at='played', # this could be 'hidden', as they are the same for projection='none'
                        delay=delay, lr_window=lr_window, q0=q0, return_extra_info=return_extra_info)


def MultiQT(Y, levels, Yhat=None, lr='adaptive++', lr_window=50, q0=0, delay=None, return_extra_info=False):
    '''
    Run Multi-Level Quantile Tracker.
    
    Inputs:
        - Y: (T,) array of values to track
        - levels: (m,) array of quantile levels to track
        - Yhat: None, or (T,m) array of forecasts, where Yhat[:,i] is the forecast for Y at quantile level levels[i]
        - lr: learning rate
        - lr_window: window size for adaptive learning rate
        - q0: scalar or (m,) array of initial quantile values
    '''
    return projectedQT(Y, levels, Yhat, lr=lr, projection='isotonic', eval_grad_at='played', 
                        delay=delay, lr_window=lr_window, q0=q0, return_extra_info=return_extra_info)

#==============================================================================
#          Evaluation and Plotting            
#==============================================================================

def compute_coverage(Y, forecasts, levels, plot_results=True, ax=None, label=None):
    '''
    Compute the empirical coverage of forecasts at each quantile level, 
    and optionally plot the results.

    Inputs: 
        - Y: (T,) array of true values
        - forecasts: (T, m) array of forecasts, where forecasts[:,i] is the forecast for Y 
          at quantile level levels[i]
        - levels: (m,) array of quantile levels 
        - plot_results: boolean, whether to make plot of actual vs. desired coverage
        - ax: (used if plot_results is True) matplotlib axes object to plot on, or None
        - label: (used if plot_results is True) label for the plotted line 
    '''

    if len(forecasts.shape) == 2 and forecasts.shape[1] == len(Y):
        forecasts = forecasts.T

    coverages = np.mean(forecasts >= Y[:, None], axis=0) 
    # print('Exact equality ct', np.sum(forecasts >= Y[:, None], axis=0))

    if plot_results:
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))
        if label is None:
            label = 'Actual coverage'
            ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect calibration') 
        ax.plot(levels, coverages, label=label)
        ax.legend()
        ax.set_xlabel('Desired coverage')
        ax.set_ylabel('Actual coverage')
        
    return coverages

def compute_running_coverage(Y, forecasts):
    '''
    Computes average coverage up through time t, for each t.

    Inputs: 
        - Y: (T,) array of true values
        - forecasts: (T, m) array of forecasts, where forecasts[:,i] is the forecast for Y 
          at quantile level levels[i]
    ''' 

    if len(forecasts.shape) == 2 and forecasts.shape[1] == len(Y):
        forecasts = forecasts.T

    coverage_indicators = forecasts >= Y[:, None]   

    # Take cumulative average along axis 0 by using np.cumsum and dividing by np.arange
    running_coverage = np.cumsum(coverage_indicators, axis=0) / np.arange(1, len(Y)+1)[:, None]
    
    # Sanity check that last row of coverages is the same as compute_coverage
    assert np.all(running_coverage[-1,:] == np.mean(forecasts >= Y[:, None], axis=0))

    return running_coverage

def compute_crossing_frac(quantiles):
    '''
    Computes fraction of time steps where quantiles are not in increasing order (i.e., they cross)

    Inputs:
        - quantiles: (T, m) array of quantile forecasts, where quantiles[t,i] is the 
          forecast for Y at time t and quantile level levels[i]
    '''
    # Transpose if shape[1] > shape[0]
    if quantiles.shape[1] > quantiles.shape[0]:
        quantiles = quantiles.T
    T = len(quantiles)
    crossed = np.zeros((T,))
    for t in range(T):
        crossed[t] = ~np.all(np.sort(quantiles[t,:]) == quantiles[t,:])

    return np.mean(crossed)

def compute_overlap_frac(quantiles):
    '''
    Computes fraction of time steps where two or more quantiles are equal (i.e., they overlap)

    Inputs:
        - quantiles: (T, m) array of quantile forecasts, where quantiles[t,i] is the 
          forecast for Y at time t and quantile level levels[i]
    '''
    # Computes fraction of time steps where two quantiles are equal
    T = len(quantiles)
    num_quantiles = len(quantiles[0])
    overlap = np.zeros((T,))
    for t in range(T):
        overlap[t] = len(np.unique(quantiles[t,:])) < num_quantiles

    return np.mean(overlap)


def plot_calibration(Ys, Y_forecasts, ax=None, color='royalblue', title=None, 
                     single_state=False, plot_yequalsx=True, label=None, alpha=0.5,
                     levels=None):
    if levels is None:
        # levels for COVID
        levels = [.01, .025, .05, .1, .15, .2, .25, .3, .35, .4, .45, .5, .55, .6, .65,
          .7, .75, .8, .85, .9, .95, .975, .99]
    if ax is None:
        _, ax = plt.subplots(figsize=(2.5, 2.5))
    if single_state: # Y is an array
        coverage = compute_coverage(Ys, Y_forecasts, levels, plot_results=False)
        ax.plot(levels, coverage, color=color, alpha=alpha, label=label)
    else: # Assume Ys is a dictionary of arrays, one for each state
        for s, st in enumerate(Ys.keys()):  
            coverage = compute_coverage(Ys[st], Y_forecasts[st], levels, plot_results=False)
            ax.plot(levels, coverage, color=color, alpha=alpha)
    ax.set_xlabel('Desired coverage')
    ax.set_ylabel('Actual coverage')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Set ticks 
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_xticklabels(['0', '','','', '1'])
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.set_yticklabels(['0', '','','', '1'])

    # despine top and right, add grid
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2)
    
    ax.set_title(title, fontsize=8)
    if plot_yequalsx:
        ax.plot([0, 1], [0, 1], linestyle='--', color='black', label='Perfect calibration') 


def plot_forecasts_shaded(Y, forecasts, levels, ax=None, title=None, plot_Y=True,
                        linewidth=0.8, base_color="tab:blue", 
                        alpha_min=0.08, alpha_max=0.35, forecast_label=None,
                        xlabel='Time', ylabel='Deaths', legend_loc='upper right'):
    '''
    Plot probabilistic forecasts as shaded quantile bands.
    Inputs:
        - Y: (T,) array of true values
        - forecasts: (T, m) array of forecasts, where forecasts[:,i] is the forecast for Y at quantile level levels[i]
        - levels: (m,) array of quantile levels
        - ax: matplotlib Axes, optional
        - title: str, optional
        - plot_Y: bool, default True. Whether to draw the ground-truth line.
        - linewidth: float, default 0.8. Width of the median line (if present) and Y.
        - base_color: str or tuple, default "tab:blue". Colour used for all bands/lines; opacity varies automatically.
        - alpha_min, alpha_max: float. Opacity for the outermost and innermost bands, respectively.
        - forecast_label: str, label for the forecast bands to show in legend
        - xlabel, ylabel: str, labels for x and y axes
        - legend_loc: str or None, location for legend (e.g., 'upper right'), or None to not show legend
    '''
    # ---------- set up figure ----------
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 3))

    # ---------- sanity checks ----------
    levels = np.asarray(levels)
    forecasts = np.asarray(forecasts)
    if forecasts.shape[0] != len(levels):
        raise ValueError(
            f"forecasts has {forecasts.shape[0]} rows but levels has {len(levels)} entries"
        )

    # ---------- sort by quantile ----------
    order = np.argsort(levels)
    levels = levels[order]
    forecasts = forecasts[order]

    # ---------- pair up symmetric quantiles ----------
    n_pairs = len(levels) // 2          # how many (lower, upper) bands
    alphas = np.linspace(alpha_min, alpha_max, n_pairs)[::-1]  # opacities for each band

    # Draw the widest band first so narrower ones sit on top.
    time = np.arange(forecasts.shape[1])
    for rank in range(n_pairs - 1, -1, -1):
        lower = forecasts[rank, :]
        upper = forecasts[-(rank + 1), :]
        ax.fill_between(
            time,
            lower,
            upper,
            color=base_color,
            alpha=alphas[n_pairs - 1 - rank],
            linewidth=0,
            label =  forecast_label if rank == 0 else None,
        )

    # ---------- median line (if odd number of levels) ----------
    if len(levels) % 2 == 1:
        median = forecasts[n_pairs, :]
        ax.plot(time, median, color=base_color, linewidth=linewidth)
                #, label="median")

    # ---------- truth ----------
    if plot_Y:
        ax.plot(time, Y, color="black", linewidth=linewidth, label="True value")

    # ---------- cosmetics ----------
    ax.set_title(title or "")
    ax.set_xlim(time[0], time[-1])
    if legend_loc is not None:
        ax.legend(loc=legend_loc, frameon=False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)  
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linewidth=0.25, linestyle="--", alpha=0.5)
    return ax
