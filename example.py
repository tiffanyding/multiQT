import numpy as np
from sklearn.isotonic import IsotonicRegression 

# (Optional) For visualizing forecasts. Not necessary for running the main algorithm.
import matplotlib.pyplot as plt
from utils import plot_forecasts_shaded 

# Note: Numpy and scikit-learn should be installed in your environment before running this code. 

def isotonic_regression(y):
    '''
    Perform isotonic regression on the input array y.
    
    Inputs: y, a 1D numpy array of values to be ordered.
    Output: the ordered vector with the smallest l2 distance from y 
    '''
    ir = IsotonicRegression()
    x = np.arange(len(y)) # any increasing sequence will do
    ir.fit(x, y)

    return ir.predict(x)
    

def multiQT(Y, levels, base_forecasts=None, lr='adaptive+', lr_window=50, init=0, delay=None):
    '''
    Run MultiQT algorithm from Ding, Gibbs, Tibshirani (2025). 

    Inputs:
        - Y: (T,) array of values to track
        - levels: (num_levels,) array of quantile levels to track
        - Yhat: (num_levels, T) array of forecasts, where Yhat[i,t] is levels[i]-quantile forecast for Y at time t
        - lr: learning rate. Either a scalar or 'adaptive+'. If 'adaptive+', use the adaptive learning rate 
              heuristic from Ding, Gibbs, Tibshirani (2025).
        - lr_window: window size for adaptive learning rate
        - delay: list of lists. delay[t] is a list of time steps whose Y values are revealed at time t.
        - init: scalar or (num_levels,) array of values used as the first hidden iterate
        - return_extra_info: Boolean. If True, return extra information.
    '''

    ## Setup 
    T = len(Y)
    num_levels = len(levels)
    if base_forecasts is None:
        base_forecasts = np.zeros((num_levels, T))
    if delay is None:
        delay = [[t] for t in range(T)] # Each Y[t] is observed at time t

    theta_tilde = np.zeros((num_levels, T)) # hidden offsets
    q = np.zeros((num_levels, T)) # played forecasts

    ## Initialize hidden sequence
    theta_tilde[0,:] = init

    observed_idx = [] # When delay is not None, it is necessary to keep track of which
                      # Y values have been observed so far
   
    ## Main loop
    for t in range(T):
        observed_idx += delay[t] # Update observed indices with any new observations at time t

        q[:,t] = isotonic_regression(base_forecasts[:,t] + theta_tilde[:,t]) 
        if t < T - 1:

            theta_tilde[:,t+1] = theta_tilde[:,t] # Start with no update

            # Check if there are any Y values revealed at time t. If so, we update the hidden sequence
            if len(delay[t]) > 0:

                # Get current learning rate. Call it eta.
                if isinstance(lr, (int, float)) and lr > 0:
                    eta = lr
                elif lr == 'adaptive+':
                    residuals = (Y - base_forecasts)[:,observed_idx] # Only use residuals from already observed Y values
                    eta = max(0.1, 0.1 * np.quantile(np.abs(residuals[:, max(0,t+1-lr_window):t+1]), 0.9))
                else:
                    raise ValueError("Invalid learning rate. Must be a positive scalar or 'adaptive+'.")

                # Apply gradient step for each revealed Y value
                for s in delay[t]:
                    assert s <= t, "delay[t] should contain time indices <= t"
                    for i, alpha in enumerate(levels):
                        theta_tilde[i,t+1] -= eta * ((Y[t] <= q[i,t]) - alpha) 

    ## Return forecasts
    return q


if __name__ == "__main__":

    ## Example usage
    
    levels = [0.1, 0.25, 0.5, 0.75, 0.9] # Quantile levels to track
    
    # Generated simulated Y's: sine wave plus noise
    T = 100
    np.random.seed(0)
    Y = 5 + np.sin(np.linspace(0, 4 * np.pi, T)) + 0.5 * np.random.randn(T)

    # Generate base forecasts: use the mean of past Y values 
    base_forecasts = np.zeros((len(levels), T))
    for t in range(1, T):
        base_forecasts[:,t] = np.mean(Y[:t])

    # (Optional) Impose a constant feedback delay of d, meaning Y[t] is only observed at time t+d
    d = 3   # Set d=0 for standard no-delay setting
    delay = [ [] for _ in range(d) ] + [ [t] for t in range(len(Y)-d) ]

    # Run MultiQT and print forecasts 
    multiQT_forecasts = multiQT(Y, levels, base_forecasts=base_forecasts, delay=delay)
    print("Forecasts:", multiQT_forecasts)

    # (Optional) Visualize MultiQT forecasts
    plot_forecasts_shaded(Y, multiQT_forecasts, levels, ylabel='Y')
    plt.show()
    
    