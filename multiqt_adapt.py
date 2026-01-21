import numpy as np
from compute_metrics import PIT_score 
from sklearn.isotonic import IsotonicRegression 

import pdb

### pinball loss of the given vector of residuals at the given vector of levels
def pinball(residuals,levels):
    return levels*residuals - np.minimum(residuals, 0)

### Compute F^{-1}(tau) from the cdf F given by the quantiles in qs and the levels in levels.
def compute_quantile_from_PIT(tau,qs,levels, tol=0.01):
    if tau <= 0:
        return -float('inf')
    elif tau >=1:
        return float('inf')
    
    high = qs[-1]
    while PIT_score(high, qs, levels) < tau:
        high = high + 100

    low = qs[0]
    while PIT_score(low, qs, levels) > tau:
        low = low - 100

    while high-low > tol:
        mid = (high + low)/2
        score = PIT_score(mid, qs, levels)
        if score < tau:
            low = mid
        else:
            high = mid

    return (high + low)/2
    

### Input is vector Y of length T, vector of levels of length k, and Txk array of baselines b
### Return value is a tuple containing two Txk arrays providing the predicted quantile levels and the realized coverages. 
def MQ_adapt(Y,levels,b,w=200,gammas=np.array([0.001,0.002,0.004,0.008,0.0160,0.032,0.064,0.128])):
    ### Define constants
    T = len(Y)      ## num timesteps
    k = len(levels) ## num levels
    m = len(gammas) ## num experts

    ### Set parameter values
    unifs = np.random.uniform(size = 10000)
    c = np.mean([sum(pinball(U - levels,levels))**2 for U in unifs])
    eta = (np.log(2*w*m)/(w*c))**(1/2)
    delta = 1/(2*w)


    ### Define targets in alpha space
    betas = np.zeros(b.shape)
    for t in range(T):
        for i in range(k):
            betas[t,i] = PIT_score(Y[t], b[t,:], levels)

    ### Run method
    predicted_levels = np.zeros(b.shape)
    realized_coverages = np.zeros(b.shape)
    expert_alphas_hidden = np.tile(levels,(m, 1))  ### mxk array. each expert initialized using the given levels
    expert_alphas_played = np.tile(levels,(m, 1))  ### mxk array. each expert initialized using the given levels
    expert_weights = np.ones(m)/m
    for t in range(T):
        ### Get played values
        for i in range(m):
            iso = IsotonicRegression() 
            iso.fit(levels, expert_alphas_hidden[i,:])
            expert_alphas_played[i,:] = iso.predict(levels)

        
        chosen_expert = np.random.choice(np.arange(0, m), size=1, p=expert_weights)[0]
        predicted_levels[t,:] = expert_alphas_played[chosen_expert,:]
        realized_coverages[t,:] = (betas[t,:] <= predicted_levels[t,:])
        
        expert_losses = np.zeros(m)
        for i in range(m):
            expert_losses[i] = sum(pinball(betas[t,:] - expert_alphas_played[i,:], levels))
        
        ### update experts
        for i in range(m):
            expert_alphas_hidden[i,:] = expert_alphas_hidden[i,:] + gammas[i]*(levels-(betas[t,:] <= expert_alphas_played[i,:]))

        ### update weights
        expert_weights_tilde = expert_weights*np.exp(-eta*expert_losses)
        expert_weights = (1-delta)*expert_weights_tilde/sum(expert_weights_tilde) + delta/m

    predicted_quantiles = np.zeros(b.shape)
    for t in range(T):
        tol = 0.01
        for i in range(k):
            predicted_quantiles[t,i] = compute_quantile_from_PIT(predicted_levels[t,i],b[t,:],levels, tol=tol)

        # Check that quantiles are non-decreasing
        # If they are, decreasing binary search tolerance to get better quantile conversion
        # if np.diff(predicted_quantiles[t,:]).min() < 0:
        #     print("Quantiles are not ordered at time ", t)
        #     pdb.set_trace() 
        while np.any(np.diff(predicted_quantiles[t,:]) < 0):
            print('Time ', t, 'Decreasing binary search tolerance from ', tol, ' to ', tol/10)
            tol = tol/10
            for i in range(k):
                predicted_quantiles[t,i] = compute_quantile_from_PIT(predicted_levels[t,i],b[t,:],levels, tol=tol)
    
        assert not np.any(np.diff(predicted_quantiles[t,:]) < 0), "Quantiles are not ordered even after decreasing tolerance to "+str(tol)

    return predicted_quantiles

# Test
if __name__ == "__main__":
    # Example usage
    Y = np.array([10, 12, 9, 11, 13, 15, 14, 16, 18, 17])
    levels = np.array([0.1, 0.5, 0.9])
    b = np.array([[8, 10, 12],
                  [9, 11, 13],
                  [7, 9, 11],
                  [10, 12, 14],
                  [11, 13, 15],
                  [13, 15, 17],
                  [12, 14, 16],
                  [14, 16, 18],
                  [16, 18, 20],
                  [15, 17, 19]])
    
    predicted_levels = MQ_adapt(Y, levels, b)
    print("Quantile forecasts:\n", predicted_levels)


