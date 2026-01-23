import argparse
import numpy as np
import os
import time

import pdb

from utils import MultiQT, QT
from multiqt_adapt import MQ_adapt 


def run_experiment(args):
    cache_folder = os.path.join(args.cache_folder, args.dataset)
    if args.method == 'QT':
        cache_folder = os.path.join(cache_folder, 'QT')
    os.makedirs(cache_folder, exist_ok=True)

    if args.dataset == 'covid':
        from covid_utils import good_forecasters, states, levels, get_covid_data

        horizons = [1,2,3,4] # weeks ahead
        Yhat_type = 'quantile-specific'  # or 'point' for median forecasts

        total_experiments = len(good_forecasters) * len(states) * len(horizons)
        experiment_count = 0

        for forecaster in good_forecasters:
            for state in states:
                for horizon in horizons:

                    experiment_count += 1
                    print(f"Experiment {experiment_count}/{total_experiments}: {forecaster=}, {state=}, {horizon=}")
                    save_path = f"{cache_folder}/{forecaster}_{state}_Yhat={Yhat_type}_h={horizon}.npy"

                    # If the file already exists, skip the experiment (unless --override_saved is specified)
                    if not args.override_saved and os.path.exists(save_path):
                        print(f"   --> Skipping, already exists: {save_path}")
                        continue

                    Y, Yhat = get_covid_data(forecaster, state, horizon, 
                                                Yhat_type=Yhat_type, 
                                                levels=levels)
                    
                    # Account for delayed feedback
                    if args.no_delay:
                        delay = None
                    else:
                        # Add in delay of horizon length. h=1 corresponds to no delay (feedback is available immediately)
                        delay = [ [] for _ in range(horizon-1) ] + [ [t] for t in range(len(Y)-(horizon-1)) ]

                    if args.method == 'QT':
                        Y_forecast = QT(Y, levels, Yhat=Yhat, lr='adaptive+', lr_window=args.lr_window, q0=args.q0, delay=delay)
                    else:
                        # Y_forecast = MultiQT(Y, levels, Yhat=Yhat, lr='adaptive+', lr_window=args.lr_window, q0=args.q0, delay=delay)

                        # REPLACED WITH ISAAC'S METHOD
                        Y_forecast = MQ_adapt(Y,levels,Yhat.T)
                        Y_forecast = Y_forecast.T

                    # Save results
                    np.save(save_path, Y_forecast)
                    print(f"   --> Saved calibrated forecasts to {save_path}")
    
    elif args.dataset == 'energy':
        from energy_utils import get_energy_data, ISO_to_sites, levels

        ISO = 'ERCOT' # Other options available
        Yhat_type = 'quantile-specific'  
        target_variables = ['Wind', 'Solar']

        # Hours in UTC. Hours [6, 18) have no delayed feedback and hours >=18 or < 6 have a lag of 1. 
        # Actually, the boundary shifts by 1 depending on DST, so we choose hours that avoid the boundary
        hours = [0, 4, 8, 12, 16, 20] 
        total_experiments = len(hours) * ( len(ISO_to_sites[ISO]['Wind']) + len(ISO_to_sites[ISO]['Solar']) )
        experiment_count = 0

        for target_variable in target_variables:
            for site in ISO_to_sites[ISO][target_variable]:
                for hour in hours:

                    experiment_count += 1
                    print(f"Experiment {experiment_count}/{total_experiments}: {ISO=}, {site=}, {target_variable=}, {hour=}")
                    save_path = f"{cache_folder}/{ISO}_{site}_{target_variable}_Yhat={Yhat_type}_hour={hour}.npy"
                    # Check if the file already exists
                    if os.path.exists(save_path):
                        print(f"   --> Skipping, already exists: {save_path}")
                        continue

                    Y, Yhat = get_energy_data(ISO, site, hour=hour, target_variable=target_variable, 
                                              Yhat_type=Yhat_type, return_df=False)
                    
                    if args.no_delay:
                        delay = None
                    else:
                        # Hours [6, 18) have no delayed feedback and hours >=18 or < 6 have a lag of 1 (+/- offset depending on DST)
                        if hour >=6 and hour < 18:
                            delay = None
                        else:
                            print(f"   --> Using delay of 1 timestep for hour={hour}")
                            delay = [ [] ] + [ [t] for t in range(len(Y)-1) ]

                    if args.method == 'QT':
                        Y_forecast = QT(Y, levels, Yhat=Yhat, lr='adaptive+', lr_window=args.lr_window, 
                                        q0=args.q0, delay=delay)
                    else:
                        # Y_forecast = MultiQT(Y, levels, Yhat=Yhat, lr='adaptive+', 
                        #                 lr_window=args.lr_window, q0=args.q0, delay=delay)

                        # REPLACED WITH ISAAC'S METHOD
                        Y_forecast = MQ_adapt(Y,levels,Yhat.T)
                        Y_forecast = Y_forecast.T

                    # Save results
                    np.save(save_path, Y_forecast)
                    print(f"   -->  Saved calibrated forecasts to {save_path}")

if __name__ == "__main__":
    st = time.time()
    parser = argparse.ArgumentParser(description='Run MultiQT pipeline')
    parser.add_argument('--dataset', type=str, default='covid', choices=['covid', 'energy'])

    # Unless you specify --no_delay, the pipeline will account for delay in feedback
    parser.add_argument('--no_delay', dest='delay', action='store_false', help='Ignore delay in feedback')
    parser.set_defaults(no_delay=False)

    # Whether to override saved results. By default, if save results exist, the experiment is skipped
    parser.add_argument('--override_saved', dest='override_saved', action='store_true', help='Override saved results')
    parser.set_defaults(override_saved=False)


    # Optional arguments
    parser.add_argument('--method', type=str, default='MultiQT', choices=['MultiQT', 'QT'])
    parser.add_argument('--lr_window', type=int, default=50,
                        help='Window size for adaptive learning rate')
    parser.add_argument('--q0', type=float, default=0,
                        help='Initial quantile value')
    parser.add_argument('--cache_folder', type=str, default='cache',
                       help='Folder to cache results')

    args = parser.parse_args()

    run_experiment(args)
    print(f"Total time taken: {(time.time() - st) / 60:.2f} minutes")

