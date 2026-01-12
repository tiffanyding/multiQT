import pandas as pd
import numpy as np
import pdb

good_forecasters = ['PSI-DRAFT', 'JHU_IDD-CovidSP', 'UCSD_NEU-DeepGLEAM', 
 'RobertWalraven-ESG', 'JHUAPL-Bucky', 'UA-EpiCovDA', 'DDS-NBDS', 'COVIDhub-baseline', 
 'CEID-Walk', 'CovidAnalytics-DELPHI', 'BPagano-RtDriven', 'LANL-GrowthRate',
 'SteveMcConnell-CovidComplete', 'Karlen-pypm', 'COVIDhub-ensemble']

states =['ak', 'al', 'ar', 'az', 'ca', 'co', 'ct', 'de', 'fl', 'ga', 'hi', 'ia', 
        'id', 'il', 'in', 'ks', 'ky', 'la', 'ma', 'md', 'me', 'mi', 'mn', 'mo',
        'ms', 'mt', 'nc', 'nd', 'ne', 'nh', 'nj', 'nm', 'nv', 'ny', 'oh', 'ok',
        'or', 'pa', 'ri', 'sc', 'sd', 'tn', 'tx', 'ut', 'va', 'vt', 'wa', 'wi', 
        'wv', 'wy']

levels = [.01, .025, .05, .1, .15, .2, .25, .3, .35, .4, .45, .5, .55, .6, .65,
          .7, .75, .8, .85, .9, .95, .975, .99]

def get_population(state):
    # Dict of state to population in 2020 census
    # from https://www.census.gov/data/tables/2020/dec/2020-apportionment-data.html
    STATE_POPS_2020 = {
    "AL": 5024279,  "AK": 733391,   "AZ": 7151502,  "AR": 3011524,
    "CA": 39538223, "CO": 5773714,  "CT": 3605944,  "DE": 989948,
    "DC": 689545,   "FL": 21538187, "GA": 10711908, "HI": 1455271,
    "ID": 1839106,  "IL": 12812508, "IN": 6785528,  "IA": 3190369,
    "KS": 2937880,  "KY": 4505836,  "LA": 4657757,  "ME": 1362359,
    "MD": 6177224,  "MA": 7029917,  "MI": 10077331, "MN": 5706494,
    "MS": 2961279,  "MO": 6154913,  "MT": 1084225,  "NE": 1961504,
    "NV": 3104614,  "NH": 1377529,  "NJ": 9288994,  "NM": 2117522,
    "NY": 20201249, "NC": 10439388, "ND": 779094,   "OH": 11799448,
    "OK": 3959353,  "OR": 4237256,  "PA": 13002700, "RI": 1097379,
    "SC": 5118425,  "SD": 886667,   "TN": 6910840,  "TX": 29145505,
    "UT": 3271616,  "VT": 643077,   "VA": 8631393,  "WA": 7705281,
    "WV": 1793716,  "WI": 5893718,  "WY": 576851,
    }

    return STATE_POPS_2020[state.upper()]

def get_covid_data(forecaster, state,  horizon, Yhat_type='quantile-specific', 
                        levels=None, truncate_negY=True, return_df=False):
    '''
    Get forecasts and true values for COVID-19 deaths

    Inputs:
        - forecaster: string, name of forecaster. Options are anything in good_forecasters
        - state: lowercase state code (e.g., 'ca')
        - horizon: int, forecast horizon in weeks (1,2,3 or 4)
        - Yhat_type: str, type of forecast to return ('point' (median), 'quantile-specific', 'none')
        - levels: list of quantile levels (e.g., [.01, .025, .05, ...]), only needed if Yhat_type is 'quantile-specific'
        - truncate_negY: boolean, whether to truncate negative values in Y to 0
        - return_df: boolean, whether to return a DataFrame with time stamps, actual values, and forecast values
    '''
    # Load in csv for specified forecaster
    df = pd.read_csv(f"data/covid/{forecaster}.csv")

    if levels is None:
        levels = [.01, .025, .05, .1, .15, .2, .25, .3, .35, .4, .45, .5, .55, .6, .65,
                    .7, .75, .8, .85, .9, .95, .975, .99]

    # Extract the relevant values for the specified state and forecast horizon
    idx =(df['geo_value'] == state.lower()) & (df['h'] == horizon)
    Y = df[idx]['deaths'].values
    if Yhat_type == 'point':
        Yhat = df[idx][f'forecast_0.5'].values 
        Yhat = np.tile(Yhat, (len(levels), 1)) # Tile the forecast to match the number of levels
    elif Yhat_type == 'quantile-specific':
        Yhat = df[idx][[f'forecast_{level}' for level in levels]].values.T 
    elif Yhat_type == 'none':
        Yhat = np.zeros(Y.shape) # len(Y) x 1
        Yhat = np.tile(Yhat, (len(levels), 1)) # Tile the forecast to match the number of levels
    # Yhat is len(Y) x num_levels

    if truncate_negY:
        Y[Y < 0] = 0

    if return_df:
        # Create a DataFrame with the actuals and forecasts
        cols = {'time': df[idx]['target_date'].values, 'Y': Y}

        for i, level in enumerate(levels):
            cols[f'p{int(level*100)}'] = Yhat[i,:]
        df = pd.DataFrame(cols)
        return df

    return Y, Yhat


if __name__ == "__main__":
    ## Example usage

    ## Get forecasts and true values
    Y, Yhat = get_covid_data('COVIDhub-ensemble', 'ca', 1, Yhat_type='quantile-specific')
    print("True values:", Y)
    print("Forecasts:", Yhat)

    ## Get forecasts and true values and also get time stamps
    df = get_covid_data('COVIDhub-ensemble', 'ca', 1, Yhat_type='quantile-specific',
                                 return_df=True)
    print(df.head())
    