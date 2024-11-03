import datetime
import numpy
import pandas
import pandas
import matplotlib.pyplot as plt
#import seaborn as sns
import warnings

#sns.set(palette='viridis')
warnings.filterwarnings("ignore")

import jumpdiffcalibrator as jdcal



if __name__ == '__main__':
    # ----- set parameters
    #s0 = 100
    nsteps = 5
    nsim = 10000
    r = 0.05
    q = 0.02
    # ----- calibrate parameters
    n_mcmc_steps = 10000
    burn_in = 5000

    # ----- load data
    df = pandas.read_csv(r'D:\TU_DORTMUND\ThesisCode\aapl_data.csv')
    df['Date'] = pandas.to_datetime(df['Date'], dayfirst=True)
    df.sort_values(['Date'], ascending=True, inplace=True)
    date = df['Date'].values
    price = df["Adj Close"].values
    all_params = None
    for i in range(1000, len(price)-5):
        print(i)
        print('###################################')
        stock_data = price[i-1000:i]
        s0 = stock_data[-1]

        if i%5 == 0:
            heston_cal = jdcal.HestonCalibrator(price_series=stock_data, cost_of_carry=r - q)
            heston_cal.calibrate(n_mcmc_steps=n_mcmc_steps, burn_in=burn_in)
            all_params = heston_cal.params_dict
            mu = all_params.get("mu_final")
            kappa = all_params.get("kappa_final")
            theta = all_params.get("theta_final")
            sigma = all_params.get("volvol_final")
            rho = all_params.get("rho_final")
        else:
            heston_cal = jdcal.HestonCalibrator(parameter_dict = all_params, price_series=stock_data, cost_of_carry=r - q)
        print(all_params)
        pandas.DataFrame(all_params, index = [0]).to_csv(f"D:\TU_DORTMUND\Thesis\Data\params\Heston\params_{date[i].astype('datetime64[s]').astype(datetime.datetime).strftime('%Y-%m-%d')}.csv", index = False)
        simulated_paths, simulated_variances = heston_cal.get_paths(s0=s0, nsteps=nsteps, nsim=nsim, risk_neutral=False)
        simulated_paths = pandas.DataFrame(simulated_paths).transpose()
        simulated_paths['PRICE'] = price[i - 1:i + 5]
        pandas.DataFrame(simulated_paths).to_csv(f"D:\TU_DORTMUND\Thesis\Data\simulation\price\Heston\price_{date[i].astype('datetime64[s]').astype(datetime.datetime).strftime('%Y-%m-%d')}.csv", index =False)
        pandas.DataFrame(simulated_variances).transpose().to_csv(rf"D:\TU_DORTMUND\Thesis\Data\simulation\vol\Heston\vol_{date[i].astype('datetime64[s]').astype(datetime.datetime).strftime('%Y-%m-%d')}.csv", index = False)

### date is the date that simulation starts + 1
    peng

    # ================================== Heston CRPS ====================================

    import glob
    import os

    # Folder path containing CSV files
    folder_path = 'D:\TU_DORTMUND\Thesis\Data\simulation\price\Heston'

    # Get a list of all CSV files in the folder
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    price = dict()
    price['DATE'] = []
    price['PRICE'] = []
    df_predict = pandas.DataFrame()
    for file in csv_files:
        df = pandas.read_csv(file)
        df_log = numpy.log(df).diff()
        col_name =file[-14:-4]
        df_temp = pandas.DataFrame(df_log.iloc[-1][:-1])
        df_temp.columns = [col_name]
        df_predict = pandas.concat([df_predict, df_temp], axis =1)
        price['DATE'].append(col_name)
        price['PRICE'].append(df.iloc[-1][-1])

    pandas.DataFrame(price).plot()


    ########### CRPS ##############

    import glob
    import os

    # Folder path containing CSV files
    folder_path = 'D:\TU_DORTMUND\Thesis\Data\simulation\price\Heston'

    # Get a list of all CSV files in the folder
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    price = dict()
    price['DATE'] = []
    price['PRICE'] = []
    df_predict = pandas.DataFrame()
    for file in csv_files:
        df = pandas.read_csv(file)
        #df_log = numpy.log(df).diff()
        col_name =file[-14:-4]
        #df_temp = pandas.DataFrame(df_log.iloc[-1][:-1])
        df_temp = pandas.DataFrame(df.iloc[-1][:-1])
        df_temp.columns = [col_name]
        df_predict = pandas.concat([df_predict, df_temp], axis =1)
        price['DATE'].append(col_name)
        price['PRICE'].append(df.iloc[-1][-1])

    real_price = pandas.DataFrame(price)


    def suppress_warnings(msg=None):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', msg)
            yield
    def _crps_ensemble_vectorized(observations, forecasts, weights=1):
        """
        An alternative but simpler implementation of CRPS for testing purposes

        This implementation is based on the identity:

        .. math::
            CRPS(F, x) = E_F|X - x| - 1/2 * E_F|X - X'|

        where X and X' denote independent random variables drawn from the forecast
        distribution F, and E_F denotes the expectation value under F.

        Hence it has runtime O(n^2) instead of O(n log(n)) where n is the number of
        ensemble members.

        Reference
        ---------
        Tilmann Gneiting and Adrian E. Raftery. Strictly proper scoring rules,
            prediction, and estimation, 2005. University of Washington Department of
            Statistics Technical Report no. 463R.
            https://www.stat.washington.edu/research/reports/2004/tr463R.pdf
        """
        observations = numpy.asarray(observations)
        forecasts = numpy.asarray(forecasts)
        weights = numpy.asarray(weights)
        if weights.ndim > 0:
            weights = numpy.where(~numpy.isnan(forecasts), weights, numpy.nan)
            weights = weights / numpy.nanmean(weights, axis=-1, keepdims=True)

        if observations.ndim == forecasts.ndim - 1:
            # sum over the last axis
            assert observations.shape == forecasts.shape[:-1]
            observations = observations[..., numpy.newaxis]
            with suppress_warnings('Mean of empty slice'):
                print('step1')
                score = numpy.nanmean(weights * abs(forecasts - observations), -1)
            # insert new axes along last and second to last forecast dimensions so
            # forecasts_diff expands with the array broadcasting
            forecasts_diff = (numpy.expand_dims(forecasts, -1) -
                              numpy.expand_dims(forecasts, -2))
            weights_matrix = (numpy.expand_dims(weights, -1) *
                              numpy.expand_dims(weights, -2))
            with suppress_warnings('Mean of empty slice'):
                score += -0.5 * numpy.nanmean(weights_matrix * abs(forecasts_diff),
                                           axis=(-2, -1))
            return score
        elif observations.ndim == forecasts.ndim:
            # there is no 'realization' axis to sum over (this is a deterministic
            # forecast)
            return abs(observations - forecasts)


    crps = []
    for n in range(0,len(real_price)):
        crps.append(1/nsim*numpy.sum(abs(real_price['PRICE'][n] - df_predict.iloc[:, n]))) - 1/(2*nsim**2)*numpy.sum()
        print("")

