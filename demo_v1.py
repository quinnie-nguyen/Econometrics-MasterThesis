import datetime
import numpy
import pandas
import pandas
import matplotlib.pyplot as plt
#import seaborn as sns
import warnings
import dateutil.relativedelta as reldel
#sns.set(palette='viridis')
warnings.filterwarnings("ignore")
import sys
sys.path.append(r'C:\Users\UI620224\PycharmProjects\OPTION_TRADING_STRATEGIES\optiontradingstrategies')
import VectorAutoReg.jumpdiffcalibrator as jdcal
import util.pb_utilities as pb

month_to_symbol_dict = {'Jan': 'F',
                        'Feb': 'G',
                        'Mar': 'H',
                        'Apr': 'J',
                        'May': 'K',
                        'Jun': 'M',
                        'Jul': 'N',
                        'Aug': 'Q',
                        'Sep': 'U',
                        'Oct': 'V',
                        'Nov': 'X',
                        'Dec': 'Z'}
def get_gas_price(yr = [18, 19, 20, 21, 22, 23, 24]):
    df_list = dict()
    df_list['cnt_name'] = []
    df_list['data'] = []
    start_date =  datetime.datetime(2017, 11, 1)
    for y in yr:
        for m in list(month_to_symbol_dict.keys()):
            tckr = f"{month_to_symbol_dict[m]}_{y}_1"
            print(tckr)
            df_temp = pb.get_price_timeseries(pb.cnt_from_tckr(tckr), start_date, pb.get_expiry_date_non_standard(pb.cnt_from_tckr(tckr)))
            df_list['cnt_name'] = pb.cnt_from_tckr(tckr).name
            df_list['data'].append(df_temp)
            start_date = pb.get_expiry_date_non_standard(pb.cnt_from_tckr(tckr))
    df_total = pandas.concat([df for df in df_list['data']], axis=0).drop_duplicates(keep='first')
    return df_total

def get_gas_atm(yr = [18, 19, 20, 21, 22, 23, 24]):
    df_list = dict()
    df_list['cnt_name'] = []
    df_list['data'] = []
    start_date =  datetime.datetime(2017, 11, 1)
    for y in yr:
        for m in list(month_to_symbol_dict.keys()):
            tckr = f"{month_to_symbol_dict[m]}_{y}_1"
            print(tckr)
            df_temp = pb.get_atm_vol_timeseries(pb.cnt_from_tckr(tckr), start_date, pb.get_expiry_date_non_standard(pb.cnt_from_tckr(tckr)))
            df_list['cnt_name'] = pb.cnt_from_tckr(tckr).name
            df_list['data'].append(df_temp)
            start_date = pb.get_expiry_date_non_standard(pb.cnt_from_tckr(tckr))
    df_total = pandas.concat([df for df in df_list['data']], axis=0).drop_duplicates(keep='first')
    return df_total

if __name__ == '__main__':
    vol = get_gas_atm()
    vol.to_csv(r'C:\temp\gas_vol.csv')

    df = get_gas_price()
    df.to_csv(r"C:\temp\gas_price.csv")


    
    # ----- set parameters
    #s0 = 100
    nsteps = 5
    nsim = 10000
    r = 0 #0.05
    q = 0 #0.02
    # ----- calibrate parameters
    n_mcmc_steps = 10000
    burn_in = 5000

    month_to_symbol_dict = {'Jan': 'F',
                            'Feb': 'G',
                            'Mar': 'H',
                            'Apr': 'J',
                            'May': 'K',
                            'Jun': 'M',
                            'Jul': 'N',
                            'Aug': 'Q',
                            'Sep': 'U',
                            'Oct': 'V',
                            'Nov': 'X',
                            'Dec': 'Z'}

    contract_key_dict = {1: ('GAS', 'TTF', None), 2: ('GAS', 'HH', None), 
                         4: ('OIL', 'WTI', None), 5: ('OIL', 'BRENT', None)}
    
    yr = 24
    for com in contract_key_dict.keys():
        for m in month_to_symbol_dict.values():
            tckr = m + '_' + str(yr) + '_' + str(com)
            print(tckr)
            start_date = pb.get_expiry_date_non_standard(pb.cnt_from_tckr(tckr)) - reldel.relativedelta(months=12)
            print(start_date)
            # ----- load data
            df = pb.get_price_timeseries(pb.cnt_from_tckr(tckr), start_date)
            df.index = pandas.to_datetime(df.index)
            df = df[df.index.weekday < 5]
            df.reset_index(inplace=True, drop=False)
            df.columns = ['Date', 'Price']
            #df = pandas.read_csv(r'D:\TU_DORTMUND\ThesisCode\aapl_data.csv')
            #df['Date'] = pandas.to_datetime(df['Date'], dayfirst=True)
            #df.sort_values(['Date'], ascending=True, inplace=True)
            date = df['Date'].values
            price = df["Price"].values
            all_params = None
            for i in range(120, len(price)-6):
                print(i)
                print(date[i])
                print('###################################')
                stock_data = price[i+1-120:i+1]
                s0 = stock_data[-1]
        
                if True:
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
                param_df = pandas.DataFrame(all_params, index = [0])
                param_df['ValDate'] = date[i].astype('datetime64[s]').astype(datetime.datetime).strftime('%Y-%m-%d')
                param_df['TCKR'] = tckr
                try:
                    param_df_prev = pandas.read_csv(rf"C:\temp\Thesis\Heston\params\{tckr}_params.csv")
                    param_df_temp = pandas.concat([param_df_prev, param_df], axis = 0)
                    param_df_temp.to_csv(rf"C:\temp\Thesis\Heston\params\{tckr}_params.csv", index=False)
                except:
                    param_df.to_csv(rf"C:\temp\Thesis\Heston\params\{tckr}_params.csv", index = False)
                simulated_paths, simulated_variances = heston_cal.get_paths(s0=s0, nsteps=nsteps, nsim=nsim, risk_neutral=False)
                simulated_paths = pandas.DataFrame(simulated_paths).transpose()
                simulated_paths['ValDate'] = date[i].astype('datetime64[s]').astype(datetime.datetime).strftime('%Y-%m-%d')
                simulated_paths['TradingDate'] = date[i: i + nsteps + 1]
                simulated_paths['PRICE'] = price[i:i + nsteps + 1]
                try:
                    simulated_paths_prev = pandas.read_csv(rf"C:\temp\Thesis\Heston\price\{tckr}_price.csv")
                    simulated_paths_temp = pandas.concat([simulated_paths_prev, simulated_paths], axis=0)
                    simulated_paths_temp.to_csv(rf"C:\temp\Thesis\Heston\params\{tckr}_price_simulations.csv", index=False)
                except:
                    simulated_paths.to_csv(rf"C:\temp\Thesis\Heston\price\{tckr}_price_simulations.csv", index =False)
                #pandas.DataFrame(simulated_variances).transpose().to_csv(rf"C:\temp\Thesis\Heston\vols\{tckr}_vols_{date[i].astype('datetime64[s]').astype(datetime.datetime).strftime('%Y-%m-%d')}.csv", index = False)

### date is the date that simulation starts + 1
    # peng
    # 
    # # ================================== Heston CRPS ====================================
    # 
    # import glob
    # import os
    # 
    # # Folder path containing CSV files
    # folder_path = 'D:\TU_DORTMUND\Thesis\Data\simulation\price\Heston'
    # 
    # # Get a list of all CSV files in the folder
    # csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    # price = dict()
    # price['DATE'] = []
    # price['PRICE'] = []
    # df_predict = pandas.DataFrame()
    # for file in csv_files:
    #     df = pandas.read_csv(file)
    #     df_log = numpy.log(df).diff()
    #     col_name =file[-14:-4]
    #     df_temp = pandas.DataFrame(df_log.iloc[-1][:-1])
    #     df_temp.columns = [col_name]
    #     df_predict = pandas.concat([df_predict, df_temp], axis =1)
    #     price['DATE'].append(col_name)
    #     price['PRICE'].append(df.iloc[-1][-1])
    # 
    # pandas.DataFrame(price).plot()
    # 
    # 
    # ########### CRPS ##############
    # 
    # import glob
    # import os
    # 
    # # Folder path containing CSV files
    # folder_path = 'D:\TU_DORTMUND\Thesis\Data\simulation\price\Heston'
    # 
    # # Get a list of all CSV files in the folder
    # csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    # price = dict()
    # price['DATE'] = []
    # price['PRICE'] = []
    # df_predict = pandas.DataFrame()
    # for file in csv_files:
    #     df = pandas.read_csv(file)
    #     #df_log = numpy.log(df).diff()
    #     col_name =file[-14:-4]
    #     #df_temp = pandas.DataFrame(df_log.iloc[-1][:-1])
    #     df_temp = pandas.DataFrame(df.iloc[-1][:-1])
    #     df_temp.columns = [col_name]
    #     df_predict = pandas.concat([df_predict, df_temp], axis =1)
    #     price['DATE'].append(col_name)
    #     price['PRICE'].append(df.iloc[-1][-1])
    # 
    # real_price = pandas.DataFrame(price)
    # 
    # 
    # def suppress_warnings(msg=None):
    #     with warnings.catch_warnings():
    #         warnings.filterwarnings('ignore', msg)
    #         yield
    # def _crps_ensemble_vectorized(observations, forecasts, weights=1):
    #     """
    #     An alternative but simpler implementation of CRPS for testing purposes
    # 
    #     This implementation is based on the identity:
    # 
    #     .. math::
    #         CRPS(F, x) = E_F|X - x| - 1/2 * E_F|X - X'|
    # 
    #     where X and X' denote independent random variables drawn from the forecast
    #     distribution F, and E_F denotes the expectation value under F.
    # 
    #     Hence it has runtime O(n^2) instead of O(n log(n)) where n is the number of
    #     ensemble members.
    # 
    #     Reference
    #     ---------
    #     Tilmann Gneiting and Adrian E. Raftery. Strictly proper scoring rules,
    #         prediction, and estimation, 2005. University of Washington Department of
    #         Statistics Technical Report no. 463R.
    #         https://www.stat.washington.edu/research/reports/2004/tr463R.pdf
    #     """
    #     observations = numpy.asarray(observations)
    #     forecasts = numpy.asarray(forecasts)
    #     weights = numpy.asarray(weights)
    #     if weights.ndim > 0:
    #         weights = numpy.where(~numpy.isnan(forecasts), weights, numpy.nan)
    #         weights = weights / numpy.nanmean(weights, axis=-1, keepdims=True)
    # 
    #     if observations.ndim == forecasts.ndim - 1:
    #         # sum over the last axis
    #         assert observations.shape == forecasts.shape[:-1]
    #         observations = observations[..., numpy.newaxis]
    #         with suppress_warnings('Mean of empty slice'):
    #             print('step1')
    #             score = numpy.nanmean(weights * abs(forecasts - observations), -1)
    #         # insert new axes along last and second to last forecast dimensions so
    #         # forecasts_diff expands with the array broadcasting
    #         forecasts_diff = (numpy.expand_dims(forecasts, -1) -
    #                           numpy.expand_dims(forecasts, -2))
    #         weights_matrix = (numpy.expand_dims(weights, -1) *
    #                           numpy.expand_dims(weights, -2))
    #         with suppress_warnings('Mean of empty slice'):
    #             score += -0.5 * numpy.nanmean(weights_matrix * abs(forecasts_diff),
    #                                        axis=(-2, -1))
    #         return score
    #     elif observations.ndim == forecasts.ndim:
    #         # there is no 'realization' axis to sum over (this is a deterministic
    #         # forecast)
    #         return abs(observations - forecasts)
    # 
    # 
    # crps = []
    # for n in range(0,len(real_price)):
    #     crps.append(1/nsim*numpy.sum(abs(real_price['PRICE'][n] - df_predict.iloc[:, n]))) - 1/(2*nsim**2)*numpy.sum()
    #     print("")
    # 
