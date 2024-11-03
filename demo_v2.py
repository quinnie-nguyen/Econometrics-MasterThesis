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
            bates_cal = jdcal.BatesCalibrator(price_series=stock_data, cost_of_carry=r - q)
            bates_cal.calibrate(n_mcmc_steps=n_mcmc_steps, burn_in=burn_in)
            all_params = bates_cal.params_dict
            mu = all_params.get("mu_final")
            kappa = all_params.get("kappa_final")
            theta = all_params.get("theta_final")
            sigma = all_params.get("volvol_final")
            rho = all_params.get("rho_final")
            mu_s = all_params.get("mu_s_final")
            sigma_s = numpy.sqrt(all_params.get("sigma_sq_s_final"))
            lambda_d = all_params.get("lambda_d_final")
        else:
            bates_cal = jdcal.BatesCalibrator(parameter_dict = all_params, price_series=stock_data, cost_of_carry=r - q)
        print(all_params)
        pandas.DataFrame(all_params, index = [0]).to_csv(f"D:\TU_DORTMUND\Thesis\Data\params\Bates\params_{date[i].astype('datetime64[s]').astype(datetime.datetime).strftime('%Y-%m-%d')}.csv", index = False)
        simulated_paths, simulated_variances = bates_cal.get_paths(s0=s0, nsteps=nsteps, nsim=nsim, risk_neutral=False)
        simulated_paths = pandas.DataFrame(simulated_paths).transpose()
        simulated_paths['PRICE'] = price[i - 1:i + 5]
        pandas.DataFrame(simulated_paths).to_csv(f"D:\TU_DORTMUND\Thesis\Data\simulation\price\Bates\price_{date[i].astype('datetime64[s]').astype(datetime.datetime).strftime('%Y-%m-%d')}.csv", index =False)
        pandas.DataFrame(simulated_variances).transpose().to_csv(rf"D:\TU_DORTMUND\Thesis\Data\simulation\vol\Bates\vol_{date[i].astype('datetime64[s]').astype(datetime.datetime).strftime('%Y-%m-%d')}.csv", index = False)




    # ================================== Heston calibration====================================
    #
    #
    # heston_cal = jdcal.HestonCalibrator(price_series=stock_data, cost_of_carry=r - q)
    #
    # start = datetime.datetime.now()
    # heston_cal.calibrate(n_mcmc_steps=n_mcmc_steps, burn_in=burn_in)
    # finish = datetime.datetime.now()
    # print(f"{(finish - start) / 60} minutes elapsed")
    #
    # # ----- get the calibrated parameters
    # all_params = heston_cal.params_dict
    # mu = all_params.get("mu_final")
    # kappa = all_params.get("kappa_final")
    # theta = all_params.get("theta_final")
    # sigma = all_params.get("volvol_final")
    # rho = all_params.get("rho_final")
    #
    # # ----- get stock and variance trajectories
    # simulated_paths, simulated_variances = heston_cal.get_paths(s0=s0, nsteps=nsteps, nsim=nsim, risk_neutral=False)