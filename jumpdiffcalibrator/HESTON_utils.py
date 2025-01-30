import pandas
import numpy
import datetime
import dateutil.relativedelta as relativedelta
import sys
sys.path.append(r'D:\TU_DORTMUND\ThesisCode')
import jumpdiffcalibrator as jdcal
import CRPS.CRPS as pscore
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from pandas.tseries.holiday import USFederalHolidayCalendar
cal = USFederalHolidayCalendar()

class HESTON_CNT():
    def __init__(self, contract = 'Jan-24', valdate = datetime.datetime(2023, 10, 1),
                 nsim = 10000,
                 r=0,
                 q=0,
                 n_mcmc_steps = 10000,
                 burn_in=5000,
                 estimation_length = 3, # month
                 plot = False, rate = 0.05,
                 fixed_param = None):
        price_df = pandas.read_csv(fr"D:\TU_DORTMUND\Thesis\Data\price\{contract}.csv")
        self.cnt = contract
        #self.expiry_date = datetime.datetime.strptime(contract, '%b-%y')
        price_df['Date'] = pandas.to_datetime(price_df['Date'])
        price_df = price_df[price_df['Date'].dt.weekday < 5]
        price_df = price_df[['Date', 'CLOSE']]
        price_df.columns = ['Date', 'Price']
        self.nsim = nsim
        self.r = r
        self.q = q
        self.n_mcmc_steps = n_mcmc_steps
        self.burn_in = burn_in
        self.estimation_length = estimation_length
        self.plot = plot
        self.rate = rate
        self.df = price_df.iloc[-252:, :]
        self.expiry_date = price_df['Date'].iloc[-1]
        self.valdate = valdate
        self.fixed_param = fixed_param
        self.gas_data = self.df[(self.df['Date'] >= self.valdate - relativedelta.relativedelta(months=self.estimation_length)) & (self.df['Date'] < self.valdate)]['Price'].to_numpy()
        self.HestonClibration()
        self.insample_simulation()
        # spot = pandas.read_csv(r'D:\TU_DORTMUND\Thesis\Data\price\Spot_Price.csv')
        # spot['Date'] = pandas.to_datetime(spot['Date'])
        # self.spot = spot
        self.get_step_till_expiry()
        self.get_step_till_end()
        #self.outsample_simulation()


    def HestonClibration(self):
        self.heston_cal = jdcal.HestonCalibrator(price_series=self.gas_data, cost_of_carry=self.r - self.q,
                                                 vol_prior_mean = 0.09, vol_prior_std = 0.1)
        self.heston_cal.calibrate(n_mcmc_steps=self.n_mcmc_steps, burn_in=self.burn_in, fixed_param=self.fixed_param)
        self.all_params = self.heston_cal.params_dict
        self.mu = self.all_params.get("mu_final")
        self.kappa = self.all_params.get("kappa_final")
        self.theta = self.all_params.get("theta_final")
        self.sigma = self.all_params.get("volvol_final")
        self.rho = self.all_params.get("rho_final")
        self._params_dict = {'mu_final': self.mu,
                             'kappa_final': self.kappa,
                             'theta_final': self.theta,
                             'volvol_final': self.sigma,
                             'rho_final': self.rho
                             }

    def insample_simulation(self):

        s0 = self.gas_data[0]
        simulated_paths, simulated_variances = self.heston_cal.get_paths(s0=s0, nsteps=len(self.gas_data)-1, nsim=self.nsim, risk_neutral=False,
                                                                    dt=1 / 252)
        self.price_simu_insample = pandas.DataFrame(simulated_paths).T
        self.vpast = pandas.DataFrame(simulated_variances).iloc[:,-1].mean()
        price_forecast = [x for x in self.price_simu_insample.iloc[-1] if x > 0 and x < numpy.inf]
        if len(price_forecast)>0:
            self.insample_crps, fcrps, acrps = pscore(self.price_simu_insample.iloc[-1], self.gas_data[:len(self.gas_data)][-1]).compute()
        else:
            self.insample_crps = numpy.nan
        print(f"CRPS In-Sample: {self.insample_crps}")
    def get_business_day(self, dd):
        cal = USFederalHolidayCalendar()
        while dd.isoweekday() > 5 or dd in cal.holidays():
            dd += datetime.timedelta(days=1)
        return dd

    def get_step_till_expiry(self):
        #self.first_busday_after_expiry = self.get_business_day(self.expiry_date)
        self.steps_needed = numpy.busday_count(self.valdate.date(), self.expiry_date.date())

    def get_step_till_end(self):
        self.steps_end = numpy.busday_count(self.valdate.date(), self.df['Date'].iloc[-1].date())

    def outsample_simulation(self, step):

        s0 = self.gas_data[-1]
        simulated_paths, simulated_variances = self.heston_cal.get_paths(s0=s0, nsteps=step, nsim=self.nsim,
                                                                        risk_neutral=True,
                                                                        v0 = self.vpast)
        price_simu_outsample = pandas.DataFrame(simulated_paths).T
        #real_gas = self.df.iloc[self.si + self.estimation_length: self.si + self.estimation_length + step + 1]['Price'].to_numpy()
        #self.terminal_price = self.spot[self.spot['Date'] == self.first_busday_after_expiry]['Close'].values[0]
        self.terminal_price = self.df['Price'].iloc[-1]
        price_forecast = [x for x in price_simu_outsample.iloc[-1] if x > 0 and x < numpy.inf]
        if len(price_forecast) > 0:
            self.outsample_crps, fcrps1, acrps1 = pscore(price_forecast, self.terminal_price).compute()
        else:
            self.outsample_crps = numpy.nan
        print(f"CRPS Out-Sample: {self.outsample_crps}")
        return price_simu_outsample, self.outsample_crps

    def call_option_price(self, step, moneyness=0.9):
        price_simu, _ = self.outsample_simulation(step)
        self.strike = self.gas_data[-1]*moneyness
        self.t2e = (self.expiry_date - self.valdate).days // 30
        payoff = numpy.mean([i if i > 0 else 0 for i in list(price_simu.iloc[-1,:] - self.strike)])
        discount = (1+self.rate)**(self.t2e/12)
        call_price = payoff/discount
        return call_price