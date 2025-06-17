import datetime
import dateutil.relativedelta as relativedelta
import numpy
import scipy.special.cython_special
#from networkx import volume
from scipy.optimize import minimize
import pandas
import CRPS.CRPS as pscore
from pandas.tseries.holiday import USFederalHolidayCalendar
cal = USFederalHolidayCalendar()
from sstudentt import SST

class GARCH():

    def __init__(self, price_series, p: int = 1, q: int = 1, o: int = 1,
                 init_sigma: float = [0.1], init_r: float = [0.01], dist: str = 'gaussian'):
        self.gas_data = price_series
        self.returns = numpy.diff(numpy.log(price_series))# "Y" is "returns" here
        self.mean_init = numpy.average(self.returns)
        self.var_init = numpy.std(self.returns)**2
        self.s0 = price_series[0]
        self.T = len(self.returns)
        self.p = p
        self.q = q
        self.o = o
        self.init_sigma = init_sigma
        #self.init_r = init_r
        self.dist = dist
        self.garch_mle()
        #self.get_paths()

    def loss(self, params):
        mu = params[0]
        omega = params[1]
        alpha = params[2]
        beta = params[3]

        #calculating long-run volatility

        long_run = (omega/(1-alpha-beta))**0.5

        #calculating realized and conditional volatilities

        resi = self.returns - mu
        realised = abs(resi)

        conditional = numpy.zeros(len(self.returns))
        conditional[0] = long_run

        for t in range(1, len(self.returns)):

            conditional[t] = (omega+alpha*resi[t-1]**2+beta*conditional[t-1]**2)**0.5

        likelihood = 1/((2*numpy.pi)**0.5*conditional)*numpy.exp(-realised**2/(2*conditional**2))

        log_likelihood = numpy.sum(numpy.log(likelihood))

        return -log_likelihood

    def garch_mle(self):

        self.result = minimize(self.loss, [self.mean_init, self.var_init, 0, 0], method="SLSQP",options={'maxiter' : 10000})
        self._params_dict = {'mu': self.result['x'][0],
                             'omega': self.result['x'][1],
                             'alpha': self.result['x'][2],
                             'beta': self.result['x'][3]
                             }
        return self.result
    def get_paths(self, s0=18.56111935, nsteps=2000, nsim=100, v0 = None):
        assert self._params_dict is not None, "Parameters have not been calibrated yet"
        mu = self._params_dict.get("mu")
        omega = self._params_dict.get("omega")
        alpha = self._params_dict.get("alpha")
        beta = self._params_dict.get("beta")
        long_run = (omega/(1-alpha-beta))**0.5
        if v0 is None:
            v0 = long_run
        #dt = 1 / nsteps
        resi = self.returns - mu
        simulated_r = numpy.zeros([nsim, nsteps + 1])
        simulated_r[:, 0] = resi[0] + mu
        simulated_volas = numpy.zeros([nsim, nsteps + 1])
        simulated_volas[:, 0] = v0

        # --- get randomness (correlated for each t=1,...,T, as corr(W_S, W_V) = rho)
        Z_V = numpy.random.normal(size=[nsim, nsteps + 1])

        # ----- generate paths
        for t in range(1, nsteps+1):
            # --- get the time-varying volatility component
            simulated_volas[:, t] = (omega+alpha*simulated_r[:, t-1]**2+beta*simulated_volas[:, t-1]**2)**0.5
            simulated_r[:, t] = mu +Z_V[:, t] * simulated_volas[:, t]

            # --- get the total price dynamics
        simulated_paths = numpy.zeros([nsim, nsteps + 1])
        simulated_paths[:, 0] = s0
        for i in range(nsteps):
            simulated_paths[:, i + 1] = simulated_paths[:, i] * numpy.exp(simulated_r[:, i+1])
        return simulated_paths, simulated_volas


class EGARCH():

    def __init__(self, nsim: int = 10000,
                 estimation_length: int = 60,
                 si: int = 0,
                 p: int = 1,
                 q: int = 1,
                 o: int = 1,
                 long_run: float = 0.1, dist: str = 'gaussian',
                 plot: str = False, vix: str = False):
        price_df = pandas.read_csv(r"C:\temp\gas_price.csv")
        price_df['Unnamed: 0'] = pandas.to_datetime(price_df['Unnamed: 0'])
        price_df = price_df[price_df['Unnamed: 0'].dt.weekday < 5]
        price_df.columns = ['Date', 'Price']
        vol_df = pandas.read_csv(r"C:\temp\gas_vol.csv")
        vol_df['Unnamed: 0'] = pandas.to_datetime(vol_df['Unnamed: 0'])
        vol_df = vol_df[vol_df['Unnamed: 0'].dt.weekday < 5]
        vol_df.columns = ['Date', 'Vol']
        df = price_df.merge(vol_df, how='left', on='Date')
        self.df = df
        self.estimation_length = estimation_length
        self.si = si
        self.nsim = nsim
        self.gas_data = self.df.iloc[self.si:self.si + self.estimation_length + 1]['Price'].to_numpy()
        self.returns = numpy.diff(numpy.log(self.gas_data))# "Y" is "returns" here
        self.mean_init = numpy.average(self.returns)
        self.var_init = numpy.std(self.returns)**2
        self.vix_series = self.df.iloc[self.si:self.si + self.estimation_length + 1]['Vol'].to_numpy()
        self.vixTF = vix
        if self.vixTF is True:
            self.vix_series = self.vix_series/100
        #self.s0 = price_series[0]
        self.T = len(self.returns)
        self.p = p
        self.q = q
        self.o = o
        self.long_run = long_run
        self.plot = plot
        #self.init_r = init_r
        self.dist = dist
        self.egarch_mle()
        self.insample_simulation()

    def loss(self, params):
        mu = params[0]
        omega = params[1]
        alpha = params[2]
        gamma = params[3]
        beta = params[4]

        #calculating long-run volatility

        long_run = numpy.exp(omega/(1-beta))**0.5

        #calculating realized and conditional volatilities

        resi = self.returns - mu
        indic = [1 if resi[i] > 0 else -1 for i in range(self.T)]
        realised = abs(resi)

        conditional = numpy.zeros(len(self.returns))
        conditional[0] = long_run

        for t in range(1, len(self.returns)):

            # conditional[t] = numpy.sqrt(numpy.exp(omega + alpha*(abs(resi[t-1]/conditional[t-1]) - (2/numpy.pi)**0.5) +
            #                                       gamma*(resi[t-1]/conditional[t-1]) + beta*numpy.log(conditional[t-1]**2)))

            conditional[t] = numpy.sqrt(numpy.exp(omega + alpha*indic[t-1]*(resi[t-1]/conditional[t-1]) +
                                                  gamma*(resi[t-1]/conditional[t-1]) + beta*numpy.log(conditional[t-1]**2)))

        likelihood = 1/((2*numpy.pi)**0.5*conditional)*numpy.exp(-realised**2/(2*conditional**2))

        log_likelihood = numpy.sum(numpy.log(likelihood))

        return -log_likelihood

    def loss_vix(self, params):
        mu = params[0]
        omega = params[1]
        alpha = params[2]
        gamma = params[3]
        beta = params[4]
        theta = params[5]

        #calculating long-run volatility

        long_run = numpy.exp((omega)/(1-beta))**0.5#numpy.exp((omega+theta*numpy.log(self.long_run**2))/(1-beta))**0.5

        #calculating realized and conditional volatilities

        resi = self.returns - mu
        indic = [1 if resi[i] > 0 else -1 for i in range(self.T)]
        realised = abs(resi)

        conditional = numpy.zeros(len(self.returns))
        conditional[0] = long_run

        for t in range(1, len(self.returns)):

            # conditional[t] = numpy.sqrt(numpy.exp(omega + alpha*(abs(resi[t-1]/conditional[t-1]) - (2/numpy.pi)**0.5) +
            #                                       gamma*(resi[t-1]/conditional[t-1]) + beta*numpy.log(conditional[t-1]**2)))

            conditional[t] = numpy.sqrt(numpy.exp(omega + alpha*indic[t-1]*(resi[t-1]/conditional[t-1]) +
                                                  gamma*(resi[t-1]/conditional[t-1]) + beta*numpy.log(conditional[t-1]**2) + theta*numpy.log(self.vix_series[t-1]**2)))
        

        likelihood = 1/((2*numpy.pi)**0.5*conditional)*numpy.exp(-realised**2/(2*conditional**2))

        log_likelihood = numpy.sum(numpy.log(likelihood))
        
        print(log_likelihood)

        return -log_likelihood

    def egarch_mle(self):
        if self.vixTF is False:
            self.result = minimize(self.loss, [self.mean_init, self.var_init, 0, 0, 0], method="SLSQP",options={'maxiter' : 10000})
            self._params_dict = {'mu': self.result['x'][0],
                                 'omega': self.result['x'][1],
                                 'alpha': self.result['x'][2],
                                 'gamma': self.result['x'][3],
                                 'beta': self.result['x'][4]
                                 }
        else:
            self.result = minimize(self.loss_vix, [self.mean_init, self.var_init, 0, 0, 0, 0], method="SLSQP",options={'maxiter' : 10000})
            self._params_dict = {'mu': self.result['x'][0],
                                 'omega': self.result['x'][1],
                                 'alpha': self.result['x'][2],
                                 'gamma': self.result['x'][3],
                                 'beta': self.result['x'][4],
                                 'theta': self.result['x'][5]
                                 }
        
        return self.result
    
    def get_paths(self, s0=18.56111935, nsteps=2000, nsim=100, v0 = None):
        assert self._params_dict is not None, "Parameters have not been calibrated yet"
        mu = self._params_dict.get("mu")
        omega = self._params_dict.get("omega")
        alpha = self._params_dict.get("alpha")
        gamma = self._params_dict.get("gamma")
        beta = self._params_dict.get("beta")
        long_run = numpy.exp(omega/(1-beta))**0.5
        if v0 is None:
            v0 = long_run
        #dt = 1 / nsteps
        resi = self.returns - mu
        simulated_r = numpy.zeros([nsim, nsteps + 1])
        simulated_r[:, 0] = resi[0] + mu
        simulated_volas = numpy.zeros([nsim, nsteps + 1])
        simulated_volas[:, 0] = v0

        # --- get randomness (correlated for each t=1,...,T, as corr(W_S, W_V) = rho)
        numpy.random.seed(42)
        Z_V = numpy.random.normal(size=[nsim, nsteps + 1])
        indic = numpy.sign(Z_V)

        # ----- generate paths
        for t in range(1, nsteps+1):
            # --- get the time-varying volatility component
            #simulated_volas[:, t] = numpy.sqrt(numpy.exp(omega + alpha*(abs(simulated_r[:, t-1]/simulated_volas[:, t-1]) - (2/numpy.pi)**0.5) +
            #                                      gamma*(simulated_r[:, t-1]/simulated_volas[:, t-1]) + beta*numpy.log(simulated_volas[:, t-1]**2)))
            simulated_volas[:, t] = numpy.sqrt(numpy.exp(omega + alpha*indic[:, t-1] *simulated_r[:, t-1]/simulated_volas[:, t-1] +
                                                         gamma*(simulated_r[:, t-1]/simulated_volas[:, t-1]) + beta*numpy.log(simulated_volas[:, t-1]**2)))
            simulated_r[:, t] = mu +Z_V[:, t] * simulated_volas[:, t]

            # --- get the total price dynamics
        simulated_paths = numpy.zeros([nsim, nsteps + 1])
        simulated_paths[:, 0] = s0
        for i in range(nsteps):
            simulated_paths[:, i + 1] = simulated_paths[:, i] * numpy.exp(simulated_r[:, i+1])
        return simulated_paths, simulated_volas

    def insample_simulation(self):

        s0 = self.gas_data[0]
        simulated_paths, simulated_variances = self.get_paths(s0=s0, nsteps=len(self.gas_data)-1, nsim=self.nsim, v0=None)
        self.price_simu_insample = pandas.DataFrame(simulated_paths).T
        self.vpast = pandas.DataFrame(simulated_variances).iloc[:,-1].mean()
        self.insample_crps, fcrps, acrps = pscore(self.price_simu_insample.iloc[-1], self.gas_data[:len(self.gas_data)][-1]).compute()
        print(f"CRPS In-Sample: {self.insample_crps}")

        if self.plot==True:
            pandas.concat([self.price_simu_insample.iloc[:, :10], pandas.DataFrame(self.gas_data[:len(self.gas_data)]).rename(columns={0: "Price"})],
                          axis=1).plot(color = ['black', 'black', 'black', 'black', 'black', 'black', 'black', 'black', 'black', 'black',
                                                 'red'])

    def outsample_simulation(self, step):

        s0 = self.gas_data[-1]
        simulated_paths, simulated_variances = self.get_paths(s0=s0, nsteps=step, nsim=self.nsim, v0=self.vpast)
        price_simu_outsample = pandas.DataFrame(simulated_paths).T
        real_gas = self.df.iloc[self.si + self.estimation_length: self.si + self.estimation_length + step + 1]['Price'].to_numpy()
        outsample_crps, fcrps1, acrps1 = pscore(price_simu_outsample.iloc[-1], real_gas[-1]).compute()
        print(f"CRPS Out-Sample: {outsample_crps}")
        if self.plot == True:
            pandas.concat([price_simu_outsample.iloc[:, :10], pandas.DataFrame(real_gas).rename(columns={0: "Price"})],
                          axis=1).plot(color= ['black', 'black', 'black', 'black', 'black', 'black', 'black', 'black', 'black', 'black',
                                                 'red'])
        return price_simu_outsample, real_gas, outsample_crps


    
    def get_paths_vixincluded(self, s0=18.56111935, nsteps=2000, nsim=100, v0 = None):
        assert self._params_dict is not None, "Parameters have not been calibrated yet"
        mu = self._params_dict.get("mu")
        omega = self._params_dict.get("omega")
        alpha = self._params_dict.get("alpha")
        gamma = self._params_dict.get("gamma")
        beta = self._params_dict.get("beta")
        theta = self._params_dict.get("theta")
        long_run = numpy.exp((omega)/(1-beta))**0.5#numpy.exp((omega+theta*numpy.log(self.long_run**2))/(1-beta))**0.5
        if v0 is None:
            v0 = long_run
        #dt = 1 / nsteps
        resi = self.returns - mu
        simulated_r = numpy.zeros([nsim, nsteps + 1])
        simulated_r[:, 0] = resi[0] + mu
        simulated_volas = numpy.zeros([nsim, nsteps + 1])
        simulated_volas[:, 0] = v0

        # --- get randomness (correlated for each t=1,...,T, as corr(W_S, W_V) = rho)
        numpy.random.seed(42)
        Z_V = numpy.random.normal(size=[nsim, nsteps + 1])
        indic = numpy.sign(Z_V)

        # ----- generate paths
        for t in range(1, nsteps+1):
            # --- get the time-varying volatility component
            #simulated_volas[:, t] = numpy.sqrt(numpy.exp(omega + alpha*(abs(simulated_r[:, t-1]/simulated_volas[:, t-1]) - (2/numpy.pi)**0.5) +
            #                                      gamma*(simulated_r[:, t-1]/simulated_volas[:, t-1]) + beta*numpy.log(simulated_volas[:, t-1]**2)))
            simulated_volas[:, t] = numpy.sqrt(numpy.exp(omega + alpha*indic[:, t-1] *simulated_r[:, t-1]/simulated_volas[:, t-1] +
                                                         gamma*(simulated_r[:, t-1]/simulated_volas[:, t-1]) + beta*numpy.log(simulated_volas[:, t-1]**2) +
                                                         theta*numpy.log(self.vix_series[t-1]**2)))
            simulated_volas[:, t][simulated_volas[:, t] == numpy.inf] = 0
            simulated_r[:, t] = mu +Z_V[:, t] * simulated_volas[:, t]

            # --- get the total price dynamics
        simulated_paths = numpy.zeros([nsim, nsteps + 1])
        simulated_paths[:, 0] = s0
        for i in range(nsteps):
            simulated_paths[:, i + 1] = simulated_paths[:, i] * numpy.exp(simulated_r[:, i+1])


        return simulated_paths, simulated_volas


class EGARCH_CNT():

    def __init__(self, contract = 'Jan-24', nsim: int = 10000,
                 estimation_length: int = 3, #month
                 valdate = datetime.datetime(2023, 10, 1),
                 p: int = 1,
                 q: int = 1,
                 o: int = 1,
                 long_run: float = 0.1, dist: str = 'gaussian',
                 plot: str = False, rate = 0.05):
        price_df = pandas.read_csv(fr"D:\TU_DORTMUND\Thesis\Data\price\{contract}.csv")
        self.cnt = contract
        price_df['Date'] = pandas.to_datetime(price_df['Date'])
        price_df = price_df[price_df['Date'].dt.weekday < 5]
        price_df = price_df[['Date', 'CLOSE']]
        price_df.columns = ['Date', 'Price']
        self.expiry_date = price_df['Date'][-1]#datetime.datetime.strptime(contract, '%b-%y')
        #vol_df = pandas.read_csv(r"C:\temp\gas_vol.csv")
        #vol_df['Unnamed: 0'] = pandas.to_datetime(vol_df['Unnamed: 0'])
        #vol_df = vol_df[vol_df['Unnamed: 0'].dt.weekday < 5]
        #vol_df.columns = ['Date', 'Vol']
        #df = price_df.merge(vol_df, how='left', on='Date')
        self.df = price_df.iloc[-252:,:]
        self.estimation_length = estimation_length
        self.valdate = valdate
        self.nsim = nsim
        self.gas_data = self.df[(self.df['Date'] >= self.valdate - relativedelta.relativedelta(months=self.estimation_length)) & (self.df['Date'] < self.valdate)]['Price'].to_numpy()
        self.returns = numpy.diff(numpy.log(self.gas_data))  # "Y" is "returns" here
        self.mean_init = numpy.average(self.returns)
        self.var_init = numpy.std(self.returns) ** 2
        #self.vix_series = self.df.iloc[self.si:self.si + self.estimation_length + 1]['Vol'].to_numpy()
        #self.vixTF = vix
        #if self.vixTF is True:
        #    self.vix_series = self.vix_series / 100
        # self.s0 = price_series[0]
        self.T = len(self.returns)
        self.p = p
        self.q = q
        self.o = o
        self.long_run = long_run
        self.plot = plot
        # self.init_r = init_r
        self.dist = dist
        self.rate = rate
        self.egarch_mle()
        self.insample_simulation()

        # spot = pandas.read_csv(r'D:\TU_DORTMUND\Thesis\Data\price\Spot_Price.csv')
        # spot['Date'] = pandas.to_datetime(spot['Date'])
        # self.spot = spot
        # self.get_step_till_expiry()
        # self.get_step_till_end()

    def loss(self, params):
        mu = params[0]
        omega = params[1]
        alpha = params[2]
        gamma = params[3]
        beta = params[4]

        # calculating long-run volatility

        long_run = numpy.exp(omega / (1 - beta)) ** 0.5

        # calculating realized and conditional volatilities

        resi = self.returns - mu
        indic = [1 if resi[i] > 0 else -1 for i in range(self.T)]
        realised = abs(resi)

        conditional = numpy.zeros(len(self.returns))
        conditional[0] = long_run

        for t in range(1, len(self.returns)):
            # conditional[t] = numpy.sqrt(numpy.exp(omega + alpha*(abs(resi[t-1]/conditional[t-1]) - (2/numpy.pi)**0.5) +
            #                                       gamma*(resi[t-1]/conditional[t-1]) + beta*numpy.log(conditional[t-1]**2)))

            conditional[t] = numpy.sqrt(numpy.exp(omega + alpha * indic[t - 1] * (resi[t - 1] / conditional[t - 1]) +
                                                  gamma * (resi[t - 1] / conditional[t - 1]) + beta * numpy.log(
                conditional[t - 1] ** 2)))

        likelihood = 1 / ((2 * numpy.pi) ** 0.5 * conditional) * numpy.exp(-realised ** 2 / (2 * conditional ** 2))

        log_likelihood = numpy.sum(numpy.log(likelihood))

        return -log_likelihood

    def loss_vix(self, params):
        mu = params[0]
        omega = params[1]
        alpha = params[2]
        gamma = params[3]
        beta = params[4]
        theta = params[5]

        # calculating long-run volatility

        long_run = numpy.exp(
            (omega) / (1 - beta)) ** 0.5  # numpy.exp((omega+theta*numpy.log(self.long_run**2))/(1-beta))**0.5

        # calculating realized and conditional volatilities

        resi = self.returns - mu
        indic = [1 if resi[i] > 0 else -1 for i in range(self.T)]
        realised = abs(resi)

        conditional = numpy.zeros(len(self.returns))
        conditional[0] = long_run

        for t in range(1, len(self.returns)):
            # conditional[t] = numpy.sqrt(numpy.exp(omega + alpha*(abs(resi[t-1]/conditional[t-1]) - (2/numpy.pi)**0.5) +
            #                                       gamma*(resi[t-1]/conditional[t-1]) + beta*numpy.log(conditional[t-1]**2)))

            conditional[t] = numpy.sqrt(numpy.exp(omega + alpha * indic[t - 1] * (resi[t - 1] / conditional[t - 1]) +
                                                  gamma * (resi[t - 1] / conditional[t - 1]) + beta * numpy.log(
                conditional[t - 1] ** 2) + theta * numpy.log(self.vix_series[t - 1] ** 2)))

        likelihood = 1 / ((2 * numpy.pi) ** 0.5 * conditional) * numpy.exp(-realised ** 2 / (2 * conditional ** 2))

        log_likelihood = numpy.sum(numpy.log(likelihood))

        print(log_likelihood)

        return -log_likelihood

    def egarch_mle(self):

        self.result = minimize(self.loss, [self.mean_init, self.var_init, 0, 0, 0], method="SLSQP",
                               options={'maxiter': 10000})
        self._params_dict = {'mu': self.result['x'][0],
                             'omega': self.result['x'][1],
                             'alpha': self.result['x'][2],
                             'gamma': self.result['x'][3],
                             'beta': self.result['x'][4]
                             }
        return self.result

    def get_paths(self, s0=18.56111935, nsteps=2000, nsim=100, v0=None):
        assert self._params_dict is not None, "Parameters have not been calibrated yet"
        mu = self._params_dict.get("mu")
        omega = self._params_dict.get("omega")
        alpha = self._params_dict.get("alpha")
        gamma = self._params_dict.get("gamma")
        beta = self._params_dict.get("beta")
        long_run = numpy.exp(omega / (1 - beta)) ** 0.5
        if v0 is None:
            v0 = long_run
        # dt = 1 / nsteps
        resi = self.returns - mu
        simulated_r = numpy.zeros([nsim, nsteps + 1])
        simulated_r[:, 0] = resi[0] + mu
        simulated_volas = numpy.zeros([nsim, nsteps + 1])
        simulated_volas[:, 0] = v0

        # --- get randomness (correlated for each t=1,...,T, as corr(W_S, W_V) = rho)
        numpy.random.seed(42)
        Z_V = numpy.random.normal(size=[nsim, nsteps + 1])
        indic = numpy.sign(Z_V)

        # ----- generate paths
        for t in range(1, nsteps + 1):
            # --- get the time-varying volatility component
            # simulated_volas[:, t] = numpy.sqrt(numpy.exp(omega + alpha*(abs(simulated_r[:, t-1]/simulated_volas[:, t-1]) - (2/numpy.pi)**0.5) +
            #                                      gamma*(simulated_r[:, t-1]/simulated_volas[:, t-1]) + beta*numpy.log(simulated_volas[:, t-1]**2)))
            simulated_volas[:, t] = numpy.sqrt(
                numpy.exp(omega + alpha * indic[:, t - 1] * simulated_r[:, t - 1] / simulated_volas[:, t - 1] +
                          gamma * (simulated_r[:, t - 1] / simulated_volas[:, t - 1]) + beta * numpy.log(
                    simulated_volas[:, t - 1] ** 2)))
            simulated_r[:, t] = mu + Z_V[:, t] * simulated_volas[:, t]

            # --- get the total price dynamics
        simulated_paths = numpy.zeros([nsim, nsteps + 1])
        simulated_paths[:, 0] = s0
        for i in range(nsteps):
            simulated_paths[:, i + 1] = simulated_paths[:, i] * numpy.exp(simulated_r[:, i + 1])
        return simulated_paths, simulated_volas

    def insample_simulation(self):

        s0 = self.gas_data[0]
        simulated_paths, simulated_variances = self.get_paths(s0=s0, nsteps=len(self.gas_data) - 1, nsim=self.nsim,
                                                              v0=None)
        self.price_simu_insample = pandas.DataFrame(simulated_paths).T
        self.vpast = pandas.DataFrame(simulated_variances).iloc[:, -1].mean()
        self.insample_crps, fcrps, acrps = pscore(self.price_simu_insample.iloc[-1],
                                                  self.gas_data[:len(self.gas_data)][-1]).compute()
        print(f"CRPS In-Sample: {self.insample_crps}")

        if self.plot == True:
            pandas.concat([self.price_simu_insample.iloc[:, :10],
                           pandas.DataFrame(self.gas_data[:len(self.gas_data)]).rename(columns={0: "Price"})],
                          axis=1).plot(
                color=['black', 'black', 'black', 'black', 'black', 'black', 'black', 'black', 'black', 'black',
                       'red'])

    def get_business_day(self, dd):
        cal = USFederalHolidayCalendar()
        while dd.isoweekday() > 5 or dd in cal.holidays():
            dd += datetime.timedelta(days=1)
        return dd

    def get_step_till_expiry(self):
        self.first_busday_after_expiry = self.get_business_day(self.expiry_date)
        self.steps_needed = numpy.busday_count(self.valdate.date(), self.first_busday_after_expiry.date())

    def get_step_till_end(self):
        self.steps_end = numpy.busday_count(self.valdate.date(), self.df['Date'].iloc[-1].date())

    def outsample_simulation(self, step):

        s0 = self.gas_data[-1]
        simulated_paths, simulated_variances = self.get_paths(s0=s0, nsteps=step, nsim=self.nsim, v0=self.vpast)
        price_simu_outsample = pandas.DataFrame(simulated_paths).T
        self.terminal_price = self.df['Price'][-1]#self.spot[self.spot['Date'] == self.first_busday_after_expiry]['Close'].values[0]
        #terminal_price = self.df['Price'].iloc[-1]
        self.outsample_crps, fcrps1, acrps1 = pscore(price_simu_outsample.iloc[-1], self.terminal_price).compute()
        print(f"CRPS Out-Sample: {self.outsample_crps}")
        return price_simu_outsample, self.outsample_crps

    def call_option_price(self, step, moneyness=0.9):
        price_simu, _ = self.outsample_simulation(step)
        self.strike = self.gas_data[-1]*moneyness
        self.t2e = (self.first_busday_after_expiry - self.valdate).days // 30
        payoff = numpy.mean([i if i > 0 else 0 for i in list(price_simu.iloc[-1,:] - self.strike)])
        discount = (1+self.rate)**(self.t2e/12)
        call_price = payoff/discount
        return call_price


class EGARCH_SST():
    ### class for fitting EGARCH with skewed-t assumption for residual distributed.
    ### class will be used for CNT data
    def __init__(self, contract = 'Jan-24', nsim: int = 10000,
                 estimation_length: int = 3, #month
                 valdate = datetime.datetime(2023, 10, 1),
                 p: int = 1,
                 q: int = 1,
                 o: int = 1,
                 long_run: float = 0.1, dist: str = 'gaussian',
                 plot: str = False, rate = 0.05,
                 fixed_result = None):
        price_df = pandas.read_csv(fr"D:\TU_DORTMUND\Thesis\Data\price\{contract}.csv")
        self.cnt = contract
        #self.expiry_date = datetime.datetime.strptime(contract, '%b-%y')
        price_df['Date'] = pandas.to_datetime(price_df['Date'])
        price_df = price_df[price_df['Date'].dt.weekday < 5]
        price_df = price_df[['Date', 'CLOSE']]
        price_df.columns = ['Date', 'Price']
        self.expiry_date = price_df['Date'].iloc[-1]
        #vol_df = pandas.read_csv(r"C:\temp\gas_vol.csv")
        #vol_df['Unnamed: 0'] = pandas.to_datetime(vol_df['Unnamed: 0'])
        #vol_df = vol_df[vol_df['Unnamed: 0'].dt.weekday < 5]
        #vol_df.columns = ['Date', 'Vol']
        #df = price_df.merge(vol_df, how='left', on='Date')
        self.df = price_df.iloc[-252:,:]
        self.estimation_length = estimation_length
        self.valdate = valdate
        self.nsim = nsim
        self.gas_data = self.df[(self.df['Date'] >= self.valdate - relativedelta.relativedelta(months=self.estimation_length)) & (self.df['Date'] < self.valdate)]['Price'].to_numpy()
        self.returns = numpy.diff(numpy.log(self.gas_data))  # "Y" is "returns" here
        self.mean_init = numpy.average(self.returns)
        self.var_init = numpy.std(self.returns) ** 2
        self.fixed_result = fixed_result
        #self.vix_series = self.df.iloc[self.si:self.si + self.estimation_length + 1]['Vol'].to_numpy()
        #self.vixTF = vix
        #if self.vixTF is True:
        #    self.vix_series = self.vix_series / 100
        # self.s0 = price_series[0]
        self.T = len(self.returns)
        self.p = p
        self.q = q
        self.o = o
        self.long_run = long_run
        self.plot = plot
        # self.init_r = init_r
        self.dist = dist
        self.rate = rate
        self.egarch_mle()
        # if self.result.success == True:
        #     self.insample_simulation()
        #
        #     # spot = pandas.read_csv(r'D:\TU_DORTMUND\Thesis\Data\price\Spot_Price.csv')
        #     # spot['Date'] = pandas.to_datetime(spot['Date'])
        #     # self.spot = spot
        #     self.get_step_till_expiry()
        #     self.get_step_till_end()
        # else:
        #     pass
        self.get_step_till_expiry()
        self.get_step_till_end()
        self.insample_simulation()

    def loss(self, params):
        mu = params[0]
        omega = params[1]
        alpha = params[2]
        gamma = params[3]
        beta = params[4]
        nu = params[5] # degree of freedom
        delta = params[6] # skewness level

        # calculating long-run volatility

        long_run = numpy.exp(omega / (1 - beta)) ** 0.5

        # calculating realized and conditional volatilities

        resi = self.returns - mu
        indic = [1 if resi[i] > 0 else -1 for i in range(self.T)]
        realised = abs(resi)

        conditional = numpy.zeros(len(self.returns))
        conditional[0] = long_run

        for t in range(1, len(self.returns)):
            # conditional[t] = numpy.sqrt(numpy.exp(omega + alpha*(abs(resi[t-1]/conditional[t-1]) - (2/numpy.pi)**0.5) +
            #                                       gamma*(resi[t-1]/conditional[t-1]) + beta*numpy.log(conditional[t-1]**2)))

            conditional[t] = numpy.sqrt(numpy.exp(omega + alpha * indic[t - 1] * (resi[t - 1] / conditional[t - 1]) +
                                                  gamma * (resi[t - 1] / conditional[t - 1]) + beta * numpy.log(
                conditional[t - 1] ** 2)))

        likelihood = 2/(delta + 1/delta) * scipy.special.gamma(0.5*(nu + 1))/ (scipy.special.gamma(0.5*nu)*(numpy.pi*nu)**0.5*conditional) * \
                     (1 + realised**2/(conditional**2*nu)*(1/delta**2 * numpy.sign(realised) + delta**2 * numpy.sign(realised)))**(-(nu+1)/2)

        log_likelihood = numpy.sum(numpy.log(likelihood))

        return -log_likelihood

    def loss_vix(self, params):
        mu = params[0]
        omega = params[1]
        alpha = params[2]
        gamma = params[3]
        beta = params[4]
        theta = params[5]

        # calculating long-run volatility

        long_run = numpy.exp(
            (omega) / (1 - beta)) ** 0.5  # numpy.exp((omega+theta*numpy.log(self.long_run**2))/(1-beta))**0.5

        # calculating realized and conditional volatilities

        resi = self.returns - mu
        indic = [1 if resi[i] > 0 else -1 for i in range(self.T)]
        realised = abs(resi)

        conditional = numpy.zeros(len(self.returns))
        conditional[0] = long_run

        for t in range(1, len(self.returns)):
            # conditional[t] = numpy.sqrt(numpy.exp(omega + alpha*(abs(resi[t-1]/conditional[t-1]) - (2/numpy.pi)**0.5) +
            #                                       gamma*(resi[t-1]/conditional[t-1]) + beta*numpy.log(conditional[t-1]**2)))

            conditional[t] = numpy.sqrt(numpy.exp(omega + alpha * indic[t - 1] * (resi[t - 1] / conditional[t - 1]) +
                                                  gamma * (resi[t - 1] / conditional[t - 1]) + beta * numpy.log(
                conditional[t - 1] ** 2) + theta * numpy.log(self.vix_series[t - 1] ** 2)))

        likelihood = 1 / ((2 * numpy.pi) ** 0.5 * conditional) * numpy.exp(-realised ** 2 / (2 * conditional ** 2))

        log_likelihood = numpy.sum(numpy.log(likelihood))

        print(log_likelihood)

        return -log_likelihood

    def egarch_mle(self):
        if self.fixed_result is None:
            self.result = minimize(self.loss, [self.mean_init, self.var_init, 0, 0, 0, 2.1, 1], method="SLSQP",
                               options={'maxiter': 10000})
            self._params_dict = {'mu': self.result['x'][0],
                             'omega': self.result['x'][1],
                             'alpha': self.result['x'][2],
                             'gamma': self.result['x'][3],
                             'beta': self.result['x'][4],
                             'nu': self.result['x'][5],
                             'delta': self.result['x'][6]}
        else:
            self.result = self.fixed_result
            self._params_dict = {'mu': self.result['mu'],
                             'omega': self.result['omega'],
                             'alpha': self.result['alpha'],
                             'gamma': self.result['gamma'],
                             'beta': self.result['beta'],
                             'nu': self.result['nu'],
                             'delta': self.result['delta']}
        return self.result

    def get_paths(self, s0=18.56111935, nsteps=2000, nsim=100, v0=None):
        assert self._params_dict is not None, "Parameters have not been calibrated yet"
        mu = self._params_dict.get("mu")
        omega = self._params_dict.get("omega")
        alpha = self._params_dict.get("alpha")
        gamma = self._params_dict.get("gamma")
        beta = self._params_dict.get("beta")
        nu = self._params_dict.get("nu")
        delta = self._params_dict.get("delta")
        long_run = numpy.exp(omega / (1 - beta)) ** 0.5
        if v0 is None:
            v0 = long_run
        # dt = 1 / nsteps
        resi = self.returns - mu
        simulated_r = numpy.zeros([nsim, nsteps + 1])
        simulated_r[:, 0] = resi[0] + mu
        simulated_volas = numpy.zeros([nsim, nsteps + 1])
        simulated_volas[:, 0] = v0

        # --- get randomness (correlated for each t=1,...,T, as corr(W_S, W_V) = rho)
        numpy.random.seed(42)
        #Z_V = numpy.random.normal(size=[nsim, nsteps + 1])
        Z_V = SST(mu=0, sigma = 1, nu = delta, tau = nu).r([nsim, nsteps + 1])
        indic = numpy.sign(Z_V)

        # ----- generate paths
        for t in range(1, nsteps + 1):
            # --- get the time-varying volatility component
            # simulated_volas[:, t] = numpy.sqrt(numpy.exp(omega + alpha*(abs(simulated_r[:, t-1]/simulated_volas[:, t-1]) - (2/numpy.pi)**0.5) +
            #                                      gamma*(simulated_r[:, t-1]/simulated_volas[:, t-1]) + beta*numpy.log(simulated_volas[:, t-1]**2)))
            simulated_volas[:, t] = numpy.sqrt(
                numpy.exp(omega + alpha * indic[:, t - 1] * simulated_r[:, t - 1] / simulated_volas[:, t - 1] +
                          gamma * (simulated_r[:, t - 1] / simulated_volas[:, t - 1]) + beta * numpy.log(
                    simulated_volas[:, t - 1] ** 2)))
            simulated_r[:, t] = mu + Z_V[:, t] * simulated_volas[:, t]

            # --- get the total price dynamics
        simulated_paths = numpy.zeros([nsim, nsteps + 1])
        simulated_paths[:, 0] = s0
        for i in range(nsteps):
            simulated_paths[:, i + 1] = simulated_paths[:, i] * numpy.exp(simulated_r[:, i + 1])
        return simulated_paths, simulated_volas

    def insample_simulation(self):

        s0 = self.gas_data[0]
        simulated_paths, simulated_variances = self.get_paths(s0=s0, nsteps=len(self.gas_data) - 1, nsim=self.nsim,
                                                              v0=None)
        self.price_simu_insample = pandas.DataFrame(simulated_paths).T
        self.vpast = pandas.DataFrame(simulated_variances).iloc[:, -1].mean()
        price_forecast = [x for x in self.price_simu_insample.iloc[-1] if x >0 and x < numpy.inf]
        if len(price_forecast) > 0:
            self.insample_crps, fcrps, acrps = pscore(price_forecast,
                                                  self.gas_data[:len(self.gas_data)][-1]).compute()
        else:
            self.insample_crps = numpy.nan
        print(f"CRPS In-Sample: {self.insample_crps}")

        if self.plot == True:
            pandas.concat([self.price_simu_insample.iloc[:, :10],
                           pandas.DataFrame(self.gas_data[:len(self.gas_data)]).rename(columns={0: "Price"})],
                          axis=1).plot(
                color=['black', 'black', 'black', 'black', 'black', 'black', 'black', 'black', 'black', 'black',
                       'red'])

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
        simulated_paths, simulated_variances = self.get_paths(s0=s0, nsteps=step, nsim=self.nsim, v0=self.vpast)
        price_simu_outsample = pandas.DataFrame(simulated_paths).T
        self.terminal_price = self.df['Price'].iloc[-1]#self.spot[self.spot['Date'] == self.first_busday_after_expiry]['Close'].values[0]
        #terminal_price = self.df['Price'].iloc[-1]
        price_forecast = [x for x in price_simu_outsample.iloc[-1] if x > 0 and x < numpy.inf]
        if len(price_forecast) > 0:
            self.outsample_crps, fcrps1, acrps1 = pscore(price_forecast, self.terminal_price).compute()
        else:
            self.outsample_crps = numpy.nan
        print(f"CRPS Out-Sample: {self.outsample_crps}")
        self.outsample_forecast = price_forecast
        return price_simu_outsample, self.outsample_crps

    def call_option_price(self, step, moneyness=0.9):
        price_simu, _ = self.outsample_simulation(step)
        self.strike = self.gas_data[-1]*moneyness
        self.t2e = (self.expiry_date - self.valdate).days // 30
        payoff = numpy.mean([i if i > 0 else 0 for i in list(price_simu.iloc[-1,:] - self.strike)])
        discount = (1+self.rate)**(self.t2e/12)
        call_price = payoff/discount
        return call_price

class Delta_Hedge():

    def __init__(self, egarch_obj, rate = 0, step = 60,
                 s0_1 = 0.999,
                 s0_2 = 1.001,
                 moneyness = 0.9):
        self.obj = egarch_obj
        self.t2e = (self.obj.expiry_date - self.obj.valdate).days//30
        self.rate = rate
        self.step = step
        self.s0_1 = s0_1
        self.s0_2 = s0_2
        self.moneyness = moneyness
        self.delta_compute(self.moneyness)

    def outsample_simulation(self, step, s_rate):
        s0 = self.obj.gas_data[-1] * s_rate
        simulated_paths, simulated_variances = self.obj.get_paths(s0=s0, nsteps=step, nsim=self.obj.nsim, v0=self.obj.vpast)
        price_simu_outsample = pandas.DataFrame(simulated_paths).T
        self.terminal_price = self.obj.spot[self.obj.spot['Date'] == self.obj.first_busday_after_expiry]['Close'].values[0]
        #terminal_price = self.df['Price'].iloc[-1]
        outsample_crps, fcrps1, acrps1 = pscore(price_simu_outsample.iloc[-1], self.terminal_price).compute()
        print(f"CRPS Out-Sample: {outsample_crps}")
        return price_simu_outsample, outsample_crps

    def call_option_price(self, step, s0_rate, moneyness):
        price_simu, _ = self.outsample_simulation(step, s0_rate)
        self.strike = self.obj.gas_data[-1]*moneyness
        payoff = numpy.mean([i if i > 0 else 0 for i in list(price_simu.iloc[-1,:] - self.strike)])
        discount = (1+self.rate)**(self.t2e/12)
        call_price = payoff/discount
        return call_price

    def delta_compute(self, moneyness):
        call_1 = self.call_option_price(self.step, self.s0_1, moneyness)
        call_2 = self.call_option_price(self.step, self.s0_2, moneyness)

        s0_1 = self.obj.gas_data[-1] * self.s0_1
        s0_2 = self.obj.gas_data[-1] * self.s0_2

        self.delta = (call_1-call_2)/(s0_1-s0_2)

        return self.delta

if __name__ == '__main__':
    contract = 'Nov-20'
    valdate = datetime.datetime(2020, 8, 4)
    obj0 = EGARCH_CNT(contract=contract, valdate=valdate, estimation_length=3)
    obj0.outsample_simulation(step=obj0.steps_needed)
    obj1 = EGARCH_SST(contract=contract, valdate=valdate, estimation_length=3)
    obj1.outsample_simulation(step=obj0.steps_needed)
    ### RUN EGARCH for contract
    contract = 'Jan-24'
    start_date = datetime.datetime(2023, 11, 1)-datetime.timedelta(days=1)
    moneyness = 1.1
    call_price = 0
    while call_price <=0:
        start_date = start_date + datetime.timedelta(days=1)
        while start_date.isoweekday() > 5 or start_date in cal.holidays():
            start_date = start_date + datetime.timedelta(days=1)
        obj_0 = EGARCH_CNT(contract=contract, valdate=start_date, estimation_length=3)
        call_price = obj_0.call_option_price(step=obj_0.steps_needed, moneyness=moneyness)
    valdate = start_date
    rslt_dict = dict()
    rslt_dict['valdate'] = [start_date-datetime.timedelta(days=1)]
    rslt_dict['S'] = [0]
    rslt_dict['delta'] = [0]
    while valdate < datetime.datetime(2024, 1, 2):
        while valdate.isoweekday() > 5 or valdate in cal.holidays():
            valdate = valdate + datetime.timedelta(days=1)
        try:
            obj = EGARCH_CNT(contract=contract, valdate=valdate, estimation_length=3)
            delta_obj = Delta_Hedge(egarch_obj=obj, step=obj.steps_needed, moneyness=moneyness)
            if delta_obj.delta > 0:
                rslt_dict['valdate'].append(valdate)
                rslt_dict['S'].append(obj.gas_data[-1])
                rslt_dict['delta'].append(delta_obj.delta)
        except:
            pass

        valdate = valdate + datetime.timedelta(days=1)


    hedge_df = pandas.DataFrame(rslt_dict)
    delta_hedge = [(hedge_df['delta'][i] - hedge_df['delta'][i-1])*hedge_df['S'][i]*(1+obj_0.rate)**(obj_0.t2e/12) for i in range(1, hedge_df.shape[0])]
    delta_hedge.insert(0, 0)
    hedge_df['delta_hedge'] = delta_hedge
    strike = obj_0.gas_data[-1]*moneyness
    execute_price = obj_0.terminal_price
    portfolio = -numpy.sum(hedge_df['delta_hedge']) + call_price*(1+obj_0.rate)**(obj_0.t2e/12) + hedge_df['delta'].iloc[-1]*execute_price
    payoff = numpy.max([execute_price - strike, 0])

    #### Short a Call Option and long underlying to hedge the position
    pnl = portfolio-payoff








    #
    # insample_crps_series = []
    # outsample_crps_series_5 = []
    # outsample_crps_series_20 = []
    # outsample_crps_series_60 = []
    # estimation_length = 60
    # for si in range(100):
    #
    #     obj = EGARCH(nsim=10000,
    #                  estimation_length=estimation_length,
    #                  si=si,
    #                  p = 1,
    #                  q = 1,
    #                  o = 1,
    #                  long_run = 0.1,
    #                  dist = 'gaussian',
    #                  plot = False)
    #     insample_crps_series.append(obj.insample_crps)
    #     price_path, real_path, outsample_crps_5 = obj.outsample_simulation(step=5)
    #     outsample_crps_series_5.append(outsample_crps_5)
    #
    #     price_path, real_path, outsample_crps_20 = obj.outsample_simulation(step=20)
    #     outsample_crps_series_20.append(outsample_crps_20)
    #
    #     price_path, real_path, outsample_crps_60 = obj.outsample_simulation(step=60)
    #     outsample_crps_series_60.append(outsample_crps_60)
    #
    # # df = pandas.read_csv(r"C:\temp\gas_price.csv")
    # # df['Unnamed: 0'] = pandas.to_datetime(df['Unnamed: 0'])
    # # df = df[df['Unnamed: 0'].dt.weekday < 5]
    # # df.columns = ['Date', 'Price']
    # #valdate = df['Date'][estimation_length+1:estimation_length+100+1]
    # crps_df = pandas.DataFrame([outsample_crps_series_5, outsample_crps_series_20, outsample_crps_series_60]).T
    # crps_df.columns = ['Step5', 'Step20', 'Step60']
    #
    # crps_df.to_csv(rf'C:\temp\crps_EGARCH.csv', index=False)


    #
    #
    #
    #
    # price_df = pandas.read_csv(r"C:\temp\gas_price.csv")
    # price_df['Unnamed: 0'] = pandas.to_datetime(price_df['Unnamed: 0'])
    # price_df = price_df[price_df['Unnamed: 0'].dt.weekday < 5]
    # price_df.columns = ['Date', 'Price']
    # # vix_df = pandas.read_csv(r'C:\temp\VIX_History.csv')[['DATE', 'CLOSE']]
    # # vix_df['DATE'] = pandas.to_datetime(vix_df['DATE'])
    # # vix_df.columns = ['Date', 'VIX']
    #
    # vol_df = pandas.read_csv(r"C:\temp\gas_vol.csv")
    # vol_df['Unnamed: 0'] = pandas.to_datetime(vol_df['Unnamed: 0'])
    # vol_df = vol_df[vol_df['Unnamed: 0'].dt.weekday < 5]
    # vol_df.columns = ['Date', 'Vol']
    # df = price_df.merge(vol_df, how='left',on = 'Date')
    # df.fillna(method="ffill", inplace=True)
    #
    # # from arch import arch_model
    # # vix_arch = arch_model(vix/100, p = 1, q = 1, o = 1, power = 2, dist = 'gaussian', vol = 'EGARCH')
    # # rs = vix_arch.fit()
    # #
    # # obj = EGARCH(gas_data, vix, long_run=(vix/100)[-1])
    # # ml0 = -obj.result.fun
    # # simulated_paths, simulated_vol = obj.get_paths_vixincluded(s0=gas_data[0], nsteps=len(gas_data)-1, nsim=1000)
    # # insample_price_sim = pandas.DataFrame(simulated_paths).T
    # # insample_vol_sim = pandas.DataFrame(simulated_vol).T
    # # insample_score0 = pscore(insample_price_sim.iloc[-1], gas_data[-1]).compute()
    # # print(f"VIX:{pscore(insample_price_sim.iloc[-1], gas_data[-1]).compute()}")
    #
    # ### loop to see the performance of EGARCH overtime
    #
    #
    # estimation_length_list = [60]
    # nsteps_list = [5]
    # for nsteps, estimation_length in zip(nsteps_list, estimation_length_list):
    #     insample_crps = []
    #     outsample_crps = []
    #     for si in range(100):
    #         print( df['Date'][si])
    #         print('-----')
    #
    #         gas_data = df.iloc[si:si + estimation_length + 1]['Price'].to_numpy()
    #         vix = df.iloc[si:si + estimation_length + 1]['Vol'].to_numpy()
    #         vix = vix/numpy.sqrt(252)
    #         #### insample crps
    #         obj0 = EGARCH(gas_data, None)
    #         ml0 = -obj0.result.fun
    #         print(f"Log LH: {ml0}")
    #         simulated_paths0, simulated_vol0 = obj0.get_paths(s0=gas_data[0], nsteps=len(gas_data)-1, nsim=1000)
    #         insample_price_sim = pandas.DataFrame(simulated_paths0).T
    #         insample_vol_sim = pandas.DataFrame(simulated_vol0).T
    #         insample_score = pscore(insample_price_sim.iloc[-1], gas_data[-1]).compute()
    #
    #         ### initial_vol for outsample forecast
    #         v0_outsample = insample_vol_sim.iloc[-1].mean()
    #
    #
    #         #### outsample crps
    #         simulated_paths_1, simulated_vol_1 = obj0.get_paths(s0=gas_data[-1], nsteps=nsteps, nsim=1000)
    #         outsample_price_sim = pandas.DataFrame(simulated_paths_1).T
    #         outsample_vol_sim = pandas.DataFrame(simulated_vol_1).T
    #         real_gas = df.iloc[si + estimation_length: si + estimation_length + nsteps + 1]['Price'].to_numpy()
    #         outsample_score = pscore(outsample_price_sim.iloc[-1], real_gas[-1]).compute()
    #
    #
    #         insample_crps.append(insample_score[0])
    #         outsample_crps.append(outsample_score[0])
    #
    #         print(si/1700)
    #
    #     crps_df = pandas.DataFrame([insample_crps, outsample_crps]).T
    #
    #     crps_df.to_csv(rf'C:\temp\crps_EGARCH_{nsteps}.csv', index=False)








