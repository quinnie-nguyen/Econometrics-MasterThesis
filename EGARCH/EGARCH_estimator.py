from arch import arch_model
import pandas
import numpy
import datetime
import CRPS.CRPS as pscore
import warnings
warnings.filterwarnings("ignore")
class EGARCH():

    def __init__(self, p = 1, o = 1, q = 1, power = 2, error_distribution = 'StudentsT', estimation_length = 60,
                 nsteps = 60, nsimulation = 10000, si = 0):
        self.p = p
        self.q = q
        self.o = o
        self.power = power
        self.error_dist = error_distribution
        self.estimation_length = estimation_length
        self.nsteps = nsteps
        self.nsimulation = nsimulation
        self.si = si
        df = pandas.read_csv(r"C:\temp\gas_price.csv")
        df['Unnamed: 0'] = pandas.to_datetime(df['Unnamed: 0'])
        df = df[df['Unnamed: 0'].dt.weekday < 5]
        df.columns = ['Date', 'Price']
        self.df = df

        self.gas_data = self.df.iloc[self.si:self.si + self.estimation_length + 1]['Price'].to_numpy()
        self.egarch_fit()
        self.egarch_forecast()
    def egarch_fit(self):
        rt = pandas.DataFrame(numpy.log(self.gas_data)).diff().dropna().to_numpy().flatten()
        am = arch_model(rt, p = self.p, q = self.q, o = self.o, power = self.power, dist = self.error_dist, vol = 'EGARCH')
        self.fitted = am.fit(disp = 'off')

    def egarch_forecast(self):
        self.simulated_r = self.fitted.forecast(horizon = self.nsteps, method = 'simulation', simulations = self.nsimulation).simulations.values[self.estimation_length-1]
        s0 = self.gas_data[-1]
        simulated_paths = numpy.zeros([self.nsimulation, self.nsteps + 1])
        simulated_paths[:, 0] = s0
        for i in range(self.nsteps):
            simulated_paths[:, i + 1] = simulated_paths[:, i] * numpy.exp(self.simulated_r[:, i])

        self.price_simu_outsample = pandas.DataFrame(simulated_paths).T
        self.real_gas = self.df.iloc[self.si + self.estimation_length: self.si + self.estimation_length + self.nsteps + 1]['Price'].to_numpy()
        self.outsample_crps, fcrps1, acrps1 = pscore(self.price_simu_outsample.iloc[-1], self.real_gas[-1]).compute()
        print(f"CRPS Out-Sample: {self.outsample_crps}")


if __name__ == '__main__':
    nsteps = 5

    #insample_crps = []
    outsample_crps = []
    for si in range(100):
        obj = EGARCH(p = 1, o = 1, q = 1, power = 2, error_distribution = 'gaussian', estimation_length = 60,
                     nsteps = nsteps, nsimulation = 10000, si = si)
        print(obj.outsample_crps)

        #insample_crps.append(obj.insample_crps)
        outsample_crps.append(obj.outsample_crps)

    crps_df = pandas.DataFrame([outsample_crps]).T

    crps_df.to_csv(rf'C:\temp\crps_EGARCHP_{nsteps}.csv', index=False)


    print(1)



