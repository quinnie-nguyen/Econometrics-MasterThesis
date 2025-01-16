import sys

import dateutil.relativedelta
import pandas
sys.path.append(r'C:\Users\UI620224\PycharmProjects\OPTION_TRADING_STRATEGIES\optiontradingstrategies')
import util.pb_utilities as pb
import datetime
import numpy
from scipy.optimize import curve_fit

def term_struct_vol(tckr):

    cnt = pb.cnt_from_tckr(tckr)
    expiry = pb.get_expiry_date_non_standard(cnt)
    start_date = expiry - datetime.timedelta(days=365)
    end_date = expiry - datetime.timedelta(days=100)
    df = pb.get_price_timeseries(cnt, start_date, end_date)
    df.index = pandas.to_datetime(df.index)
    df = df[df.index.weekday <5]

    std_df = numpy.log(df).diff().dropna().rolling(20).std().dropna()

    t2e = numpy.array([(expiry - i).days / 252 for i in std_df.index])
    vol = std_df.to_numpy().flatten()
    #z = numpy.polyfit(t2e, vol,3)
    #f = numpy.poly1d(z)
    def func(t2e, a, b, alpha):
        return a + b * numpy.exp(-alpha * t2e)

    fittedParameters, pcov = curve_fit(func, t2e, vol,maxfev = 100000000)

    std_hat = pandas.DataFrame(func(t2e, *fittedParameters))

    result = pandas.DataFrame([vol]).transpose()
    result.columns = ['Realized_Vol']
    result['Fitted_Vol'] = std_hat
    result.index = std_df.index
    result.plot()

    return result, fittedParameters

def term_struct_sim(params,dt = 1/252,f_0 = 2.5, ts_length = 100, nr_paths = 200):
    a = params[0]
    b = params[1]
    alpha = 4#params[2]

    sample = numpy.random.normal(0, 1, size=[ts_length, nr_paths]) #or 1/252
    sample = sample.reshape(ts_length, 1, nr_paths)
    brwn_inc_price_array = sample[:, 0]
    t2e_array = [t2e/252 for t2e in range(ts_length, -1, -1)]
    
    vol_path = [a + b*numpy.exp(-alpha*t2e) for t2e in t2e_array]
    vol_path = numpy.array(vol_path)
    f = [numpy.ones(nr_paths)*f_0]
    for t in range(1, ts_length):
        f.append(f[t-1]*numpy.exp(vol_path[t-1]*brwn_inc_price_array[t] - 0.5*vol_path[t-1]**2*dt))
    df_price = pandas.DataFrame(f)
    df_vol = pandas.DataFrame(vol_path)
    return df_price, df_vol

#Simulate Normal Compound Poisson Process
def normal_compound_poisson_process(lambda_, T, m, delta):

    t = 0
    jumps = numpy.zeros(200)
    event_values = []
    event_times = []
    event_values.append(0)
    event_times.append(0)
    while t < T:
        t = t + numpy.random.exponential(1 / lambda_)
        jumps = jumps + numpy.random.normal(m, delta, size=(1,200))
        if t < T:
            event_values.append(jumps)
            event_times.append(t)

    return event_values, event_times


def MJD_process(S0, mu, sigma, lambda_, T, m, delta, num_steps):
    dT = T / num_steps
    price = [numpy.ones(200)*S0]
    Z = numpy.random.normal(0, 1, size=(num_steps, 200))
    dW = Z*numpy.sqrt(dT)
    k = numpy.exp(m + .5 * delta ** 2) - 1
    for t in range(1, num_steps):
        price.append(price[t-1]*numpy.exp((mu - .5 * sigma ** 2 - lambda_ * k) * dT + sigma * dW[t] + jumps))


    return pandas.DataFrame(price)


if __name__ == "__main__":
    tckr = "J_24_1"
    fitted_std, params = term_struct_vol(tckr)
    end_fitted = fitted_std.index[-1]
    # if end_fitted.weekday < 4:
    #     start_simu = end_fitted + dateutil.relativedelta.relativedelta(days=1)
    # elif end_fitted.weekday == 5:
    #     start_simua = end_fitted + dateutil.relativedelta.relativedelta(days=3)
    real_price = pb.get_price_timeseries(pb.cnt_from_tckr(tckr), end_fitted)
    real_price.index = pandas.to_datetime(real_price.index)
    real_price = real_price[real_price.index.weekday<5]
    f_0 = pb.get_contract_price(pb.cnt_from_tckr(tckr), end_fitted)
    mu =  -0.010565607404298774
    sigma = 0.0011631564477022562**0.5*252**0.5
    lambda_ =0.43531851710781366
    m = 0.01739289829788182
    delta = 0.010032292974480913
    simu_len = numpy.busday_count(begindates=end_fitted.strftime("%Y-%m-%d"),
                                  enddates=pb.get_expiry_date_non_standard(pb.cnt_from_tckr(tckr)).strftime("%Y-%m-%d"))
    dt = 1/252
    T = dt*simu_len
    x = MJD_process(f_0, mu, sigma, lambda_, T, m, delta, simu_len)

    df_price, df_vol = term_struct_sim(params, f_0 = f_0, ts_length=simu_len, nr_paths=200)

    
    valdate = datetime.datetime(2015, 1, 1)
    df_price = pb.get_price_timeseries(pb.cnt_from_tckr(tckr), valdate)
    numpy.log(df_price).diff().dropna().plot()