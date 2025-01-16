import sys

import pandas

sys.path.append(r'C:\Users\UI620224\PycharmProjects\OPTION_TRADING_STRATEGIES\optiontradingstrategies')
import util.pb_utilities as pb
import datetime
import pandas


def get_fwd_cnt_tckr(cnt, month_list = [1, 2, 3, 4, 5, 6]):
    fwd_cnt_list = [pb.cnt_tckr(cnt)]
    for mm in month_list:
        fwd_cnt_list.append(pb.get_front_month_contract(cnt.commodity, cnt.region, cnt.product, pb.get_expiry_date_non_standard(cnt)+dateutil.relativedelta.relativedelta(months = mm), False))
    return fwd_cnt_list

def get_price_ts(cnt_list, valdate):
    df = pandas.DataFrame()
    df_list = [pb.get_price_timeseries(cnt, valdate) for cnt in cnt_list]
    


if __name__ == '__main__':

    tckr = 'Z_24_1'
    valdate = datetime.datetime(2023, 6, 1)
    
    cnt = pb.cnt_from_tckr(tckr)
    fwd_tckr_list = get_fwd_cnt_tckr(cnt)
    df_list = [pb.get_price_timeseries(pb.cnt_from_tckr(tckr), valdate) for tckr in fwd_tckr_list]
    df_total = pb.merge_df_list(df_list)
    df_total.columns = fwd_tckr_list
    df_total = df_total[df_total.index.weekday < 5]
    log_price_df = numpy.log(df_total)
    # putin = pandas.read_csv(r'%s\historical_data.csv' % (r'C:\Users\UI620224\PycharmProjects\news\result\Putin'))
    # putin.Date = pandas.to_datetime(putin.Date)
    # putin.index = putin.Date
    # putin.index.name = None
    # putin = putin[['No. of articles']]
    # df = pb.merge_pb(log_price_df, putin)
    fit_data = log_price_df.iloc[:180, :]
    var_obj = pb.VAR(fit_data)
    fit = var_obj.fit(1, trend = 'n')
    forecast_length = len(log_price_df) - len(fit_data)
    price_forecast = pandas.DataFrame(fit.forecast(y =fit_data.iloc[-(1):].to_numpy(),steps = forecast_length), columns =fit_data.columns)
    real_price = log_price_df.iloc[180:, :]
    
    
    