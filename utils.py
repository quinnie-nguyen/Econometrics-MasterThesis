import pandas
import numpy
import datetime
from pandas.tseries.holiday import USFederalHolidayCalendar
cal = USFederalHolidayCalendar()

rootdir = r'D:\TU_DORTMUND\Thesis\Data\price'

def get_price_data(no_month_ahead=1, start_date = datetime.datetime(2016, 2, 1), end_date = datetime.datetime(2025, 1, 1)):
    month_to_symbol = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    yr_list = [str(yr)[-2:] for yr in range(start_date.year, end_date.year)]
    df_list = dict()
    df_list['cnt_name'] = []
    df_list['data'] = []
    for y in yr_list:
        for m in list(month_to_symbol):
            cnt_name = f"{m}-{y}"
            print(cnt_name)
            df_temp = pandas.read_csv(rf"{rootdir}\{cnt_name}.csv")
            df_list['cnt_name'] = cnt_name
            df_list['data'].append(df_temp)
    df_total = pandas.concat([df for df in df_list['data']], axis=0).drop_duplicates(keep='first')
    return df_total

def get_data_by_cnt(month, year, length = 252):
    cnt_name = f"{month}-{year}"
    price = pandas.read_csv(rf"{rootdir}\{cnt_name}.csv")
    price['LOG_RETURN'] = numpy.log(price['CLOSE']).diff()
    price['Date'] = pandas.to_datetime(price['Date'])
    price = price[['Date', 'CLOSE', 'LOG_RETURN']].iloc[-length:, :]
    return price

if __name__ == '__main__':
    data = get_price_data()
