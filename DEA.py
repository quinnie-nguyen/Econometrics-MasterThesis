import pandas
import numpy
import datetime
from pandas.tseries.holiday import USFederalHolidayCalendar
cal = USFederalHolidayCalendar()
import utils
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

month = ['Jan',
      'Feb',
      'Mar',
      'Apr',
      'May',
      'Jun',
      'Jul',
      'Aug',
      'Sep',
      'Oct',
      'Nov',
      'Dec']

year = [17, 18, 19, 20, 21, 22, 23, 24]
crps_egarch = r"D:\TU_DORTMUND\Thesis\Data\crps\EGARCH"

crps_root_dir = r'D:\TU_DORTMUND\Thesis\Data\crps\EGARCH'
crps_df = pandas.DataFrame()
for yy in year:
    for mm in month:
        contract = f"{mm}-{yy}"
        df_temp = pandas.read_csv(f"{crps_root_dir}\{contract}.csv", index_col=0, parse_dates=True).iloc[-3:, :].T
        df_temp.columns = ['6M', '3M', '1M']
        df_temp.index = [contract]
        crps_df = pandas.concat([crps_df, df_temp], axis=0)


if __name__ == '__main__':
    # Example DataFrame Structure
    # Replace with your actual data loading step
    jan = {
        'Year': [16, 17, 18, 19, 20, 21, 22, 23, 24],
        'Month': ['Jan'],
    }
    data = dict()
    data['Cnt'] = []
    data['data'] = []
    for mm in jan['Month']:
        for yy in jan['Year']:
            cnt_name = f"{mm}-{yy}"
            df_temp = utils.get_data_by_cnt(mm, yy, 252)
            df_temp.reset_index(inplace=True)
            data['Cnt'].append(cnt_name)
            data['data'].append(df_temp)

    price_data = pandas.DataFrame()
    for price in data['data']:
        price_data = pandas.concat([price_data, price['CLOSE']], axis=1)
    price_data.columns = data['Cnt']
    price_data.reset_index(inplace=True)
    price_data = price_data.melt(value_name='CLOSE', var_name='CNT', id_vars='index')

    return_data = pandas.DataFrame()
    for returns in data['data']:
        return_data = pandas.concat([return_data, returns['LOG_RETURN']], axis=1)
    return_data.columns = data['Cnt']
    return_data.reset_index(inplace=True)

    fig, axes = plt.subplots(1, 1, figsize=(14, 6), sharey=True)
    sns.lineplot(data=price_data, x='index', y='CLOSE', hue='CNT', linewidth=1)
    #axes.set_title('January Futures Contract daily prices within one year before expiry')
    axes.set_xlabel('Day')
    axes.set_ylabel('USD/MMBtu')

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharey=True)
    for col, ax in zip(['Jan-16', 'Jan-19', 'Jan-22', 'Jan-24'], axes.flatten()):
        ax.plot(return_data[col].values)
        ax.set_title(col, fontsize = 8)
    plt.xlabel('Day')
    plt.ylabel('Log Return')
    plt.show()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    for col, ax in zip(['Jan-16', 'Jan-19', 'Jan-22', 'Jan-24'], axes.flatten()):
        ax.hist(return_data[col].values, bins = 25)
        ax.grid(color='grey', linestyle='--', linewidth=0.2)
        ax.set_title(col, fontsize = 8)
    plt.show()

    # Example DataFrame Structure July
    # Replace with your actual data loading step
    jul = {
        'Year': [16, 17, 18, 19, 20, 21, 22, 23, 24],
        'Month': ['Jul'],
    }
    data = dict()
    data['Cnt'] = []
    data['data'] = []
    for mm in jul['Month']:
        for yy in jul['Year']:
            cnt_name = f"{mm}-{yy}"
            df_temp = utils.get_data_by_cnt(mm, yy, 252)
            df_temp.reset_index(inplace=True)
            data['Cnt'].append(cnt_name)
            data['data'].append(df_temp)

    price_data = pandas.DataFrame()
    for price in data['data']:
        price_data = pandas.concat([price_data, price['CLOSE']], axis=1)
    price_data.columns = data['Cnt']
    price_data.reset_index(inplace=True)
    price_data = price_data.melt(value_name='CLOSE', var_name='CNT', id_vars='index')

    return_data = pandas.DataFrame()
    for returns in data['data']:
        return_data = pandas.concat([return_data, returns['LOG_RETURN']], axis=1)
    return_data.columns = data['Cnt']
    return_data.reset_index(inplace=True)

    fig, axes = plt.subplots(1, 1, figsize=(14, 6), sharey=True)
    sns.lineplot(data=price_data, x='index', y='CLOSE', hue='CNT', linewidth=1)
    #axes.set_title('July Futures Contract daily prices within one year before expiry')
    axes.set_xlabel('Day')
    axes.set_ylabel('USD/MMBtu')

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharey=True)
    for col, ax in zip(['Jul-16', 'Jul-19', 'Jul-22', 'Jul-24'], axes.flatten()):
        ax.plot(return_data[col].values)
        ax.set_title(col, fontsize = 8)
    plt.xlabel('Day')
    plt.ylabel('Log Return')
    plt.show()



