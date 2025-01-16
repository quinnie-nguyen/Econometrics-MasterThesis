import pandas
import numpy
import datetime
from pandas.tseries.holiday import USFederalHolidayCalendar
cal = USFederalHolidayCalendar()
import utils
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def compute_pacf(data, nlags):
    """
    Calculate Partial Autocorrelation Function (PACF) using Durbin-Levinson recursion.

    Parameters:
        data (array-like): Time series data.
        nlags (int): Number of lags for which to calculate PACF.

    Returns:
        pacf (array): PACF values for lags 0 to nlags.
    """
    n = len(data)
    mean = np.mean(data)
    data = data - mean  # Demean the data

    # Compute autocorrelations
    acf = np.correlate(data, data, mode='full') / (n * np.var(data))
    acf = acf[n-1:]  # Keep only non-negative lags

    pacf = [1]  # PACF(0) is always 1
    phi_prev = []

    for k in range(1, nlags + 1):
        # Solve Yule-Walker equations for lag k
        toeplitz_matrix = np.array([acf[abs(i - j)] for i in range(k) for j in range(k)]).reshape(k, k)
        rhs = acf[1:k+1]
        phi_k = np.linalg.solve(toeplitz_matrix, rhs)
        pacf.append(phi_k[-1])  # Append PACF value for lag k

    return np.array(pacf)


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





    df = pd.DataFrame(data)

    # Convert Month to a categorical variable to ensure proper ordering
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    df['Month'] = pd.Categorical(df['Month'], categories=month_order, ordered=True)

    # Calculate average Day-Ahead and Month-Ahead values across years for each month
    avg_df = df.groupby('Month')[['Day-Ahead', 'Month-Ahead']].mean().reset_index()
    avg_df['Year'] = 'Average'

    # Append average data for plotting
    df = pd.concat([df, avg_df], ignore_index=True)

    # Set up the plotting style
    sns.set(style='whitegrid')

    # Create a grid of two plots: Day-Ahead and Month-Ahead
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    colors = sns.color_palette('tab10', len(df['Year'].unique()))

    # Plot Day-Ahead data
    sns.lineplot(
        data=df,
        x='Month',
        y='Day-Ahead',
        hue='Year',
        ax=axes[0],
        palette=colors,
        linewidth=1
    )
    axes[0].set_title('Day-Ahead')
    axes[0].axhline(0, color='black', linestyle='--', linewidth=0.8)  # Add zero line

    # Plot Month-Ahead data
    sns.lineplot(
        data=df,
        x='Month',
        y='Month-Ahead',
        hue='Year',
        ax=axes[1],
        palette=colors,
        linewidth=1
    )
    axes[1].set_title('Month-Ahead')
    axes[1].axhline(0, color='black', linestyle='--', linewidth=0.8)  # Add zero line

    # Adjust legend and layout
    for ax in axes:
        ax.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_xlabel('Month')
        ax.set_ylabel('Value')
        ax.set_xticklabels(month_order, rotation=45)

    plt.tight_layout()
    plt.show()


