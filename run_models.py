import pandas
import datetime
import utils
import warnings
import numpy
from pandas.tseries.holiday import USFederalHolidayCalendar
from scipy.stats import wilcoxon
cal = USFederalHolidayCalendar()
import matplotlib.pyplot as plt

# Suppress all warnings
warnings.filterwarnings("ignore")

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

t2m = [3/12, 1/12]
def main():
    for mm in month:
        for yy in year:
            contract = f"{mm}-{yy}"
            print(contract)
            for tt in t2m:
                print(tt)
                egarch_obj = utils.calibration_CNT(contract=contract, t2m=tt, valdate=None, model='EGARCH', delta_hegde=False, estimation_length=3, fixed_result= None)
                egarch_obj.archive_params()
                egarch_obj.archive_crps()

                heston_obj = utils.calibration_CNT(contract=contract, t2m=tt, valdate=None, model='HESTON', delta_hegde=False, estimation_length=3, fixed_result=None)
                heston_obj.archive_params()
                heston_obj.archive_crps()

hparam_root_dir = r'D:\TU_DORTMUND\Thesis\Data\params\HESTON'
eparam_root_dir = r'D:\TU_DORTMUND\Thesis\Data\params\EGARCH'
def get_imsample_crps():
    for mm in month:
        for yy in year:
            contract = f"{mm}-{yy}"
            print(contract)
            for tt in t2m:
                print(tt)
                if tt == 3/12:
                    hfixed_params = pandas.DataFrame(pandas.read_csv(f"{hparam_root_dir}\{contract}.csv", index_col=0, parse_dates=True).iloc[0,:]).T.iloc[0,:].to_dict()
                    efixed_params = pandas.DataFrame(pandas.read_csv(f"{eparam_root_dir}\{contract}.csv", index_col=0, parse_dates=True).iloc[0,:]).T.iloc[0,:].to_dict()
                else:
                    hfixed_params = pandas.DataFrame(
                        pandas.read_csv(f"{hparam_root_dir}\{contract}.csv", index_col=0, parse_dates=True).iloc[1,
                        :]).T.iloc[0, :].to_dict()
                    efixed_params = pandas.DataFrame(
                        pandas.read_csv(f"{eparam_root_dir}\{contract}.csv", index_col=0, parse_dates=True).iloc[1,
                        :]).T.iloc[0, :].to_dict()

                egarch_obj = utils.calibration_CNT(contract=contract, t2m=tt, valdate=None, model='EGARCH', delta_hegde=False, estimation_length=3, fixed_result= efixed_params)
                egarch_obj.archive_probprice()


                #egarch_obj.archive_crps_insample()
                #egarch_obj.archive_crps()


                heston_obj = utils.calibration_CNT(contract=contract, t2m=tt, valdate=None, model='HESTON', delta_hegde=False, estimation_length=3, fixed_result=hfixed_params)
                heston_obj.archive_probprice()
                #heston_obj.archive_crps_insample()
                #heston_obj.archive_crps()


def replication_rslt(contract,
                     model,
                     moneyness,
                     t2e
                     ):

    mobj = utils.calibration_CNT(contract=contract, t2m=t2e, valdate=None, model=model, delta_hegde=False, estimation_length=3, fixed_result=None)
    #target = utils.hegding_pnl(origin_obj=model_obj, moneyness=moneyness)
    call_price = mobj.model_obj.call_option_price(step=mobj.model_obj.steps_needed, moneyness=moneyness)
    if model == "EGARCH":
        fixed_rslt = mobj.model_obj.result
        fixed_param = {'mu': fixed_rslt['x'][0],
                       'omega': fixed_rslt['x'][1],
                       'alpha': fixed_rslt['x'][2],
                       'gamma': fixed_rslt['x'][3],
                       'beta': fixed_rslt['x'][4],
                       'nu': fixed_rslt['x'][5],
                       'delta': fixed_rslt['x'][6]}
    else:
        fixed_param = mobj.model_obj._params_dict

    rslt_dict = dict()
    start_date = mobj.model_obj.valdate + datetime.timedelta(days = 1)
    rslt_dict['valdate'] = [start_date-datetime.timedelta(days=1)]
    rslt_dict['S'] = [0]
    rslt_dict['delta'] = [0]
    while start_date < mobj.model_obj.expiry_date:
        while start_date.isoweekday() > 5 or start_date in cal.holidays():
            start_date = start_date + datetime.timedelta(days=1)
        try:
            obj = utils.calibration_CNT(contract=contract, t2m=None, valdate= start_date, model=model, delta_hegde=False, estimation_length=3, fixed_result=fixed_param)
            delta_obj = utils.Delta_Hedge(egarch_obj=obj.model_obj, step=obj.model_obj.steps_needed, moneyness=moneyness, model=model)
            if delta_obj.delta > 0:
                rslt_dict['valdate'].append(start_date)
                rslt_dict['S'].append(obj.model_obj.gas_data[-1])
                rslt_dict['delta'].append(delta_obj.delta)
        except:
            pass
        start_date = start_date + datetime.timedelta(days=1)

    hedge_df = pandas.DataFrame(rslt_dict)
    delta_hedge = [(hedge_df['delta'][i] - hedge_df['delta'][i-1])*hedge_df['S'][i]*(1+mobj.model_obj.rate)**(1/12) for i in range(1, hedge_df.shape[0])]
    delta_hedge.insert(0, 0)
    hedge_df['delta_hedge'] = delta_hedge
    strike = mobj.model_obj.gas_data[-1]*moneyness
    execute_price = mobj.model_obj.terminal_price
    portfolio = -numpy.sum(hedge_df['delta_hedge']) + call_price*(1+mobj.model_obj.rate)**(1/12) + hedge_df['delta'].iloc[-1]*execute_price
    payoff = numpy.max([execute_price - strike, 0])
    pnl = portfolio - payoff

    dict_temp = {"cnt": contract,
                 "moneyness": moneyness,
                 "t2e": t2e,
                 "model": model,
                 "hedged_pnl": pnl}

    return dict_temp

t2e_pnl = 1/12
model = ['EGARCH', 'HESTON']
moneyness = [0.95, 1.05]
def pnl_main():
    df = pandas.DataFrame()
    for mm in month:
        for yy in year:
            contract = f"{mm}-{yy}"
            print(contract)
            for md in model:
                for mo in moneyness:
                    rslt = replication_rslt(contract,
                                     md,
                                     mo,
                                     t2e_pnl
                                     )
                    df = pandas.concat([df, pandas.DataFrame(rslt, index=[0])], axis = 0)
                    df.to_csv("D:\TU_DORTMUND\Thesis\Data\pnl3.csv")


if __name__ == '__main__':
    import ast
    probprice_dir = r'D:\TU_DORTMUND\Thesis\Data\probprice'
    df_egarch = pandas.DataFrame()
    df_heston = pandas.DataFrame()
    real_egarch = []
    real_heston = []
    cnt = []
    for yy in year:
        for mm in month:
            contract = f"{mm}-{yy}"
            print(contract)

            df_temp_egarch = pandas.read_csv(fr"{probprice_dir}\EGARCH\{contract}.csv", index_col=0).iloc[:, 1] # 0: 3 month, 1: 1 month
            df_temp_heston = pandas.read_csv(fr"{probprice_dir}\HESTON\{contract}.csv", index_col=0).iloc[:, 1]
            if pandas.DataFrame(ast.literal_eval(df_temp_egarch[1])).shape[0] != 0:
                real_egarch.append(float(df_temp_egarch[0]))
                real_heston.append(float(df_temp_heston[0]))
                df_egarch = pandas.concat([df_egarch, pandas.DataFrame(ast.literal_eval(df_temp_egarch[1]))], axis=1)
                df_heston = pandas.concat([df_heston, pandas.DataFrame(ast.literal_eval(df_temp_heston[1]))], axis=1)
                cnt.append(contract)

    df_egarch.columns = cnt
    df_heston.columns = cnt
    df_egarch = df_egarch[(df_egarch <= 1000).all(axis=1)]
    df_heston = df_heston[df_egarch.columns]
    quantiles_e = df_egarch.T.quantile([0.05, 0.25, 0.5, 0.75, 0.95], axis=1).T
    quantiles_e.columns = ['q05', 'q25', 'q50', 'q75', 'q95']
    quantiles_e['real_price'] = real_egarch
    quantiles_e['date'] = cnt

    quantiles_h = df_heston.T.quantile([0.05, 0.25, 0.5, 0.75, 0.95], axis=1).T
    quantiles_h.columns = ['q05', 'q25', 'q50', 'q75', 'q95']
    quantiles_h['real_price'] = real_heston
    quantiles_h['date'] = cnt

    fig, axes = plt.subplots(
        nrows=2,  # Number of rows = number of time series
        ncols=1,  # Single column
        figsize=(8, 6),  # Figure size (width, height)
        sharex=True  # Share x-axis for alignment
    )

    # Fan plot: 90% and 50% intervals
    axes[0].fill_between(quantiles_e['date'], quantiles_e['q05'], quantiles_e['q95'], color='gray', alpha=0.3,
                     label='90% Interval')
    axes[0].fill_between(quantiles_e['date'], quantiles_e['q25'], quantiles_e['q75'], color='gray', alpha=0.5,
                     label='50% Interval')

    # Median forecast
    axes[0].plot(quantiles_e['date'], quantiles_e['q50'], color='black', linestyle='--', label='Forecast Median')

    # Real price
    axes[0].plot(quantiles_e['date'], quantiles_e['real_price'], color='blue', label='Real Price')

    axes[1].fill_between(quantiles_h['date'], quantiles_h['q05'], quantiles_h['q95'], color='gray', alpha=0.3,
                     label='90% Interval')
    axes[1].fill_between(quantiles_h['date'], quantiles_h['q25'], quantiles_h['q75'], color='gray', alpha=0.5,
                     label='50% Interval')

    # Median forecast
    axes[1].plot(quantiles_h['date'], quantiles_h['q50'], color='black', linestyle='--', label='Forecast Median')

    # Real price
    axes[1].plot(quantiles_h['date'], quantiles_h['real_price'], color='blue', label='Real Price')

    #plt.title('Probabilistic Forecast for Gas Futures')
    axes[0].legend(loc="upper right", bbox_to_anchor=(1.0, 1.0))
    axes[1].legend(loc="upper right", bbox_to_anchor=(1.0, 1.0))
    plt.xticks(cnt[::5], rotation='vertical')

    # Plot
    plt.figure(figsize=(14, 6))

    # Fan plot: 90% and 50% intervals
    plt.fill_between(quantiles['date'], quantiles['q05'], quantiles['q95'], color='gray', alpha=0.3,
                     label='90% Interval')
    plt.fill_between(quantiles['date'], quantiles['q25'], quantiles['q75'], color='gray', alpha=0.5,
                     label='50% Interval')

    # Median forecast
    plt.plot(quantiles['date'], quantiles['q50'], color='black', linestyle='--', label='Forecast Median')

    # Real price
    plt.plot(quantiles['date'], quantiles['real_price'], color='blue', label='Real Price')

    plt.ylabel('USD')
    plt.title('Probabilistic Forecast for Oil Futures')
    plt.legend()
    plt.xticks(cnt[::5], rotation='vertical')
    #plt.grid(True)
    #plt.tight_layout()
    plt.show()
    #get_imsample_crps()

    pnl = pandas.read_csv("D:/TU_DORTMUND/Thesis/Data/total_pnl.csv")
    pnl = pnl[['cnt', 'moneyness', 'model', 'hedged_pnl']]
    h = pandas.DataFrame(pnl[pnl['model'] == "HESTON"].groupby('cnt')['hedged_pnl'].mean()).reset_index()
    rslt = dict()
    rslt['cnt'] = []
    rslt['pnl'] = []
    for yy in year:
        for mm in month:
            contract = f"{yy}-{mm}"
            rslt['cnt'].append(contract)
            rslt['pnl'].append(abs(float(h[h['cnt'] == contract]['hedged_pnl'])))

    h = pandas.DataFrame(pnl[pnl['model'] == "HESTON"].groupby('cnt')['hedged_pnl'].mean()).reset_index()
    e = pandas.DataFrame(pnl[pnl['model'] == "EGARCH"].groupby('cnt')['hedged_pnl'].mean()).reset_index()
    rslt = dict()
    rslt['cnt'] = []
    rslt['h_pnl'] = []
    rslt['e_pnl'] = []
    for yy in year:
        for mm in month:
            contract = f"{yy}-{mm}"
            rslt['cnt'].append(contract)
            rslt['h_pnl'].append(abs(float(h[h['cnt'] == contract]['hedged_pnl'])))
            rslt['e_pnl'].append(abs(float(e[e['cnt'] == contract]['hedged_pnl'])))


    rslt = pandas.DataFrame(rslt)
    rslt = rslt[rslt['e_pnl'] < 5]
    plt.plot(rslt.cnt, rslt['h_pnl'])

    plt.plot(rslt.cnt, rslt['e_pnl'])
    plt.xticks(rslt.cnt[::5], rotation='vertical')
    plt.legend(['Heston PnL', 'EGARCH PnL'])
    #t2e_pnl = 1/ 12
    #model = ['EGARCH', 'HESTON']
    #moneyness = [0.95, 1.05]
    #pnl_main()
    peng







    contract = "Jan-23"
    model = "EGARCH"
    moneyness = 1.05
    t2e = 1/12
    egarch_obj = utils.calibration_CNT(contract=contract, t2m=t2e, valdate=None, model=model, delta_hegde=False, estimation_length=3, fixed_result=None)
    target = utils.hegding_pnl(origin_obj=egarch_obj, moneyness= moneyness)
    call_price = egarch_obj.model_obj.call_option_price(step=egarch_obj.model_obj.steps_needed, moneyness=moneyness)
    fixed_rslt = egarch_obj.model_obj.result
    fixed_param = {'mu': fixed_rslt['x'][0],
                   'omega': fixed_rslt['x'][1],
                   'alpha': fixed_rslt['x'][2],
                   'gamma': fixed_rslt['x'][3],
                   'beta': fixed_rslt['x'][4],
                   'nu': fixed_rslt['x'][5],
                   'delta': fixed_rslt['x'][6]}
    rslt_dict = dict()
    start_date = egarch_obj.model_obj.valdate + datetime.timedelta(days = 1)
    rslt_dict['valdate'] = [start_date-datetime.timedelta(days=1)]
    rslt_dict['S'] = [0]
    rslt_dict['delta'] = [0]
    while start_date < egarch_obj.model_obj.expiry_date:
        while start_date.isoweekday() > 5 or start_date in cal.holidays():
            start_date = start_date + datetime.timedelta(days=1)
        try:
            obj = utils.calibration_CNT(contract=contract, t2m=None, valdate= start_date, model=model, delta_hegde=False, estimation_length=3, fixed_result=fixed_param)
            delta_obj = utils.Delta_Hedge(egarch_obj=obj.model_obj, step=obj.model_obj.steps_needed, moneyness=moneyness, model=model)
            if delta_obj.delta > 0:
                rslt_dict['valdate'].append(start_date)
                rslt_dict['S'].append(obj.model_obj.gas_data[-1])
                rslt_dict['delta'].append(delta_obj.delta)
        except:
            pass
        start_date = start_date + datetime.timedelta(days=1)

    hedge_df = pandas.DataFrame(rslt_dict)
    delta_hedge = [(hedge_df['delta'][i] - hedge_df['delta'][i-1])*hedge_df['S'][i]*(1+egarch_obj.model_obj.rate)**(1/12) for i in range(1, hedge_df.shape[0])]
    delta_hedge.insert(0, 0)
    hedge_df['delta_hedge'] = delta_hedge
    strike = egarch_obj.model_obj.gas_data[-1]*moneyness
    execute_price = egarch_obj.model_obj.terminal_price
    portfolio = -numpy.sum(hedge_df['delta_hedge']) + call_price*(1+egarch_obj.model_obj.rate)**(1/12) + hedge_df['delta'].iloc[-1]*execute_price
    payoff = numpy.max([execute_price - strike, 0])
    pnl = portfolio - payoff


    #### heston replication

    model = "HESTON"
    heston_obj = utils.calibration_CNT(contract=contract, t2m=t2e, valdate=None, model=model, delta_hegde=False, estimation_length=3, fixed_result=None)
    target = utils.hegding_pnl(origin_obj=heston_obj, moneyness= moneyness)
    call_price = heston_obj.model_obj.call_option_price(step=heston_obj.model_obj.steps_needed, moneyness=moneyness)
    fixed_param = heston_obj.model_obj._params_dict
    rslt_dict = dict()
    start_date = egarch_obj.model_obj.valdate + datetime.timedelta(days = 1)
    rslt_dict['valdate'] = [start_date-datetime.timedelta(days=1)]
    rslt_dict['S'] = [0]
    rslt_dict['delta'] = [0]
    while start_date < egarch_obj.model_obj.expiry_date:
        while start_date.isoweekday() > 5 or start_date in cal.holidays():
            start_date = start_date + datetime.timedelta(days=1)
        try:
            obj = utils.calibration_CNT(contract=contract, t2m=None, valdate= start_date, model=model, delta_hegde=False, estimation_length=3, fixed_result=fixed_param)
            delta_obj = utils.Delta_Hedge(egarch_obj=obj.model_obj, step=obj.model_obj.steps_needed, moneyness=moneyness, model=model)
            if delta_obj.delta > 0:
                rslt_dict['valdate'].append(start_date)
                rslt_dict['S'].append(obj.model_obj.gas_data[-1])
                rslt_dict['delta'].append(delta_obj.delta)
        except:
            pass
        start_date = start_date + datetime.timedelta(days=1)

    h_hedge_df = pandas.DataFrame(rslt_dict)
    delta_hedge = [(h_hedge_df['delta'][i] - h_hedge_df['delta'][i-1])*h_hedge_df['S'][i]*(1+egarch_obj.model_obj.rate)**(1/12) for i in range(1, h_hedge_df.shape[0])]
    delta_hedge.insert(0, 0)
    h_hedge_df['delta_hedge'] = delta_hedge
    strike = egarch_obj.model_obj.gas_data[-1]*moneyness
    execute_price = egarch_obj.model_obj.terminal_price
    portfolio = -numpy.sum(h_hedge_df['delta_hedge']) + call_price*(1+egarch_obj.model_obj.rate)**(1/12) + h_hedge_df['delta'].iloc[-1]*execute_price
    payoff = numpy.max([execute_price - strike, 0])
    pnl = portfolio - payoff







    contract = "Jan-23"
    model = "EGARCH"
    moneyness = 1
    t2e = 1/12
    egarch_obj = utils.calibration_CNT(contract=contract, t2m=t2e, valdate=None, model=model, delta_hegde=False, estimation_length=3, fixed_result=None)
    target = utils.hegding_pnl(origin_obj=egarch_obj, moneyness= moneyness)
    call_price = egarch_obj.model_obj.call_option_price(step=egarch_obj.model_obj.steps_needed, moneyness=moneyness)
    fixed_rslt = egarch_obj.model_obj.result
    rslt_dict = dict()
    start_date = egarch_obj.model_obj.valdate + datetime.timedelta(days = 1)
    rslt_dict['valdate'] = [start_date-datetime.timedelta(days=1)]
    rslt_dict['S'] = [0]
    rslt_dict['delta'] = [0]
    while start_date < egarch_obj.model_obj.expiry_date:
        while start_date.isoweekday() > 5 or start_date in cal.holidays():
            start_date = start_date + datetime.timedelta(days=1)
        try:
            obj = utils.calibration_CNT(contract=contract, t2m=None, valdate= start_date, model=model, delta_hegde=False, estimation_length=3, fixed_result=fixed_rslt)
            delta_obj = utils.Delta_Hedge(egarch_obj=obj.model_obj, step=obj.model_obj.steps_needed, moneyness=moneyness)
            if delta_obj.delta > 0:
                rslt_dict['valdate'].append(start_date)
                rslt_dict['S'].append(obj.model_obj.gas_data[-1])
                rslt_dict['delta'].append(delta_obj.delta)
        except:
            pass
        start_date = start_date + datetime.timedelta(days=1)

    hedge_df = pandas.DataFrame(rslt_dict)
    delta_hedge = [(hedge_df['delta'][i] - hedge_df['delta'][i-1])*hedge_df['S'][i]*(1+egarch_obj.model_obj.rate)**(1/12) for i in range(1, hedge_df.shape[0])]
    delta_hedge.insert(0, 0)
    hedge_df['delta_hedge'] = delta_hedge
    strike = egarch_obj.model_obj.gas_data[-1]*moneyness
    execute_price = egarch_obj.model_obj.terminal_price
    portfolio = -numpy.sum(hedge_df['delta_hedge']) + call_price*(1+egarch_obj.model_obj.rate)**(1/12) + hedge_df['delta'].iloc[-1]*execute_price
    payoff = numpy.max([execute_price - strike, 0])
    pnl = portfolio - payoff



    ## insample CRPS
    get_imsample_crps()

    ## heston performance
    crps_root_dir = r'D:\TU_DORTMUND\Thesis\Data\crps\HESTON'
    crps_df = pandas.DataFrame()
    for yy in year:
        for mm in month:
            contract = f"{mm}-{yy}"
            df_temp = pandas.read_csv(f"{crps_root_dir}\{contract}.csv", index_col=0, parse_dates=True).iloc[-2:,:].T
            df_temp.columns = ['3M', '1M']
            df_temp.index = [contract]
            crps_df = pandas.concat([crps_df, df_temp], axis = 0)

    crps_df.to_clipboard()

    ## egarch performance

    ecrps_root_dir = r'D:\TU_DORTMUND\Thesis\Data\crps\EGARCH'
    ecrps_df = pandas.DataFrame()
    for yy in year:
        for mm in month:
            contract = f"{mm}-{yy}"
            df_temp = pandas.read_csv(f"{ecrps_root_dir}\{contract}.csv", index_col=0, parse_dates=True).iloc[-2:,:].T
            df_temp.columns = ['3M', '1M']
            df_temp.index = [contract]
            ecrps_df = pandas.concat([ecrps_df, df_temp], axis = 0)

    m3_heston = numpy.array(crps_df['3M'])
    m3_egarch = numpy.array(ecrps_df['3M'])
    m3 = pandas.DataFrame([m3_heston, m3_egarch]).T
    m3.index = crps_df.index
    m3.dropna(inplace=True)
    m3.columns = ['Heston', 'EGARCH']
    m3 = m3[m3['EGARCH'] < 100]
    m3[m3['EGARCH']<100].plot()
    stats31, p31 = wilcoxon(m3['EGARCH'], m3['Heston'], alternative = 'two-sided')
    m3_22 = m3[~m3.index.isin(['Jan-22', 'Feb-22', 'Mar-22', 'Apr-22', 'May-22', 'Jun-22', 'Jul-22', 'Aug-22', 'Sep-22', 'Oct-22', 'Nov-22', 'Dec-22'])]
    m3_22.plot()
    stats32, p32 = wilcoxon(m3_22['Heston'], m3_22['EGARCH'], alternative='two-sided')
    m322 = m3[m3.index.isin(['Jan-22', 'Feb-22', 'Mar-22', 'Apr-22', 'May-22', 'Jun-22', 'Jul-22', 'Aug-22', 'Sep-22', 'Oct-22', 'Nov-22', 'Dec-22'])]

    plt.figure(figsize=(8, 6))
    m3_22.plot()

    m1_heston = numpy.array(crps_df['1M'])
    m1_egarch = numpy.array(ecrps_df['1M'])
    m1 = pandas.DataFrame([m1_heston, m1_egarch]).T
    m1.index = crps_df.index
    m1.dropna(inplace=True)
    m1.columns = ['Heston', 'EGARCH']
    m1 = m1[m1['EGARCH'] < 100]
    m1[m1['EGARCH']<10].plot()
    #stats11, p11 = wilcoxon(m1['Heston'], m1['EGARCH'], alternative = 'two-sided')
    stats11, p11 = wilcoxon(m1['EGARCH'], m1['Heston'], alternative='two-sided')
    m1_22 = m1[~m1.index.isin(['Jan-22', 'Feb-22', 'Mar-22', 'Apr-22', 'May-22', 'Jun-22', 'Jul-22', 'Aug-22', 'Sep-22', 'Oct-22', 'Nov-22', 'Dec-22'])]
    m1_22.plot()
    stats12, p12 = wilcoxon(m1_22['Heston'], m1_22['EGARCH'], alternative='two-sided')
    m122 = m1[m1.index.isin(['Jan-22', 'Feb-22', 'Mar-22', 'Apr-22', 'May-22', 'Jun-22', 'Jul-22', 'Aug-22', 'Sep-22', 'Oct-22', 'Nov-22', 'Dec-22'])]


    fig, axes = plt.subplots(
        nrows=2,  # Number of rows = number of time series
        ncols=1,  # Single column
        figsize=(8, 6),  # Figure size (width, height)
        sharex=False  # Share x-axis for alignment
    )
    time1 = m1[m1['EGARCH']<10].index
    axes[0].plot(time1, m1[m1['EGARCH']<10])
    #axes[0].set_xticks(axes[0].get_xticks()[::5])
    plt.xticks(time1[::5], rotation='vertical')
    time2 = m1_22.index
    axes[1].plot(time2, m1_22)
    #axes[1].set_xticks(axes[1].get_xticks()[::5], rotation = 45)
    #axes[1].set_xticklabels(rotation=40)
    ## heston parameter

    hparam_root_dir = r'D:\TU_DORTMUND\Thesis\Data\params\HESTON'
    hparam_df = pandas.DataFrame()
    for yy in year:
        for mm in month:
            contract = f"{mm}-{yy}"
            df_temp = pandas.DataFrame(pandas.read_csv(f"{hparam_root_dir}\{contract}.csv", index_col=0, parse_dates=True).iloc[1,:]).T
            #df_temp.index = [contract]
            df_temp.index = [contract]
            hparam_df = pandas.concat([hparam_df, df_temp], axis = 0)

    time_series_list = [hparam_df['mu_final'], hparam_df['kappa_final'],
                        hparam_df['theta_final'], hparam_df['volvol_final'],
                        hparam_df['rho_final']]

    # Create subplots
    fig, axes = plt.subplots(
        nrows=len(time_series_list),  # Number of rows = number of time series
        ncols=1,  # Single column
        figsize=(8, 5),  # Figure size (width, height)
        sharex=True  # Share x-axis for alignment
    )

    # Customize and plot each time series
    titles = ["Mu", "Kappa", "Theta", "Vol of Vol", "Rho"]
    colors = ["blue", "green", "red", "black", "grey"]

    time = hparam_df.index

    # Plot each time series with a label
    axes[0].plot(time, time_series_list[0], color=colors[0], label=titles[0])
    plt.xticks(time[::5], rotation='vertical')
    axes[1].plot(time, time_series_list[1], color=colors[1], label=titles[1])
    axes[2].plot(time, time_series_list[2], color=colors[2], label=titles[2])
    axes[3].plot(time, time_series_list[3], color=colors[3], label=titles[3])
    axes[4].plot(time, time_series_list[4], color=colors[4], label=titles[4])
    # Create a single legend for all subplots
    lines_labels = [ax.get_legend_handles_labels() for ax in axes]
    lines, labels = [sum(l, []) for l in zip(*lines_labels)]
    fig.legend(lines, labels, loc="upper right", bbox_to_anchor=(1.0, 1.0))


    # EGARCH params

    ## heston parameter

    eparam_root_dir = r'D:\TU_DORTMUND\Thesis\Data\params\EGARCH'
    eparam_df = pandas.DataFrame()
    for yy in year:
        for mm in month:
            contract = f"{mm}-{yy}"
            df_temp = pandas.DataFrame(pandas.read_csv(f"{eparam_root_dir}\{contract}.csv", index_col=0, parse_dates=True).iloc[1,:]).T
            #df_temp.index = [contract]
            df_temp.index = [contract]
            eparam_df = pandas.concat([eparam_df, df_temp], axis = 0)
    eparam_df.dropna(inplace=True)
    time_series_list = [eparam_df['mu'], eparam_df['omega'],
                        eparam_df['alpha'], eparam_df['gamma'],
                        eparam_df['beta'], eparam_df['nu'],
                        eparam_df['delta']]

    # Create subplots
    fig, axes = plt.subplots(
        nrows=len(time_series_list),  # Number of rows = number of time series
        ncols=1,  # Single column
        figsize=(8, 5),  # Figure size (width, height)
        sharex=True  # Share x-axis for alignment
    )

    # Customize and plot each time series
    titles = ["Mu", "Omega", "Alpha", "Gamma", "Beta", "Nu", "Delta"]
    colors = ["blue", "green", "red", "black", "grey", "orange", "purple"]

    time = eparam_df.index

    # Plot each time series with a label
    axes[0].plot(time, time_series_list[0], color=colors[0], label=titles[0])
    plt.xticks(time[::5], rotation='vertical')
    axes[1].plot(time, time_series_list[1], color=colors[1], label=titles[1])
    axes[2].plot(time, time_series_list[2], color=colors[2], label=titles[2])
    axes[3].plot(time, time_series_list[3], color=colors[3], label=titles[3])
    axes[4].plot(time, time_series_list[4], color=colors[4], label=titles[4])
    axes[5].plot(time, time_series_list[5], color=colors[5], label=titles[5])
    axes[6].plot(time, time_series_list[6], color=colors[6], label=titles[6])
    # Create a single legend for all subplots
    lines_labels = [ax.get_legend_handles_labels() for ax in axes]
    lines, labels = [sum(l, []) for l in zip(*lines_labels)]
    fig.legend(lines, labels, loc="upper right", bbox_to_anchor=(1.0, 1.0))

    #main()
    # contract = 'Apr-22'
    # model = 'HESTON'
    # moneyness = 1.05

    #
    #
    # heston_obj = utils.calibration_CNT(contract=contract, t2m=0.5/12, valdate=None, model=model, delta_hegde=False, estimation_length=3, fixed_result=None)
    # target_1 = utils.hegding_pnl(origin_obj=heston_obj, moneyness=moneyness)
    #
    #
    #
    # egarch_obj = utils.calibration_CNT(contract=contract, t2m=0.5/12, valdate=None, model=model, delta_hegde=False, estimation_length=3, fixed_result=None)
    # target = utils.hegding_pnl(origin_obj=egarch_obj, moneyness= moneyness)
    # call_price = egarch_obj.model_obj.call_option_price(step=egarch_obj.model_obj.steps_needed, moneyness=moneyness)
    # fixed_rslt = egarch_obj.model_obj.result
    # rslt_dict = dict()
    # start_date = egarch_obj.model_obj.valdate + datetime.timedelta(days = 1)
    # rslt_dict['valdate'] = [start_date-datetime.timedelta(days=1)]
    # rslt_dict['S'] = [0]
    # rslt_dict['delta'] = [0]
    # while start_date < egarch_obj.model_obj.expiry_date:
    #     while start_date.isoweekday() > 5 or start_date in cal.holidays():
    #         start_date = start_date + datetime.timedelta(days=1)
    #     try:
    #         obj = utils.calibration_CNT(contract=contract, t2m=None, valdate= start_date, model=model, delta_hegde=False, estimation_length=3, fixed_result=fixed_rslt)
    #         delta_obj = utils.Delta_Hedge(egarch_obj=obj.model_obj, step=obj.model_obj.steps_needed, moneyness=moneyness)
    #         if delta_obj.delta > 0:
    #             rslt_dict['valdate'].append(start_date)
    #             rslt_dict['S'].append(obj.model_obj.gas_data[-1])
    #             rslt_dict['delta'].append(delta_obj.delta)
    #     except:
    #         pass
    #     start_date = start_date + datetime.timedelta(days=1)
    #
    # hedge_df = pandas.DataFrame(rslt_dict)
    # delta_hedge = [(hedge_df['delta'][i] - hedge_df['delta'][i-1])*hedge_df['S'][i]*(1+egarch_obj.model_obj.rate)**(1/12) for i in range(1, hedge_df.shape[0])]
    # delta_hedge.insert(0, 0)
    # hedge_df['delta_hedge'] = delta_hedge
    # strike = egarch_obj.model_obj.gas_data[-1]*moneyness
    # execute_price = egarch_obj.model_obj.terminal_price
    # portfolio = -numpy.sum(hedge_df['delta_hedge']) + call_price*(1+egarch_obj.model_obj.rate)**(1/12) + hedge_df['delta'].iloc[-1]*execute_price
    # payoff = numpy.max([execute_price - strike, 0])
    # pnl = portfolio - payoff



    #main()
    # crps_root_dir = r'D:\TU_DORTMUND\Thesis\Data\crps\HESTON'
    # crps_df = pandas.DataFrame()
    # for yy in year:
    #     for mm in month:
    #         contract = f"{mm}-{yy}"
    #         df_temp = pandas.read_csv(f"{crps_root_dir}\{contract}.csv", index_col=0, parse_dates=True).iloc[-3:,:].T
    #         df_temp.columns = ['6M', '3M', '1M']
    #         df_temp.index = [contract]
    #         crps_df = pandas.concat([crps_df, df_temp], axis = 0)

    # offset = 300
    # burn_in_pos = 5000 - offset
    # param_paths = heston_obj.model_obj.heston_cal.all_params_array_full[offset:, :]
    # fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    # axes[0, 0].plot(param_paths[:, 0])
    # axes[0, 0].axvline(x=burn_in_pos, color="red", linewidth=3)
    # axes[0, 0].set_xlabel("$\\mu$")
    # axes[0, 1].plot(param_paths[:, 1])
    # axes[0, 1].axvline(x=burn_in_pos, color="red", linewidth=3)
    # axes[0, 1].set_xlabel("$\\kappa$")
    # axes[1, 0].plot(param_paths[:, 2])
    # axes[1, 0].axvline(x=burn_in_pos, color="red", linewidth=3)
    # axes[1, 0].set_xlabel("$\\theta$")
    # axes[1, 1].plot(param_paths[:, 3])
    # axes[1, 1].axvline(x=burn_in_pos, color="red", linewidth=3)
    # axes[1, 1].set_xlabel("$\\psi$")
    # axes[2, 0].plot(param_paths[:, 4])
    # axes[2, 0].axvline(x=burn_in_pos, color="red", linewidth=3)
    # axes[2, 0].set_xlabel("$\\omega$")
    # axes[2, 1].remove()
    # plt.suptitle('Posterior dynamics of parameters in Heston model (burn-in cutoff in red)')
    # plt.subplots_adjust(wspace=None, hspace=0.3)
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])




