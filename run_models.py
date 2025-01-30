import pandas
import datetime
import utils
import warnings
import numpy
from pandas.tseries.holiday import USFederalHolidayCalendar
cal = USFederalHolidayCalendar()

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


if __name__ == '__main__':
    crps_root_dir = r'D:\TU_DORTMUND\Thesis\Data\crps\HESTON'
    crps_df = pandas.DataFrame()
    for yy in year:
        for mm in month:
            contract = f"{mm}-{yy}"
            df_temp = pandas.read_csv(f"{crps_root_dir}\{contract}.csv", index_col=0, parse_dates=True).iloc[-2:,:].T
            df_temp.columns = ['3M', '1M']
            df_temp.index = [contract]
            crps_df = pandas.concat([crps_df, df_temp], axis = 0)
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




