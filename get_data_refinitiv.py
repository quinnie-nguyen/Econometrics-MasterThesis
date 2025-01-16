import eikon as ek
import pandas as pd
import datetime


ek.set_app_key("56baec8c0a6547ae98fca31207aaad54f685e9be")

##########################################################

def get_hist(tckr, start_date, end_date, granularity):
    try:
        historical_data = ek.get_timeseries(
            tckr,
            start_date=start_date,  # Define your start date
            end_date=end_date,  # Define your end date
            interval=granularity  # Choose frequency: 'daily', 'weekly', 'monthly', 'minute' etc.
        )
        print(historical_data)
        return historical_data
    except Exception as e:
        print(f"Error fetching data: {e}")

if __name__ == "__main__":

    ### Once a future has expired, tckr = NG+monthcode+last_digit_year+^+decade_contract(1or2)
    month_dict = {'Jan': 'F',
                  'Feb': 'G',
                  'Mar': 'H',
                  'Apr': 'J',
                  'May': 'K',
                  'Jun': 'M',
                  'Jul': 'N',
                  'Aug': 'Q',
                  'Sep': 'U',
                  'Oct': 'V',
                  'Nov': 'X',
                  'Dec': 'Z'}
    year = [16, 17, 18, 19, 20, 21, 22, 23, 24]
    com = 'NG'
    for yy in year:
        for m_name, m_digit in zip(month_dict.keys(), month_dict.values()):
            if yy == 16 or yy == 17:
                tckr = com+m_digit+str(yy)[1]+'^'+str(yy)[0]
            else:
                tckr = com + m_digit + str(yy) + '^' + str(yy)[0]
            print(m_name)
            print(tckr)
            df = get_hist(tckr, datetime.datetime(2015, 1, 1), datetime.datetime(2025, 1, 1), 'daily')
            df.to_csv(f"D:\TU_DORTMUND\Thesis\Data\price\{m_name}-{str(yy)}.csv")

