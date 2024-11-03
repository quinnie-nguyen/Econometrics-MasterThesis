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
    except Exception as e:
        print(f"Error fetching data: {e}")

