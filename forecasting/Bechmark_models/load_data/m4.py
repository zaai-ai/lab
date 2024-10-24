from datasetsforecast.m4 import M4
import pandas as pd
from pandas.tseries.offsets import DateOffset


class M4Dataset:
    def load_data(self, group):
        ds, _, dc = M4.load(directory='data', group=group)
        start_date = pd.to_datetime('2000-01-01')

        def add_months(start_date, months):
            return start_date + pd.DateOffset(months=months)

        if group == "Monthly":
            # ds['ds'] = ds['ds'].apply(lambda x: add_months(start_date, int(x - 1)))
            ds = ds.merge(dc, on='unique_id', how='left')

            base_date = pd.Timestamp('2000-01-01')
            ds['ds'] = ds['ds'].apply(lambda x: base_date + DateOffset(months=x - 1))

            ds['ds'] = pd.to_datetime(ds['ds'])
            ds['month'] = ds["ds"].dt.month

        elif group == "Daily":
            ds['ds'] = start_date + pd.to_timedelta(ds['ds'] - 1, unit=group[0])
            ds = ds.merge(dc, on='unique_id', how='left')
            ds['ds'] = pd.to_datetime(ds['ds'])
            ds['week'] = ds["ds"].dt.isocalendar().week
            ds['month'] = ds["ds"].dt.month
            ds['weekday'] = ds["ds"].dt.weekday

        else:
            ds['ds'] = start_date + pd.to_timedelta(ds['ds'], unit=group[0])
            ds = ds.merge(dc, on='unique_id', how='left')
            ds['ds'] = pd.to_datetime(ds['ds'])
            ds['month'] = ds["ds"].dt.month
            ds['week'] = ds["ds"].dt.isocalendar().week

        return ds
