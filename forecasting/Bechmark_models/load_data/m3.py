from datasetsforecast.m3 import M3
import pandas as pd


class M3Dataset:
    # Allowed groups:'Yearly', 'Quarterly', 'Monthly', 'Other'
    def load_data(self, group):
        if group == 'Weekly':
            ds, *_ = M3.load(directory='data', group='Other')
            ds['ds'] = pd.to_datetime(ds['ds'])

            df_dates = ds[["ds"]].drop_duplicates()
            df_dates['week'] = df_dates["ds"].dt.isocalendar().week
            df_dates['year'] = df_dates["ds"].dt.year
            df_dates = df_dates.merge(
                df_dates[df_dates['ds'].dt.weekday == 0].rename(columns={'ds': 'week_start_date'}), on=['week', 'year'],
                how='left')

            ds = ds.merge(df_dates, on='ds', how='left').drop(columns=['year'])
            ds = ds.groupby(['unique_id', 'week_start_date', 'week']).agg({'y': 'sum'}).reset_index().rename(
                columns={'week_start_date': 'ds'})
            ds['month'] = ds["ds"].dt.month
        elif group == 'Daily':
            ds, *_ = M3.load(directory='data', group='Other')
            ds['ds'] = pd.to_datetime(ds['ds'])
            ds['week'] = ds["ds"].dt.isocalendar().week
            ds['month'] = ds["ds"].dt.month
            ds['weekday'] = ds["ds"].dt.weekday

        else:
            ds, *_ = M3.load(directory='data', group=group)
            ds['ds'] = pd.to_datetime(ds['ds'])
            ds['month'] = ds["ds"].dt.month

        return ds
