import pandas as pd
import numpy as np
from datasetsforecast.hierarchical import HierarchicalData


class Wiki2Dataset:
    def load_data(self, group):
        ds, *_ = HierarchicalData.load(directory='data', group="Wiki2")
        ds["ds"] = pd.to_datetime(ds["ds"])
        if group == "Weekly":
            df_dates = ds[["ds"]].drop_duplicates()
            df_dates['week'] = df_dates["ds"].dt.isocalendar().week
            df_dates['year'] = df_dates["ds"].dt.year
            df_dates = df_dates.merge(
                df_dates[df_dates['ds'].dt.weekday == 0].rename(columns={'ds': 'week_start_date'}), on=['week', 'year'],
                how='left')

            ds = ds.merge(df_dates, on='ds', how='left').drop(columns=['week', 'year'])
            ds = ds.groupby(['unique_id', 'week_start_date']).agg({'y': 'sum'}).reset_index().rename(
                columns={'week_start_date': 'ds'})

            ds['week'] = ds["ds"].dt.isocalendar().week
            ds['month'] = ds["ds"].dt.month

        elif group == "Monthly":
            df_dates = ds[["ds"]].drop_duplicates()
            df_dates['first_day_of_month'] = df_dates['ds'].dt.to_period('M').dt.to_timestamp()
            df_dates['week_start_date'] = df_dates['first_day_of_month'] + pd.offsets.Week(weekday=0)
            ds = ds.merge(df_dates[['ds', 'week_start_date']], on='ds', how='left')

            ds = ds.groupby(['unique_id', 'week_start_date']).agg({'y': 'sum'}).reset_index().rename(
                columns={'week_start_date': 'ds'})

            ds['month'] = ds["ds"].dt.month

        else:
            ds['week'] = ds["ds"].dt.isocalendar().week
            ds['month'] = ds["ds"].dt.month
            ds['weekday'] = ds["ds"].dt.weekday

        def safe_split(x, index):
            try:
                return x.split('_')[index]
            except IndexError:
                return 0

        ds['static_1'] = ds['unique_id'].apply(lambda x: safe_split(x, 0))
        ds['static_2'] = ds['unique_id'].apply(lambda x: safe_split(x, 1))
        ds['static_3'] = ds['unique_id'].apply(lambda x: safe_split(x, 2))
        return ds
