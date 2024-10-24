from datasetsforecast.m5 import M5
import pandas as pd


class M5Dataset:
    def load_data(self, group):
        ds, da, dc = M5.load(directory='data')
        ds['ds'] = pd.to_datetime(ds['ds'])

        if group == 'Weekly':
            df_dates = ds[["ds"]].drop_duplicates()
            df_dates['week'] = df_dates["ds"].dt.isocalendar().week
            df_dates['year'] = df_dates["ds"].dt.year
            df_dates = df_dates.merge(
                df_dates[df_dates['ds'].dt.weekday == 0].rename(columns={'ds': 'week_start_date'}), on=['week', 'year'],
                how='left')

            ds = ds.merge(df_dates, on='ds', how='left').drop(columns=['week', 'year'])
            ds = ds.groupby(['unique_id', 'week_start_date']).agg({'y': 'sum'}).reset_index().rename(
                columns={'week_start_date': 'ds'})

            ds = ds.merge(dc, on='unique_id', how='left')
            ds['week'] = ds["ds"].dt.isocalendar().week
            ds['month'] = ds["ds"].dt.month

        elif group == "Monthly":
            df_dates = ds[["ds"]].drop_duplicates()
            df_dates['first_day_of_month'] = df_dates['ds'].dt.to_period('M').dt.to_timestamp()
            df_dates['week_start_date'] = df_dates['first_day_of_month'] + pd.offsets.Week(weekday=0)
            ds = ds.merge(df_dates[['ds', 'week_start_date']], on='ds', how='left')

            ds = ds.groupby(['unique_id', 'week_start_date']).agg({'y': 'sum'}).reset_index().rename(
                columns={'week_start_date': 'ds'})

            ds = ds.merge(dc, on='unique_id', how='left')
            ds['month'] = ds["ds"].dt.month

            grouped_ds = ds.groupby('unique_id').apply(lambda x: (x['y'] == 0).mean())
            valid_ids = grouped_ds[grouped_ds <= 0.5].index
            new_ds = ds[ds['unique_id'].isin(valid_ids)]
            ds = new_ds

        else:
            ds = ds.merge(dc, on='unique_id', how='left')
            ds = ds.merge(da, on=['unique_id', 'ds'], how='left')
            ds['week'] = ds["ds"].dt.isocalendar().week
            ds['month'] = ds["ds"].dt.month
            ds['weekday'] = ds["ds"].dt.weekday

        return ds
