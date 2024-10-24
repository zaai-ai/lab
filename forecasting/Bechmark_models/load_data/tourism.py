import pandas as pd
from datasetsforecast.hierarchical import HierarchicalData


class TourismDataset:
    def load_data(self, group):
        ds, *_ = HierarchicalData.load(directory='data', group="TourismLarge")
        ds["ds"] = pd.to_datetime(ds["ds"])
        ds['month'] = ds["ds"].dt.month
        return ds