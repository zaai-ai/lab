import pandas as pd
from datasetsforecast.hierarchical import HierarchicalData


class LabourDataset:
    def load_data(self, group):
        ds, *_ = HierarchicalData.load(directory='data', group="Labour")
        ds["ds"] = pd.to_datetime(ds["ds"])
        ds['month'] = ds["ds"].dt.month
        return ds