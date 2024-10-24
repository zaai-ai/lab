from load_data.m3 import M3Dataset
from load_data.m4 import M4Dataset
from load_data.m5 import M5Dataset
from load_data.labour import LabourDataset
from load_data.traffic import TrafficDataset
from load_data.wiki2 import Wiki2Dataset
from load_data.tourism import TourismDataset

DATASETS = {
    "M3": M3Dataset,
    "M4": M4Dataset,
    "M5": M5Dataset,
    "Labour": LabourDataset,
    "Traffic": TrafficDataset,
    "Wiki2": Wiki2Dataset,
    "Tourism": TourismDataset,
}

DATASETS_FREQ = {
    #"M3": ["Monthly"],  # does not have static features, poucos dados no Daily e Weekly
    #"Labour": ["Monthly"], # only monthly
    #"Traffic": ["Daily", "Weekly"], # "tiny dataset" no monthly
    #'Wiki2': ['Daily', 'Weekly'], # "tiny dataset" no monthly
    #"Tourism": ["Monthly"], # only monthly
    "M4": ["Weekly"],  # remove Monthly? has static features
    "M5": ["Weekly", "Monthly"], # takes too long to run on daily
}
