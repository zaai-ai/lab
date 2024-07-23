from dateutil.relativedelta import relativedelta
from chronos import ChronosPipeline
from darts import TimeSeries
import concurrent.futures
from tqdm import tqdm
import pandas as pd
import warnings
import logging
import torch
import utils
import os

# Set the environment variable
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TIME_COL = "Date"
TARGET = "visits"
STATIC_COV = ["static_1", "static_2", "static_3", "static_4"]
FREQ = "MS"

################################################## LOADING DATA ##################################################

DATASET = "hf://datasets/zaai-ai/time_series_datasets/data.csv"
df = pd.read_csv(DATASET).drop(columns=["Unnamed: 0"])
df[TIME_COL] = pd.to_datetime(df[TIME_COL])

################################################## FORECASTING ##################################################

FORECAST_HORIZON = 1

START_PREDICTIONS = df[TIME_COL].quantile(0.25)

# Calculate to save how many iterations we need to forecast
difference = relativedelta(df[TIME_COL].max(), START_PREDICTIONS)
NUM_ITERATIONS = int(difference.years * 12 + difference.months)

ARCHITECTURE = ("amazon/chronos-t5-large", "cuda")

# Load the Chronos pipeline
pipeline = ChronosPipeline.from_pretrained(ARCHITECTURE[0], device_map=ARCHITECTURE[1], torch_dtype=torch.bfloat16)

all_residuals_list = []
progress_bar = tqdm(total=NUM_ITERATIONS, desc='Progress', position=0)


def process_iteration(i):
    try:
        start_pred = START_PREDICTIONS + pd.DateOffset(months=i)
        test_end = START_PREDICTIONS + pd.DateOffset(months=FORECAST_HORIZON + i)

        train = df[(df[TIME_COL] <= start_pred)]
        test = df[(df[TIME_COL] > start_pred) & (df[TIME_COL] <= test_end)]

        # Read train and test datasets and transform train dataset
        train_darts = TimeSeries.from_group_dataframe(
            df=train,
            group_cols=STATIC_COV,
            time_col=TIME_COL,
            value_cols=TARGET,
            freq=FREQ,
            fill_missing_dates=True,
            fillna_value=0)

        forecast = []
        for ts in train_darts:
            # Forecast
            lower, mid, upper = utils.chronos_forecast(pipeline, ts.pd_dataframe().reset_index(), FORECAST_HORIZON,
                                                       TARGET)
            unique_id = "".join(str(ts.static_covariates[key].item()) for key in ts.static_covariates)
            forecast.append(utils.convert_forecast_to_pandas([lower, mid, upper],
                                                             test[test['unique_id'] == unique_id]))
        # Convert list to data frames
        forecast = pd.concat(forecast)
        residuals = test.drop(columns=[TARGET])

        residuals["residuals"] = test[TARGET] - forecast["forecast"]
        last_column = residuals.pop(residuals.columns[-1])
        residuals.insert(2, last_column.name, last_column)

        progress_bar.update(1)
        return residuals
    except Exception as e:
        logger.error(f"Error processing iteration {i}: {e}")
        return None


# Use ThreadPoolExecutor for parallel processing
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_iteration, i) for i in range(NUM_ITERATIONS)]
    for future in concurrent.futures.as_completed(futures):
        result = future.result()
        if result is not None:
            all_residuals_list.append(result)

progress_bar.close()

# Concatenate all residuals into a single DataFrame
all_residuals = pd.concat(all_residuals_list, ignore_index=True).sort_values(by=['unique_id', 'Date'])
all_residuals.to_csv('data/residuals.csv', index=False)
