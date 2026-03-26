from dataclasses import dataclass

@dataclass
class Paths:
    concrete_xls: str = r"C:\Users\12506\Desktop\401\dataset\Concrete_Data.xls"
    bike_hour_csv: str = r"C:\Users\12506\Desktop\401\dataset\hour.csv"
    nyc_train_csv: str = r"C:\Users\12506\Desktop\401\dataset\nyc taxi trip duration_train.csv"
    nyc_test_csv: str = r"C:\Users\12506\Desktop\401\dataset\nyc taxi trip durationtest.csv"

@dataclass
class TrainCfg:
    seed: int = 42
    test_size: float = 0.2
    hour_fullgp_max_train: int = 3000  # hour 的 fullgp 训练最多用多少条（只影响 fullgp）

    # NYC Taxi is huge: sample rows for training (change as needed)
    nyc_max_rows: int = 500_000  # keep training feasible on a laptop

    # Columns used from NYC train file (must include 'trip_duration')
    nyc_usecols = (
        "id", "vendor_id", "pickup_datetime", "dropoff_datetime",
        "passenger_count", "pickup_longitude", "pickup_latitude",
        "dropoff_longitude", "dropoff_latitude", "store_and_fwd_flag",
        "trip_duration",
    )

    # SVGP + GPz basis count
    inducing_m: int = 128

    # Training hyperparameters
    iters: int = 3000
    lr: float = 0.01

    # SVGP minibatch (used for all datasets, especially NYC)
    batch_size: int = 256
