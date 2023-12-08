import datetime
import os
import pandas as pd

from joblib import Parallel, delayed


def load_subject(path, ibi_threshold=0.5, data_source="garmin", window_length=60):
    try:
        subject_id = int(path.split("/")[-1].split("-")[-2])
    except:
        subject_id = int(path.split("/")[-1].split("_")[-1])

    path += f"/output/{data_source}/"

    ##############################
    # IBI
    ##############################
    ibi_file = f"{path}/hrv_{window_length:d}_{ibi_threshold:.2f}.csv"
    # ibi_file = path + ('../holter/hrv_60_%.2f.csv' % ibi_threshold)

    try:
        hrv_data = pd.read_csv(ibi_file, index_col=0, parse_dates=True)
        hrv_data.index = hrv_data.index.tz_localize("Europe/Zurich")
        # hrv_data = hrv_data.add_prefix('hrv_')
        hrv_data.dropna(how="all", inplace=True)
        hrv_data["subject_id"] = subject_id

        # hrv_data = round_df(hrv_data, 2, ['id'])

        data = hrv_data
    except Exception as e:
        raise Exception("Error in IBI loading", e)

    ##############################
    # ACC
    ##############################
    acc_file = f"{path}/acc_simple{window_length:d}.csv"

    # check if ACC file exists
    if os.path.exists(acc_file):
        try:
            acc_data = pd.read_csv(acc_file, index_col=0, parse_dates=True)
            acc_data.index = acc_data.index.tz_localize("Europe/Zurich")
            acc_data.sort_index(inplace=True)

            # acc_data = round_df(acc_data, 2)

            # try to fix the timing error in ACC since timestamps seem to diverge for study 1
            if subject_id < 300:
                acc_data.index = pd.date_range(
                    start=acc_data.iloc[0].name,
                    end=hrv_data.iloc[-1].name,
                    periods=len(acc_data),
                )
            data = pd.merge_asof(
                data,
                acc_data.add_prefix("acc_"),
                left_index=True,
                right_index=True,
                direction="nearest",
                tolerance=datetime.timedelta(milliseconds=1000),
            )

            data = data[~data.filter(regex="^acc").isna().any(axis=1)]

        except Exception as e:
            raise Exception("Error in ACC loading", e)

    ##############################
    # EDA
    ##############################
    eda_file = f"{path}../empatica/eda_{window_length:d}.csv"

    # check if EDA file exists
    if os.path.exists(eda_file):
        try:
            eda_data = pd.read_csv(eda_file, index_col=0, parse_dates=True)
            eda_data.index = eda_data.index.tz_convert("Europe/Zurich")
            eda_data.sort_index(inplace=True)

            data = data.join(eda_data.add_prefix("eda_"))
        except Exception as e:
            raise Exception("Error in EDA loading", e)

    data = data.loc[~data.index.duplicated(keep="first")]

    ##############################
    # LOAD TIMESTAMPS/SCENARIOS
    ##############################
    timestamps = pd.read_csv(
        path + "/../../timestamps_engine.csv",
        header=0,
        names=["from", "to"],
        usecols=[1, 2],
        parse_dates=["from", "to"],
    )
    timestamps["env"] = pd.read_csv(path + "/../../scenarios.csv", header=None)
    timestamps = timestamps[-12:].reset_index(drop=True)

    for idx, timestamp_row in timestamps.iterrows():
        data.loc[
            (
                data.index
                >= timestamp_row["from"] + datetime.timedelta(seconds=window_length)
            )
            & (data.index <= timestamp_row["to"]),
            "env",
        ] = timestamp_row["env"].capitalize()
        data.loc[
            (
                data.index
                >= timestamp_row["from"] + datetime.timedelta(seconds=window_length)
            )
            & (data.index <= timestamp_row["to"]),
            "phase",
        ] = 1 + (idx // 3)

    data.dropna(subset=["phase"], inplace=True)
    data["phase"] = data["phase"].astype(int)

    ##############################
    # BG DATA
    ##############################
    bg_file = path + "../../bloodglucose/bg.csv"
    if not os.path.exists(bg_file):
        raise Exception("Cannot continue, missing BG file")

    try:
        bg_data = pd.read_csv(
            bg_file,
            index_col=0,
            usecols=["timestamp", "bg_biosen", "bg_dexcom"],
            parse_dates=True,
        )
        # bg_data.dropna(subset=['bg_biosen'], inplace=True)
        bg_data.index = bg_data.index.tz_convert("Europe/Zurich")
        bg_data.sort_index(inplace=True)

        interpolation = "ffill"  # 'ffill' or 'time'
        assert interpolation in ["ffill", "time"]
        bg_data = bg_data.asfreq("1S").interpolate(method=interpolation)

        data["bg"] = bg_data["bg_biosen"]
        data["cgm"] = bg_data["bg_dexcom"]
    except Exception as e:
        raise Exception("Error in BG loading", e)

    return data


def read_data(subjects, ibi_threshold=0.5, data_source="garmin", window_length=60):
    data = []

    with Parallel(n_jobs=min(16, len(subjects)), verbose=10) as parallel:
        data = parallel(
            delayed(load_subject)(subject, ibi_threshold, data_source, window_length)
            for subject in subjects
        )

    data = pd.concat(data)
    data.sort_index(inplace=True)
    return data
