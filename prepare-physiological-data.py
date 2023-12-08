import glob
import os
import sys
from zipfile import ZipFile

import pandas as pd

from typing import List

import numpy as np

import flirt.acc
import flirt.eda
import flirt.hrv
import flirt.stats
import flirt.reader.garmin
import flirt.reader.empatica
import flirt.eda.preprocessing
from joblib import Parallel, delayed


def try_unzip_empatica_file_for_subject(path: str):
    for zip_file in glob.glob(path + "/empatica/*.zip"):
        print("Unzipping Empatica file", zip_file)
        with ZipFile(zip_file, "r") as zipObj:
            # Extract all the contents of zip file in current directory
            zipObj.extractall(
                path=zip_file[:-4]
            )  # target dir is new folder with name of zip file


def try_unzip_garmin_file_for_subject(path: str):
    for zip_file in glob.glob(path + "/garmin/*.zip"):
        print("Unzipping Garmin file", zip_file)
        with ZipFile(zip_file, "r") as zipObj:
            # Extract all the contents of zip file in current directory
            zipObj.extractall(
                path=path + "/garmin/input"
            )  # target dir is new folder with name of zip file


def process_empatica_ibi_file(path: str, epoch_width: int = 180, thresholds=[0.0]):
    print("Processing Empatica IBI file")
    target_dir = path + "/output/empatica"

    data = []
    for file_name in glob.glob(path + "/empatica/*/IBI.csv"):
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        empatica_ibi = flirt.reader.empatica.read_ibi_file_into_df(file_name)
        data.append(empatica_ibi)

    if len(data) > 0:
        empatica_ibi = pd.concat(data)
        empatica_ibi.sort_index(inplace=True)

        for threshold in thresholds:
            try:
                empatica_hrv = flirt.hrv.get_hrv_features(
                    empatica_ibi["ibi"],
                    epoch_width,
                    domains=["td", "fd", "stat"],
                    threshold=threshold,
                )
                empatica_hrv.to_csv(
                    target_dir + ("/hrv_%d_%.2f.csv" % (epoch_width, threshold))
                )
            except Exception as e:
                print(
                    "Unexpected error in Empatica IBI calculation for threshold %.2f at path %s: %s"
                    % (threshold, path, e)
                )
                pass


def process_empatica_eda_file(path: str, epoch_width: int = 300):
    print("Processing Empatica EDA file")

    data = []
    for file_name in glob.glob(path + "/empatica/*/EDA.csv"):
        empatica_eda = flirt.reader.empatica.read_eda_file_into_df(file_name)
        data.append(empatica_eda)

    if len(data) > 0:
        empatica_eda = pd.concat(data)
        empatica_eda.sort_index(inplace=True)

        eda = flirt.eda.get_eda_features(
            empatica_eda["eda"],
            window_length=epoch_width,
        )
        # preprocessor=flirt.eda.preprocessing.LowPassFilter(cutoff=0.05))
        target_dir = path + "/output/empatica"
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        eda.to_csv(target_dir + ("/eda_%d.csv" % epoch_width))


def process_empatica_acc_file(path: str, epoch_width: int = 60):
    print("Processing Empatica ACC file")

    data = []
    for file_name in glob.glob(path + "/empatica/*/ACC.csv"):
        empatica_acc = flirt.reader.empatica.read_acc_file_into_df(file_name)
        data.append(empatica_acc)

    if len(data) > 0:
        empatica_acc = pd.concat(data)
        empatica_acc.sort_index(inplace=True)

        acc = flirt.acc.get_acc_features(empatica_acc, epoch_width)
        target_dir = path + "/output/empatica"
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        acc.to_csv(target_dir + ("/acc_%d.csv" % epoch_width))


def process_garmin_ibi_file(path: str, epoch_width: int = 180, thresholds=[0.0]):
    print("Processing Garmin IBI file")
    file_name = glob.glob(path + "/garmin/input/data.csv")
    target_dir = path + "/output/garmin"

    if len(file_name) != 0:
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        garmin_ibi = flirt.reader.garmin.read_data_file_into_df(
            file_name[0], "hrv"
        )  # HEART_RATE_VARIABILITY')
        if garmin_ibi.empty:
            garmin_ibi = flirt.reader.garmin.read_data_file_into_df(
                file_name[0], "HEART_RATE_VARIABILITY"
            )

        for threshold in thresholds:
            try:
                garmin_hrv = flirt.hrv.get_hrv_features(
                    garmin_ibi.iloc[:, 0],
                    epoch_width,
                    domains=["td", "fd", "stat", "nl"],
                    threshold=threshold,
                )
                garmin_hrv.to_csv(
                    target_dir + ("/hrv_%d_%.2f.csv" % (epoch_width, threshold))
                )
            except Exception as e:
                print(
                    "Unexpected error in Garmin IBI calculation for threshold %.2f at path %s: %s"
                    % (threshold, path, e)
                )
                import traceback

                traceback.print_exc()
                pass


def process_garmin_acc_file(path: str, epoch_width: int = 60, diff: bool = False):
    print("Processing Garmin ACC file")
    file_name = glob.glob(path + "/garmin/input/acc.csv")
    if len(file_name) != 0:
        garmin_acc = flirt.reader.garmin.read_acc_file_into_df(file_name[0])

        # diff data twice to get acc instead of orientation
        if diff:
            garmin_acc = garmin_acc.diff().diff().dropna()

        garmin_acc = flirt.acc.get_acc_features(
            garmin_acc,
            window_length=epoch_width,
            window_step_size=1,
            data_frequency=25,
            num_cores=1,
        )
        target_dir = path + "/output/garmin"
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        if diff:
            file_name = "/acc_diff%d.csv" % epoch_width
        else:
            file_name = "/acc_simple%d.csv" % epoch_width

        garmin_acc.to_csv(target_dir + file_name)


def process_garmin_acc_file_diffs(path: str, epoch_width: int = 60):
    print("Processing Garmin ACC file")
    file_name = glob.glob(path + "/garmin/input/acc.csv")
    if len(file_name) != 0:
        garmin_acc = flirt.reader.garmin.read_acc_file_into_df(file_name[0])
        garmin_acc.columns = ["x", "y", "z"]

        data = {}
        data[""] = garmin_acc
        data["abs_"] = data[""].abs()
        data["d1_"] = data[""].diff()
        data["d1abs_"] = data["d1_"].abs()
        data["d2_"] = data["d1_"].diff()
        data["d2abs_"] = data["d2_"].abs()

        # for key in data.keys():
        data[""]["l2"] = np.linalg.norm(data[""].to_numpy(), axis=1)
        data["d1_"]["l2"] = np.linalg.norm(data["d1_"].to_numpy(), axis=1)
        data["d2_"]["l2"] = np.linalg.norm(data["d2_"].to_numpy(), axis=1)

        for key in data.keys():
            data[key] = data[key].add_prefix(key)

        garmin_acc = pd.concat(data.values(), axis=1, copy=False)
        garmin_acc.dropna(inplace=True)

        garmin_acc = flirt.stats.get_stat_features(garmin_acc, epoch_width)
        target_dir = path + "/output/garmin"
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        file_name = "/acc_%d.csv" % epoch_width

        garmin_acc.to_csv(target_dir + file_name)


def run_failsafe(fun, *args):
    try:
        fun(*args)
    except Exception as e:
        print("Unexpected error:", e)
        pass


if __name__ == "__main__":
    DATA_FOLDER = "/headwind/lab-study/"

    print("DATA_FOLDER:", DATA_FOLDER)

    def run_subject(subject: str):
        print("Processing subject", subject.split("/")[-1])

        # check if empatica file is extracted
        if len(glob.glob(subject + "/empatica/*/IBI.csv")) == 0:
            try_unzip_empatica_file_for_subject(subject)

        # check if garmin file is extracted
        if len(glob.glob(subject + "/garmin/input/data.csv")) == 0:
            try_unzip_garmin_file_for_subject(subject)

        window_lengths = [30, 60, 120, 180]
        for window_length in window_lengths:
            try:
                run_failsafe(process_empatica_eda_file, subject, window_length)
                run_failsafe(process_garmin_ibi_file, subject, window_length, [0.5])
                run_failsafe(process_garmin_acc_file, subject, window_length)
            except:
                print("Unexpected error:", sys.exc_info()[0])
                pass

    subjects = sorted(glob.glob(DATA_FOLDER + "/*-V3"))
    subjects.extend(sorted(glob.glob(DATA_FOLDER + "/*_3[01][0-9]")))
    with Parallel(n_jobs=min(40, len(subjects)), max_nbytes=None) as parallel:
        parallel(delayed(run_subject)(x) for x in subjects)
