import pandas as pd
import numpy as np
import io
import os
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split
import traceback
import re
import sys
import time
import json
import logging
from time import strftime, gmtime
import argparse
import csv


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

BASE_PATH = os.path.join("/", "opt", "ml")
PROCESSING_PATH = os.path.join(BASE_PATH, "processing")
PROCESSING_PATH_INPUT = os.path.join(PROCESSING_PATH, "input")
PROCESSING_PATH_OUTPUT = os.path.join(PROCESSING_PATH, "output")


def extract_data(file_path, percentage=100):
    try:
        files = [f for f in listdir(file_path) if isfile(join(file_path, f)) and f.endswith(".csv")]
        LOGGER.info("{}".format(files))

        frames = []

        for file in files:
            df = pd.read_csv(
                os.path.join(file_path, file),
                sep=",",
                quotechar='"',
                quoting=csv.QUOTE_ALL,
                escapechar='\\',
                encoding='utf-8',
                error_bad_lines=False
            )

            df = df.head(int(len(df) * (percentage / 100)))

            frames.append(df)

        df = pd.concat(frames)

        return df
    except Exception as e:
        stacktrace = traceback.format_exc()
        LOGGER.error("{}".format(stacktrace))

        raise e

def load_data(df, file_path, file_name, header=False):
    try:
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        path = os.path.join(file_path, file_name + ".csv")

        LOGGER.info("Saving file in {}".format(path))

        df.to_csv(
            path,
            index=False,
            header=header,
            encoding="utf-8",
        )
    except Exception as e:
        stacktrace = traceback.format_exc()
        LOGGER.error("{}".format(stacktrace))

        raise e



def transform_data(df):
    try:

        LOGGER.info("Original count: {}".format(len(df.index)))

        df = df.drop("Phone", axis=1)
        df["Area Code"] = df["Area Code"].astype(object)

        df = df.drop(["Day Charge", "Eve Charge", "Night Charge", "Intl Charge"], axis=1)

        df_dummies = pd.get_dummies(df)
        df_dummies = pd.concat(
        [df_dummies["Churn?_True."], df_dummies.drop(["Churn?_False.", "Churn?_True."], axis=1)], axis=1
        )
        df_dummies = df_dummies.astype(float)

        return df_dummies
    except Exception as e:
        stacktrace = traceback.format_exc()
        LOGGER.error("{}".format(stacktrace))

        raise e

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    LOGGER.info("Arguments: {}".format(args))

    df = extract_data(PROCESSING_PATH_INPUT, 100)

    df = transform_data(df)

    data_train, data_validation = train_test_split(df, test_size=0.2)

    load_data(data_train, os.path.join(PROCESSING_PATH_OUTPUT, "train"), "train")
    load_data(data_validation, os.path.join(PROCESSING_PATH_OUTPUT, "validation"), "validation")

    # Creating test dataset for batch inference
    # data_validation = data_validation.drop('labels', axis=1)
    load_data(data_validation, os.path.join(PROCESSING_PATH_OUTPUT, "inference"), "data", False)
