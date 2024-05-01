from argparse import ArgumentParser, Namespace
import csv
import glob
import json
import logging
import os
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from time import gmtime, strftime
import traceback
import boto3
import re
from sagemaker import get_execution_role
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import sys
import time
import json
from time import strftime, gmtime
from sagemaker.inputs import TrainingInput
from sagemaker.serializers import CSVSerializer
from sagemaker import get_execution_role



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


"""
    Read input data
"""

def __read_data(files_path, dataset_percentage=100):
    try:
        logger.info("Reading dataset from source...")

        all_files = glob.glob(os.path.join(files_path, "*.csv"))

        datasets = []

        for filename in all_files:
            data = pd.read_csv(
                filename,
                sep=',',
                quotechar='"',
                quoting=csv.QUOTE_ALL,
                escapechar='\\',
                encoding='utf-8',
                error_bad_lines=False
            )

            datasets.append(data)

        data = pd.concat(datasets, axis=0, ignore_index=True)

        data = data.head(int(len(data) * (int(dataset_percentage) / 100)))

        return data
    except Exception as e:
        stacktrace = traceback.format_exc()
        logger.error("{}".format(stacktrace))

        raise e
    
"""
    Read hyperparameters
"""

def __read_params():
    try:
        parser = ArgumentParser()

        parser.add_argument('--epochs', type=int, default=25)
        parser.add_argument('--learning_rate', type=float, default=0.001)
        parser.add_argument('--batch_size', type=int, default=100)
        parser.add_argument('--dataset_percentage', type=str, default=100)
        parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
        parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
        parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
        parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))

        args = parser.parse_args()

        if len(vars(args)) == 0:
            with open(os.path.join("/", "opt", "ml", "input", "config", "hyperparameters.json"), 'r') as f:
                training_params = json.load(f)

            args = Namespace(**training_params)

        return args
    except Exception as e:
        stacktrace = traceback.format_exc()
        logger.error("{}".format(stacktrace))

        raise e
    

if __name__ == '__main__':

    args = __read_params()

    train = __read_data(args.train, args.dataset_percentage)
    validation = __read_data(args.validation, args.dataset_percentage)


# s3_input_train = TrainingInput(
#     s3_data="s3://{}/{}/train".format(bucket, prefix), content_type="csv"
# )
# s3_input_validation = TrainingInput(
#     s3_data="s3://{}/{}/validation/".format(bucket, prefix), content_type="csv"
# )
sess = sagemaker.Session()

container = sagemaker.image_uris.retrieve("xgboost", sess.boto_region_name, "1.7-1")
display(container)

sess = sagemaker.Session()
bucket = sess.default_bucket()
role = get_execution_role()

xgb = sagemaker.estimator.Estimator(
    container,
    role,
    instance_count=1,
    instance_type="ml.m4.xlarge",
    output_path= "s3://{}/models".format(bucket),
    sagemaker_session=sess,
)
xgb.set_hyperparameters(
    max_depth=5,
    eta=0.2,
    gamma=4,
    min_child_weight=6,
    subsample=0.8,
    verbosity=0,
    objective="binary:logistic",
    num_round=100,
)


# logger.info("Start training: {}".format(strftime("%Y-%m-%d %H:%M:%S", gmtime())))


xgb.fit({"train": train, "validation": validation})