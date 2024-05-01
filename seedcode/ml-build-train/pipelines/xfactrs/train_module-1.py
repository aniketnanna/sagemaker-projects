import argparse
import logging
import os
import pickle as pkl
import glob
import csv
import traceback

import pandas as pd
import xgboost as xgb

def __read_data(files_path, dataset_percentage=100):
    try:
        logger.info("Reading dataset from source...")

        all_files = glob.glob(os.path.join(files_path, "*.csv"))

        datasets = []

        for filename in all_files:
            data = pd.read_csv(
                filename,
                sep=',',
                header=None
            )

            datasets.append(data)

        data = pd.concat(datasets, axis=0, ignore_index=True)

        data = data.head(int(len(data) * (int(dataset_percentage) / 100)))

        return data
    except Exception as e:
        stacktrace = traceback.format_exc()
        logger.error("{}".format(stacktrace))

        raise e

def _xgb_train(params, dtrain, evals, num_boost_round, model_dir, is_master):
    """Run xgb train on arguments given with rabit initialized.

    This is our rabit execution function.

    :param params: Hyperparameters for XGBoost training.
    :param dtrain: Training dataset.
    :param evals: Evaluation datasets.
    :param num_boost_round: Number of boosting rounds.
    :param model_dir: Directory to save the trained model.
    :param is_master: True if the current node is the master host.
    """
    booster = xgb.train(params=params,
                        dtrain=dtrain,
                        evals=evals,
                        num_boost_round=num_boost_round)

    if is_master:
        model_location = os.path.join(model_dir, 'xgboost-model')
        pkl.dump(booster, open(model_location, 'wb'))
        logging.info("Stored trained model at {}".format(model_location))
        return booster

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()

    # Hyperparameters are described here
    parser.add_argument('--max_depth', type=int, default=5)
    parser.add_argument('--eta', type=float, default=0.2)
    parser.add_argument('--gamma', type=float, default=4)
    parser.add_argument('--min_child_weight', type=float, default=6)
    parser.add_argument('--subsample', type=float, default=0.8)
    parser.add_argument('--verbosity', type=int, default=0)
    parser.add_argument('--objective', type=str, default='binary:logistic')
    parser.add_argument('--num_round', type=int, default=1)

    # SageMaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))

    args = parser.parse_args()



    # dtrain = xgb.DMatrix(train_data.iloc[:, :-1], label=train_data.iloc[:, -1])
    # dval = xgb.DMatrix(val_data.iloc[:, :-1], label=val_data.iloc[:, -1])
    # evals = [(dtrain, 'train'), (dval, 'validation')] if dval is not None else [(dtrain, 'train')]
 
    train_data = __read_data(args.train)
    val_data = __read_data(args.validation)
    
    dtrain_matrix = xgb.DMatrix(data=train_data.drop(columns=[0]), label=train_data[0])
    
    eval = [(train_data, 'train'), (val_data, 'validation')]
    evals_matrix = [(xgb.DMatrix(e[0].drop(columns=[0]), label=e[0][0]), e[1]) for e in eval]
    

    
    train_hp = {
        'max_depth': args.max_depth,
        'eta': args.eta,
        'gamma': args.gamma,
        'min_child_weight': args.min_child_weight,
        'subsample': args.subsample,
        'verbosity': args.verbosity,
        'objective': args.objective,
        'num_round': args.num_round
    }

    booster = _xgb_train(params=train_hp,
                         dtrain=dtrain_matrix,
                         evals=evals_matrix,
                         num_boost_round=args.num_round,
                         model_dir=args.model_dir,
                         is_master=True)

    # # Save the model to the location specified by ``model_dir``
    # model_location = os.path.join(args.model_dir, 'xgboost-model')
    # pkl.dump(booster, open(model_location, 'wb'))
    # logging.info("Stored trained model at {}".format(model_location))
