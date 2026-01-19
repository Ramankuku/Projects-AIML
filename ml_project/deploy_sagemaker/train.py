
import argparse
import joblib
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf




if __name__ == "__main__":
    print("extracting arguments")
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_estimators', type=int, default = 50)
    parser.add_argument('--criterion', type=str, default = 'entropy')
    parser.add_argument('--max_features', type=str, default = 'log2')
    parser.add_argument('--max_depth', type=int, default = 5)
    parser.add_argument('--min_samples_split', type=int, default = 2)

    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])

    args, _ = parser.parse_known_args()

    print("reading data")
    train_df = pd.read_csv(os.path.join(args.train, 'train_data.csv'))
    test_df = pd.read_csv(os.path.join(args.test, 'test_data.csv'))

    X_train = train_df.drop(columns=['Churn'])
    X_test = test_df.drop(columns=['Churn'])

    y_train = train_df['Churn']
    y_test = test_df['Churn']

    print("training model")

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        criterion=args.criterion,
        max_features=args.max_features,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split, 
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    path = os.path.join(args.model_dir, 'model.joblib')
    joblib.dump(model, path)
    print(f'Joblib dump at Path: {path}')

    y_pred = model.predict(X_test)
    print('Checking Accuracy')
    acc = accuracy_score(y_test, y_pred)
    print(f'Accuracy of the model: {acc}')

