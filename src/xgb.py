"""
This module is used to train classifers 
"""
import json

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
from hyperopt import fmin, hp, STATUS_OK, tpe, Trials
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from typing import Dict
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from build_dataset.label_mapper.apt_label_mapper import build_ta_map as get_label_mapper
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from config import config 

# Ignore the specific sklearn UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


class Classifier:

    def __init__(self,
                 config: Dict,
                 model_title: str,
                 ioc_type: str,
                 model_definition: str) -> None:
        """
        Args:
            config (Dict): config file for system directions
            model_title (str): Title of model (Your choice)
            ioc_type (str): Ioc type we are training on. Like domains, ips and uls
            model_definition (str): Type of model. Like a sequential model, pure model or any other model definition we
                                    want to come up with. Right now we have the following: pure_model & sequential_model
        """
        self.config = config
        self.model_title = model_title
        self.ioc_type = ioc_type
        self.model_definition = model_definition



class XGBoostGPUClassifierAPT(Classifier):
    class XGBoostGPUClassifierAPT(Classifier):

        def __init__(self,
                     config: Dict,
                     ioc_type: str,
                     model_definition: str,
                     model_title: str = 'APT_XGBoost'):
            super().__init__(config=config,
                             model_title=model_title,
                             ioc_type=ioc_type,
                             model_definition=model_definition)
            # During the optimization stage, we will keep track of the self.best_metric (We can choose this) and save the
            # self.best_model with title self.model_title
            self.best_metric = 0
            self.best_model = None
            # Load in datasets
            ml_dir = Path(config.get('ML_DATA'))

            # This is where we will load in the train/test/val data. This can differ from user to user
            if ioc_type == 'domains':
                self.X_train = pd.read_csv(ml_dir / 'domains_features_train.csv', index_col='Unnamed: 0')
                self.y_train = pd.read_csv(ml_dir / 'domains_labels_train.csv', index_col='Unnamed: 0')

                self.X_test = pd.read_csv(ml_dir / 'domains_features_test.csv', index_col='Unnamed: 0')
                self.y_test = pd.read_csv(ml_dir / 'domains_labels_test.csv', index_col='Unnamed: 0')

                self.X_val = pd.read_csv(ml_dir / 'domains_features_val.csv', index_col='Unnamed: 0')
                self.y_val = pd.read_csv(ml_dir / 'domains_labels_val.csv', index_col='Unnamed: 0')

            if ioc_type == 'ips':
                ip_data = np.load(ml_dir / 'ips.npz')
                X = ip_data['x']
                y = ip_data['y']

                # Initialize the LabelEncoder
                label_encoder = LabelEncoder()

                # Fit the label encoder and transform y to ensure labels are continuous
                y = label_encoder.fit_transform(y)

                # First, split the data into training+validation and test sets (e.g., 80% train+val, 20% test)
                X_train_val, self.X_test, y_train_val, self.y_test = train_test_split(X, y, test_size=0.2,
                                                                                      random_state=42)

                # Then, split the training+validation data into separate training and validation sets (e.g., 75% train, 25% val)
                self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X_train_val, y_train_val,
                                                                                      test_size=0.25, random_state=42)


    def save(self,
             model_title: str):
        """
        This method is used to save the model to their appropriate destinations
        """
        # Construct the full path where the model will be saved
        ml_model_out = Path(config.get('ML_DATA')) / self.ioc_type / self.model_definition
        model_path = ml_model_out / f"{model_title}.json"  # Using JSON format for saving the model

        # Use XGBoost's built-in save_model method
        self.best_model.save_model(model_path)
        print(f"Model saved to {model_path}")


class MultiAPT(XGBoostGPUClassifierAPT):
    """
    This class is trains a multi-classification XGBoost model for APT/Threat Actors
    """

    def train(self,
              num_evals: int,
              select_classes: list = []) -> None:
        """
        This method will train the models as follows:

        1. Address class imbalance by weighting the minority classes higher through compute_sample_weight
        2. Create DMatrix objects from training, test and validation sets
        3. Create a search space (Tree Architecture) to optimize on using hypopt library using a metric of choice.
        4. Define objective function to minimize (usually 1-{metric}) where metric range is [0,1].
        5. Iterate num_evals until we find the best model based on our objective function.
        6. Save best model

        Args:
            num_evals (int): Number of evaluations to go through optimization
        """
        # Process data before training
        self.process_data(select_classes=select_classes)

        num_classes = len(np.unique(self.y_train))
        classes = np.unique(self.y_train)
        weights = compute_class_weight(class_weight='balanced',
                                       classes=classes,
                                       y=self.y_train.squeeze())

        # Map from class labels to computed class weights (for easier lookup)
        class_weights_map = dict(zip(classes, weights))

        # Efficiently create a sample weights array for each instance in y_train
        sample_weights = np.vectorize(class_weights_map.get)(self.y_train)
        self.dtrain = xgb.DMatrix(self.X_train, label=self.y_train, weight=sample_weights)
        self.dtest = xgb.DMatrix(self.X_test, label=self.y_test)
        self.dval = xgb.DMatrix(self.X_val, label=self.y_val)

        # Modifying this is key in getting the best performance for our models. Ideally we want to get the search space
        # just right so that the optimization can find the true global min
        search_space = {
            'objective': 'multi:softprob',
            'num_class': num_classes,
            'learning_rate': hp.loguniform('learning_rate', -0.75, 0),
            'max_depth': hp.choice('max_depth', range(1, 32)),
            'min_child_weight': hp.choice('min_child_weight', range(1, 100)),
            'gamma': hp.uniform('gamma', 0, 5),
            'subsample': hp.uniform('subsample', 0, 1),
            'colsample_bytree': hp.uniform('colsample_bytree', 0, 1),
            # 'tree_method': 'gpu_hist', We have no GPU, but if we do, use this to use it
            # 'predictor': 'gpu_predictor' We have no GPU, but if we do, use this to use it
        }

        trials = Trials()

        fmin(fn=self.objective,
             space=search_space,
             algo=tpe.suggest,
             max_evals=num_evals,
             trials=trials)

    def objective(self, search_space: Dict):
        # Train the model using the chosen search space
        model = xgb.train(search_space,
                          self.dtrain, num_boost_round=75,
                          evals=[(self.dtest, 'val')],
                          early_stopping_rounds=10,
                          verbose_eval=False)

        # Get predictions for the test set
        y_pred_test = model.predict(self.dtest)
        y_pred_labels_test = np.argmax(y_pred_test, axis=1)

        # Get predictions for the validation set
        y_pred_val = model.predict(self.dval)
        y_pred_labels_val = np.argmax(y_pred_val, axis=1)

        # Get predictions for the training set
        y_pred_train = model.predict(self.dtrain)
        y_pred_labels_train = np.argmax(y_pred_train, axis=1)

        # Calculate training, testing, and validation accuracy scores
        loss_train = balanced_accuracy_score(self.y_train, y_pred_labels_train)
        acc_train = accuracy_score(self.y_train, y_pred_labels_train)

        loss_test = balanced_accuracy_score(self.y_test, y_pred_labels_test)
        acc_test = accuracy_score(self.y_test, y_pred_labels_test)

        loss_val = balanced_accuracy_score(self.y_val, y_pred_labels_val)
        acc_val = accuracy_score(self.y_val, y_pred_labels_val)

        # Update best model if the new model is better
        if self.best_metric < loss_test:
            self.best_metric = loss_test
            self.best_model = model
            self.save(model_title=self.model_title)

            # Print training, testing, and validation metrics
            print(f"""
            Current Best Metrics
            ---------------------
            Training Accuracy  (ACC):  {acc_train:.4f}
            Training Balanced Accuracy (BACC):  {loss_train:.4f}

            Testing Accuracy  (ACC):  {acc_test:.4f}
            Testing Balanced Accuracy (BACC):  {loss_test:.4f}

            Validation Accuracy  (ACC):  {acc_val:.4f}
            Validation Balanced Accuracy (BACC):  {loss_val:.4f}
            """)

        # Return the loss for hyperparameter optimization
        return {'loss': 1 - loss_test, 'status': STATUS_OK}

if __name__ == '__main__':
    ################# Train on whole dataset############################################

    MultiAPT(config=config,
             model_title=f"APT_XGBoost_domain_full",
             model_definition='pure_model',
             ioc_type='domains').train(num_evals=3000)