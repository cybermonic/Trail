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
        if ioc_type == 'domains':
            self.X_train = pd.read_csv(ml_dir / 'domains_features_train.csv', index_col='Unnamed: 0')
            self.y_train = pd.read_csv(ml_dir / 'domains_labels_train.csv', index_col='Unnamed: 0')

            self.X_test = pd.read_csv(ml_dir / 'domains_features_test.csv', index_col='Unnamed: 0')
            self.y_test = pd.read_csv(ml_dir / 'domains_labels_test.csv', index_col='Unnamed: 0')

            self.X_val = pd.read_csv(ml_dir / 'domains_features_val.csv', index_col='Unnamed: 0')
            self.y_val = pd.read_csv(ml_dir / 'domains_labels_val.csv', index_col='Unnamed: 0')

    def process_data(self,
                     select_classes: list = []) -> None:
        """
        Here we will process the data how we want to. We do this beacuse we might not want to train on all classes.
        For example in a sequential model we might want to train on only some classes.
        """
        if select_classes:
            # Load in mapper
            apt_map = get_label_mapper()
            # Glue the features and labels together. SInce we need to proceees this
            train = pd.concat([self.X_train, self.y_train], axis=1)
            test = pd.concat([self.X_test, self.y_test], axis=1)
            val = pd.concat([self.X_val, self.y_val], axis=1)
            # Make human readable labels
            train['label'] = train['0'].astype(str).map(apt_map)
            test['label'] = test['0'].astype(str).map(apt_map)
            val['label'] = val['0'].astype(str).map(apt_map)
            # Now filter DataFrame by classes in select_classes
            train_filtered = train[train['label'].isin(select_classes)]
            test_filtered = test[test['label'].isin(select_classes)]
            val_filtered = val[val['label'].isin(select_classes)]
            # Now split it up again
            self.X_train = train_filtered.drop(columns=['0', 'label']).to_numpy()
            self.X_test = test_filtered.drop(columns=['0', 'label']).to_numpy()
            self.X_val = val_filtered.drop(columns=['0', 'label']).to_numpy()

            self.y_train = train_filtered['0'].to_numpy()
            self.y_test = test_filtered['0'].to_numpy()
            self.y_val = val_filtered['0'].to_numpy()

            # Encode this.
            label_encoder = LabelEncoder()
            self.y_train = label_encoder.fit_transform(self.y_train)
            self.y_test = label_encoder.transform(self.y_test)
            self.y_val = label_encoder.transform(self.y_val)
            # Save the label encoder to use for future infrences with the model. This will
            # Let us map back to the original APT threat actor
            joblib.dump(label_encoder, f'{config.get("ML_DATA")}{self.model_title}_label_encoder.joblib')

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
        # self.y_train = np.argmax(self.y_train.values, axis=1)
        # self.y_test = np.argmax(self.y_test.values, axis=1)
        #sample_weights = compute_sample_weight(
        #    class_weight='balanced',
        #    y=self.y_train
        #)
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
        self.dval = xgb.DMatrix(self.X_test, label=self.y_test)

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

        # y_pred = self.best_model.predict(self.dtest)
        # y_pred_val = self.best_model.predict(self.dval)

        # feature_names = list(self.X_train.columns)
        # shap_values = self.best_model.predict(self.dtrain, pred_contribs=True)
        # shap_values = shap_values[:, :, :-1]

        # Assuming shap_values is your array of SHAP values with shape (341181, 24, 117)
        # mean_abs_shap = np.mean(np.abs(shap_values), axis=(0, 1))

        # Rank features based on mean absolute SHAP value
        # ranked_indices = np.argsort(mean_abs_shap)[::-1]  # [::-1] is used to sort in descending order
        # self.shap_dict = {feature_names[i]: mean_abs_shap[i] for i in ranked_indices}

    def objective(self, search_space: Dict):
        # Train model using chosen search space
        model = xgb.train(search_space,
                          self.dtrain, num_boost_round=75,
                          evals=[(self.dval, 'val')],
                          early_stopping_rounds=10,
                          verbose_eval=False)
        # Get predictions
        y_pred = model.predict(self.dtest)

        # Convert probabilities to class labels
        y_pred_labels = np.argmax(y_pred, axis=1)

        # Now, calculate the balanced accuracy score. We choose to optimize on balanced accuracy
        loss = balanced_accuracy_score(self.y_test, y_pred_labels)
        acc = accuracy_score(self.y_test, y_pred_labels)
        if self.best_metric < loss:
            self.best_metric = loss
            self.best_model = model
            self.save(model_title=self.model_title)
            print("""
            Current Best Metrics
            ---------------------
            ACC:    {acc:.4f}
            BACC:   {bacc:.4f}
            """.format(acc=acc, bacc=loss))
        # Return the loss
        return {'loss': 1 - loss, 'status': STATUS_OK}


class BinaryAPT(XGBoostGPUClassifierAPT):
    """
    This class is trains a binary-classification XGBoost model for APT/Threat Actors
    """

    def train(self,
              num_evals: int,
              select_classes: list) -> None:
        """
        This method will train the models as follows:

        1. Address class imbalance by weighting the minority classes higher through compute_sample_weight
        2. Create DMatrix objects from training, test and validation sets
        3. Create a search space (Tree Architecture) to optimize on using hypopt library using a metric of choice.
        4. Define objective function to minimize (usually 1-{metric}) where metric range is [0,1].
        5. Iterate num_evals until we find the best model based on our objective function.
        6. Save best model

        Args:
            num_evals (int): Number of evaluations to go through optimization.
            select_classes (list):
        """
        # Process data before training
        self.process_data(select_classes=select_classes)
        sample_weights = compute_sample_weight(
                class_weight='balanced',
                y=self.y_train
            )
        self.dtrain = xgb.DMatrix(self.X_train, label=self.y_train, weight=sample_weights)
        self.dtest = xgb.DMatrix(self.X_test, label=self.y_test)
        self.dval = xgb.DMatrix(self.X_test, label=self.y_test)

        search_space = {
            'objective': 'binary:logistic',
            'learning_rate': hp.loguniform('learning_rate', -0.75, 0),
            'max_depth': hp.choice('max_depth', range(1, 32)),
            'min_child_weight': hp.choice('min_child_weight', range(1, 100)),
            'gamma': hp.uniform('gamma', 0, 5),
            'subsample': hp.uniform('subsample', 0.1, 1),  # Adjusted range to avoid 0 which is invalid
            'colsample_bytree': hp.uniform('colsample_bytree', 0.1, 1),
            # 'tree_method': 'gpu_hist', We have no GPU, but if we do, use this to use it
            # 'predictor': 'gpu_predictor' We have no GPU, but if we do, use this to use it
        }

        trials = Trials()

        fmin(fn=self.objective,
             space=search_space,
             algo=tpe.suggest,
             max_evals=num_evals,
             trials=trials)

        # feature_names = list(self.X_train.columns)
        # shap_values = self.best_model.predict(self.dtrain, pred_contribs=True)
        # shap_values = shap_values[:, :, :-1]

        # mean_abs_shap = np.mean(np.abs(shap_values), axis=(0, 1))

        # Rank features based on mean absolute SHAP value
        # ranked_indices = np.argsort(mean_abs_shap)[::-1]  # [::-1] is used to sort in descending order
        # self.shap_dict = {feature_names[i]: mean_abs_shap[i] for i in ranked_indices}

    def objective(self, search_space: Dict):
        # Train model using chosen search space
        model = xgb.train(search_space,
                          self.dtrain, num_boost_round=75,
                          evals=[(self.dtest, 'test')],
                          early_stopping_rounds=10,
                          verbose_eval=False)
        # Get predictions
        y_pred = model.predict(self.dtest)

        # Convert probabilities to class labels using a threshold of 0.5
        y_pred_labels = np.where(y_pred > 0.5, 1, 0)

        # Now, calculate the balanced accuracy score. We choose to optimize on balanced accuracy
        loss = balanced_accuracy_score(self.y_test, y_pred_labels)
        acc = accuracy_score(self.y_test, y_pred_labels)
        precision = precision_score(self.y_test, y_pred_labels, zero_division=0)
        recall = recall_score(self.y_test, y_pred_labels, zero_division=0)
        f1 = f1_score(self.y_test, y_pred_labels, zero_division=0)
        auc = roc_auc_score(self.y_test, y_pred)
        if self.best_metric < loss:
            self.best_metric = loss
            self.best_model = model
            self.save(model_title=self.model_title)
            print(f"""
            Current Best Metrics
            ---------------------
            ACC:    {acc:.4f}
            BACC:   {loss:.4f}
            Precision: {precision:.4f}
            Recall: {recall:.4f}
            F1:     {f1:.4f}
            AUC:    {auc:.4f}
            """)
        # Return the loss
        return {'loss': 1 - loss, 'status': STATUS_OK}


if __name__ == '__main__':
    ################# Train on whole dataset############################################

    MultiAPT(config=config,
             model_title=f"APT_XGBoost_domain_full",
             model_definition='pure_model',
             ioc_type='domains').train(num_evals=3000)
    ##############################################################################################

    ################# Train on classes in each Cluster############################################
    import joblib

    # load in cluster for each iocs type
    domain_clusters = f"{config.get('ML_DATA')}domains_cluster_map.joblib"
    domain_clusters_map = joblib.load(domain_clusters)
    for cluster, select_classes in domain_clusters_map.items():
        # Train Domain Classifier
        # It the number of classes if more than 2, we use the multi calss model
        if len(select_classes) > 2:
            # MultiAPT(config=config, model_title=f"APT_XGBoost_domain_{cluster}").train(num_evals=1000,
            #                                                                           select_classes=select_classes)
            pass
        elif len(select_classes) == 1:
            continue
        else:
            # Other wise we can use a binary classifier
            BinaryAPT(config=config, model_title=f"APT_XGBoost_domain_{cluster}").train(num_evals=200,
                                                                                        select_classes=select_classes)
    ##############################################################################################
