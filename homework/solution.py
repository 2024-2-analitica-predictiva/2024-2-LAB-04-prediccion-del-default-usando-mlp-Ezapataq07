import pandas as pd
import numpy as np
import time
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, chi2
import pickle
import gzip
import json, os
np.set_printoptions(legacy='1.25')
from sklearn.metrics import (
    balanced_accuracy_score,
    recall_score, f1_score, confusion_matrix, precision_score, accuracy_score)

class Lab04:
    def __init__(self) -> None:
        self.time = time.time()
        self.files_path = 'files/'
        self.columnas_categoricas = ['SEX','EDUCATION','MARRIAGE']

        self.param_grid = {
            'mlp__hidden_layer_sizes': [(51, 31, 41)],#[(28,1),(28,2),(28,14,2),(28,14,1),(28,28,2),(28,28,1)],
            'mlp__learning_rate_init': [0.000172],
            'mlp__alpha': [0.26],
            'mlp__beta_1': [0.99],
            'mlp__beta_2': [0.996],
            # 'mlp__learning_rate': ['constant','adaptative','invscaling'], 
            # 'mlp__activation': ['identity','logistic', 'tanh', 'relu'], # 'relu',
            # 'mlp__solver': ['lbfgs'],
            # 'mlp__early_stopping': [False,True],
            # #'mlp__momentum': [0.9,0.5],
            'selectk__k': [20],
            # 'selectk__score_func': [f_classif]#,mutual_info_classif,chi2]
        }
        print(self.param_grid)

    def main(self):
        df_train = self.read_dataset('input/train_data.csv.zip')
        df_test = self.read_dataset('input/test_data.csv.zip')
        df_train = self.clean_dataset(df_train)
        df_test = self.clean_dataset(df_test)
        X_train,  y_train = self.train_test_split(df_train)
        X_test,  y_test = self.train_test_split(df_test)
        self.columnas_no_categoricas = list(set(X_train.columns.values) - set(self.columnas_categoricas))
        pipeline = self.make_pipeline(MLPClassifier(random_state=2024,max_iter=15000))#max_iter=10000, random_state=0, activation='relu', validation_fraction=0.3, early_stopping=True))
        estimator = self.make_grid_search(pipeline, 'accuracy', cv=StratifiedKFold(n_splits=10))#,shuffle=True,random_state=2024))
        estimator = estimator.fit(X_train, y_train)
        self.save_estimator(estimator)#self.save_model_if_best(estimator, X_train, y_train)
        print(estimator.best_params_)
        y_train_pred = estimator.predict(X_train)
        y_test_pred = estimator.predict(X_test) 
        metrics_train = self.eval_metrics('train', y_train, y_train_pred)
        metrics_test = self.eval_metrics('test', y_test, y_test_pred)
        cm_train = self.eval_confusion_matrix('train', y_train, y_train_pred)
        cm_test = self.eval_confusion_matrix('test', y_test, y_test_pred)
        self.save_metrics(metrics_train, metrics_test, cm_train, cm_test)
        print(f'Minutos: {(time.time() - self.time)/60}')



    def read_dataset(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(self.files_path + path)
        return df
    
    def clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.rename(columns={'default payment next month': 'default'})
        df.drop('ID', axis=1, inplace=True)
        df['EDUCATION'] = df['EDUCATION'].apply(lambda x: 4 if x>4 else x)
        df = df.query('MARRIAGE != 0 and EDUCATION != 0')
        return df
    
    def train_test_split(self, df):
        return df.drop('default', axis=1), df['default']
    
    def make_pipeline(self, estimator):
        transformer = ColumnTransformer(
            transformers=[
                ('ohe', OneHotEncoder(dtype='int64'), self.columnas_categoricas),
            ],
            remainder='passthrough'
        )

        pipeline = Pipeline(
            steps=[
                ('transformer', transformer),
                ('scaler', StandardScaler()),
                ('pca', PCA(random_state=2024)),
                ('selectk', SelectKBest(score_func=f_classif, k='all')),
                ('mlp', estimator)
            ]
        )
        return pipeline
        
    def make_grid_search(self, estimator, scoring, cv=10):
        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=self.param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
        return grid_search

    def save_model_if_best(self, estimator, X, y):
        best_estimator = self.load_model()
        if best_estimator:
            saved_accuracy = balanced_accuracy_score(y, best_estimator.predict(X))
            current_accuracy = balanced_accuracy_score(y, estimator.predict(X))

            if current_accuracy > saved_accuracy:
                self.save_estimator(estimator)
            else:
                estimator = best_estimator
        else:
            self.save_estimator(estimator)
        return estimator


    def save_estimator(self, estimator):
        with gzip.open(self.files_path + 'models/model.pkl.gz', 'wb') as file:
            pickle.dump(estimator, file)

    def load_model(self):
        try:
            with gzip.open(self.files_path + "models/model.pkl.gz", "rb") as file:
                estimator = pickle.load(file)
            return estimator
        except Exception as E:
            print(E)
            return None
    
    def eval_metrics(self, dataset,y_true, y_pred):
        accuracy = precision_score(y_true, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        return {"type": "metrics","dataset": dataset, "precision": accuracy, "balanced_accuracy": balanced_accuracy, "recall": recall, "f1_score": f1} 
    
    def eval_confusion_matrix(self,dataset,y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        cm_train_dict = {
            'type': 'cm_matrix',
            'dataset': dataset,
            'true_0': {
                'predicted_0': int(cm[0,0]),
                'predicted_1': int(cm[0,1])
            },
            'true_1': {
                'predicted_0': int(cm[1,0]),
                'predicted_1': int(cm[1,1])
            }
        }
        return cm_train_dict
    
    def save_metrics(self, metrics_train, metrics_test, cm_train, cm_test):
        os.remove(self.files_path + 'output/metrics.json')
        with open(self.files_path + 'output/metrics.json', mode='w') as file:
            file.write(json.dumps(metrics_train)+"\n")
            file.write(json.dumps(metrics_test)+"\n")
            file.write(json.dumps(cm_train)+"\n")
            file.write(json.dumps(cm_test)+"\n")


if __name__=='__main__':
    obj = Lab04()
    obj.main()