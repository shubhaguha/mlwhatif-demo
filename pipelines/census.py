import datetime
import os
import random

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, RobustScaler, StandardScaler
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier

from pipelines.datasets.healthcare.healthcare_utils import MyW2VTransformer
from utils import get_project_root

def get_featurization():
    union = FeatureUnion([("pca", PCA(n_components=(len(numerical_columns) - 1))),
                          ("svd", TruncatedSVD(n_iter=1, n_components=(len(numerical_columns) - 1)))])
    num_pipe = Pipeline([('union', union), ('scaler', StandardScaler())])

    transformers = [('num', num_pipe, numerical_columns)]

    def my_imputer(df_with_categorical_columns):
        return df_with_categorical_columns.fillna('__missing__').astype(str)

    for cat_column in categorical_columns:
        cat_pipe = Pipeline([('anothersimpleimputer', FunctionTransformer(my_imputer)),
                             ('onehotencoder', OneHotEncoder(handle_unknown='ignore'))])
        transformers.append((f"cat_{cat_column}", cat_pipe, [cat_column]))

    featurization = ColumnTransformer(transformers)
    return featurization

np.random.seed(42)
random.seed(42)

acs_data = pd.read_csv(os.path.join(str(get_project_root()), "pipelines", "datasets", "folktables",
                                    "acs_income_RI_2017_5y.csv"), delimiter=";")
columns = ['AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'WKHP', 'SEX', 'RAC1P', 'PINCP']
acs_data = acs_data[columns]

numerical_columns = ['AGEP', 'WKHP']
categorical_columns = ['COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP']
train, test = train_test_split(acs_data)
train_labels = train['PINCP']
test_labels = test['PINCP']
featurization = get_featurization()
model = XGBClassifier(max_depth=12, tree_method='hist', n_jobs=1)
pipeline = Pipeline([('featurization', featurization), ('model', model)])
pipeline = pipeline.fit(train, train_labels)
predictions = pipeline.predict(test)
print('    Score: ', accuracy_score(test_labels, predictions))
