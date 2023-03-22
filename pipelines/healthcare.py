import os
import warnings
import numpy
import pandas as pd
import fuzzy_pandas as fpd
from fairlearn.metrics import MetricFrame, false_negative_rate, equalized_odds_difference
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from example_pipelines.healthcare.healthcare_utils import MyW2VTransformer
from utils import get_project_root
warnings.filterwarnings('ignore')
numpy.random.seed(13)

COUNTIES_OF_INTEREST = ['county2', 'county3']

patients = pd.read_csv(os.path.join(str(get_project_root()), "pipelines", "datasets", "healthcare",
                                    "patients.csv"), na_values='?')
histories = pd.read_csv(os.path.join(str(get_project_root()), "pipelines", "datasets", "healthcare",
                                     "histories.csv"), na_values='?')

data = fpd.fuzzy_merge(patients, histories, on='full_name', method='levenshtein',
                       keep_right=['smoker', 'complications'], threshold=0.95)
data = data[data['county'].isin(COUNTIES_OF_INTEREST)]

train_data, test_data = train_test_split(data)

complications = train_data.groupby('age_group') \
    .agg(mean_complications=('complications', 'mean'))
train_data = train_data.merge(complications, on=['age_group'])
test_data = test_data.merge(complications, on=['age_group'])

train_data['label'] = train_data['complications'] > 1.2 * train_data['mean_complications']
test_data['label'] = test_data['complications'] > 1.2 * test_data['mean_complications']

train_data = train_data[['smoker', 'last_name', 'county', 'num_children', 'race', 'income', 'label']]
test_data = test_data[['smoker', 'last_name', 'county', 'num_children', 'race', 'income', 'label']]

impute_and_one_hot_encode = Pipeline([
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('encode', OneHotEncoder(sparse=False, handle_unknown='ignore'))
])
featurisation = ColumnTransformer(transformers=[
    ("impute_and_one_hot_encode", impute_and_one_hot_encode, ['smoker', 'county', 'race']),
    ('word2vec', MyW2VTransformer(min_count=2), ['last_name']),
    ('numeric', StandardScaler(), ['num_children', 'income']),
], remainder='drop')
pipeline = Pipeline([
    ('features', featurisation),
    ('learner', LogisticRegression())])

model = pipeline.fit(train_data, train_data['label'])
test_predictions = model.predict(test_data)
print("Mean accuracy: {}".format(accuracy_score(test_data['label'], test_predictions)))

sensitive_features = test_data[['race']]
sensitive_features['race'] = sensitive_features['race'].astype(str)
fnr_by_group = MetricFrame(metrics=false_negative_rate, y_pred=test_predictions, y_true=test_data['label'],
                           sensitive_features=sensitive_features)
print(f"False-negative by group: {fnr_by_group.by_group}")
equalized_odds_diff = equalized_odds_difference(y_pred=test_predictions, y_true=test_data['label'],
                                                sensitive_features=sensitive_features)
print(f"Equalized odds difference: {equalized_odds_diff}")
