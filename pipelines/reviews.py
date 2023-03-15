import datetime
import os
import random

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, RobustScaler
from sklearn.preprocessing import label_binarize

from utils import get_project_root

data_root = os.path.join(str(get_project_root()), "pipelines", "datasets")

def random_subset(arr):
    size = np.random.randint(low=1, high=len(arr) + 1)
    choice = np.random.choice(arr, size=size, replace=False)
    return [str(item) for item in choice]


def load_data():
    reviews = pd.read_csv(os.path.join(data_root, "reviews", "reviews.csv.gz"), compression='gzip',
                          index_col=0)
    ratings = pd.read_csv(os.path.join(data_root, "reviews", "ratings.csv"), index_col=0)
    products = pd.read_csv(os.path.join(data_root, "reviews", "products.csv"), index_col=0)
    categories = pd.read_csv(os.path.join(data_root, "reviews", "categories.csv"), index_col=0)

    return reviews, ratings, products, categories


def integrate_data(reviews, ratings, products, categories, start_date):
    reviews = reviews[reviews['review_date'] >= start_date.strftime('%Y-%m-%d')]

    reviews_with_ratings = reviews.merge(ratings, on='review_id')
    products_with_categories = products.merge(left_on='category_id', right_on='id', right=categories)

    random_categories = random_subset(list(categories.category))
    products_with_categories = products_with_categories[
        products_with_categories['category'].isin(random_categories)]

    reviews_with_products_and_ratings = reviews_with_ratings.merge(products_with_categories, on='product_id')

    return reviews_with_products_and_ratings


def compute_feature_and_label_data(reviews_with_products_and_ratings, final_columns, split_date):
    reviews_with_products_and_ratings['is_helpful'] = reviews_with_products_and_ratings['helpful_votes'] > 0

    projected_reviews = reviews_with_products_and_ratings[final_columns]

    train_data = projected_reviews[projected_reviews['review_date'] <= split_date.strftime('%Y-%m-%d')]
    train_labels = label_binarize(train_data['is_helpful'], classes=[True, False]).ravel()

    test_data = projected_reviews[projected_reviews['review_date'] > split_date.strftime('%Y-%m-%d')]
    test_labels = label_binarize(test_data['is_helpful'], classes=[True, False]).ravel()

    return train_data, train_labels, test_data, test_labels

def get_featurization(numerical_columns, categorical_columns, text_columns):
    transformers = [('num', RobustScaler(), numerical_columns)]
    if len(text_columns) >= 1:
        assert len(text_columns) == 1
        transformers.append(('text', HashingVectorizer(n_features=2 ** 5), text_columns[0]))
    for cat_column in categorical_columns:
        def another_imputer(df_with_categorical_columns):
            return df_with_categorical_columns.fillna('__missing__').astype(str)

        cat_pipe = Pipeline([('anothersimpleimputer', FunctionTransformer(another_imputer)),
                             ('onehotencoder', OneHotEncoder(handle_unknown='ignore'))])

        transformers.append((f"cat_{cat_column}", cat_pipe, [cat_column]))

    featurization = ColumnTransformer(transformers)
    return featurization


def get_model():
    model = SGDClassifier(loss='log', max_iter=30, n_jobs=1)
    return model

seed = 42
np.random.seed(seed)
random.seed(seed)

numerical_columns = ['total_votes', 'star_rating']
categorical_columns = ['vine', 'category']
text_columns = ["review_body"]
final_columns = numerical_columns + categorical_columns + text_columns + ['is_helpful', 'review_date']

reviews, ratings, products, categories = load_data()

start_date = datetime.date(2011, 6, 22)
integrated_data = integrate_data(reviews, ratings, products, categories, start_date)

split_date = datetime.date(2013, 12, 20)
train, train_labels, test, test_labels = compute_feature_and_label_data(integrated_data, final_columns, split_date)

featurization = get_featurization(numerical_columns, categorical_columns, text_columns)
model = get_model()
pipeline = Pipeline([('featurization', featurization),
                     ('model', model)])
pipeline = pipeline.fit(train, train_labels)
predictions = pipeline.predict(test)
score = accuracy_score(test_labels, predictions)
print('    Score: ', score)
