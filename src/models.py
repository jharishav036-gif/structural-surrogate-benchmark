"""The nine-model tournament: estimators, pipelines and tuning spaces.

Every model is wrapped in a pipeline with standardisation of the inputs
so that distance-based and gradient-based models see scaled features
while tree models are unaffected.  Hyperparameters are selected by
randomised search (n_iter = 10, inner 5-fold CV) in train_eval.py; the
spaces below define the candidate ranges.
"""

from scipy.stats import loguniform, randint, uniform

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


def make_pipeline(name, seed=42, **params):
    if name == 'Linear':
        est = LinearRegression()
    elif name == 'KNN':
        est = KNeighborsRegressor(**params)
    elif name == 'SVR':
        est = SVR(**params)
    elif name == 'Tree':
        est = DecisionTreeRegressor(random_state=seed, **params)
    elif name == 'RF':
        est = RandomForestRegressor(random_state=seed, n_jobs=1, **params)
    elif name == 'XGB':
        est = XGBRegressor(objective='reg:squarederror', random_state=seed,
                           n_jobs=1, verbosity=0, **params)
    elif name == 'LGBM':
        est = LGBMRegressor(random_state=seed, n_jobs=1, verbose=-1,
                            subsample_freq=1, **params)
    elif name == 'CatBoost':
        est = CatBoostRegressor(random_seed=seed, verbose=0,
                                allow_writing_files=False, thread_count=1,
                                **params)
    elif name == 'MLP':
        est = MLPRegressor(random_state=seed, max_iter=1000,
                           early_stopping=True, validation_fraction=0.15,
                           **params)
    else:
        raise ValueError(name)
    return Pipeline([('scaler', StandardScaler()), ('model', est)])


TREE_MODELS = ('RF', 'XGB', 'LGBM', 'CatBoost')

MODEL_NAMES = ['Linear', 'KNN', 'SVR', 'Tree', 'RF', 'XGB', 'LGBM',
               'CatBoost', 'MLP']

SEARCH_SPACES = {
    'Linear': {},
    'KNN': {
        'model__n_neighbors': randint(3, 22),
        'model__weights': ['uniform', 'distance'],
        'model__p': [1, 2],
    },
    'SVR': {
        'model__C': loguniform(1e-1, 1e2),
        'model__epsilon': [0.01, 0.05, 0.1, 0.2],
        'model__gamma': ['scale', 0.05, 0.1, 0.2],
    },
    'Tree': {
        'model__max_depth': [6, 8, 10, 12, 16, 20, None],
        'model__min_samples_leaf': [1, 2, 5, 10],
    },
    'RF': {
        'model__n_estimators': [300, 500, 800],
        'model__max_depth': [None, 12, 20, 30],
        'model__min_samples_leaf': [1, 2, 4],
        'model__max_features': ['sqrt', 0.5],
    },
    'XGB': {
        'model__n_estimators': [400, 600, 900],
        'model__max_depth': [3, 4, 5, 6, 8],
        'model__learning_rate': loguniform(0.02, 0.2),
        'model__subsample': uniform(0.7, 0.3),
        'model__colsample_bytree': uniform(0.7, 0.3),
        'model__min_child_weight': [1, 3, 5],
    },
    'LGBM': {
        'model__n_estimators': [400, 600, 900],
        'model__num_leaves': [15, 31, 63],
        'model__learning_rate': loguniform(0.02, 0.2),
        'model__subsample': uniform(0.7, 0.3),
        'model__colsample_bytree': uniform(0.7, 0.3),
        'model__min_child_samples': [5, 10, 20],
    },
    'CatBoost': {
        'model__iterations': [400, 600, 900],
        'model__depth': [4, 6, 8],
        'model__learning_rate': loguniform(0.03, 0.15),
        'model__l2_leaf_reg': [1, 3, 5, 9],
    },
    'MLP': {
        'model__hidden_layer_sizes': [(32,), (64,), (64, 32), (128, 64),
                                      (64, 64)],
        'model__alpha': loguniform(1e-5, 1e-2),
        'model__learning_rate_init': [1e-3, 3e-3],
        'model__batch_size': [32, 64],
    },
}
