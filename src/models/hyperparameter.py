#importing all necessary libraries
from hyperopt import hp, scope
from hyperopt.pyll.base import scope
import pickle
from pathlib import Path


# Define the hyperparameter search space for each model
hyperparameters = {
    'linear_reg': {},
    
    'svr': {
        'C': hp.loguniform('C_svr', -3, 3),
        'kernel': hp.choice('kernel_svr', ['linear', 'rbf', 'poly']),
        'degree': scope.int(hp.quniform('degree_svr', 2, 5, 1)),
        'gamma': hp.loguniform('gamma_svr', -3, 3)
    },
    
    'ridge': {
        'alpha': hp.loguniform('alpha_ridge', -3, 3)
    },
    
    'LASSO': {
        'alpha': hp.loguniform('alpha_lasso', -3, 3)
    },
    
    'decision_tree': {
        'max_depth': scope.int(hp.quniform('max_depth_dt', 5, 20, 1)),
        'min_samples_split': scope.int(hp.quniform('min_samples_split_dt', 2, 20, 1)),
        'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf_dt', 1, 10, 1)),
        'criterion': hp.choice('criterion_dt', ['mse', 'friedman_mse', 'mae'])
    },
    
    'random_forest': {
        'n_estimators': scope.int(hp.quniform('n_estimators_rf', 10, 20, 1)),
        'max_depth': scope.int(hp.quniform('max_depth_rf', 5, 20, 1)),
        'min_samples_split': scope.int(hp.quniform('min_samples_split_rf', 2, 20, 1)),
        'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf_rf', 1, 10, 1)),
        'max_features': hp.choice('max_features_rf', ['auto', 'sqrt', 'log2'])
    },
    
    'extra_trees': {
        'n_estimators': scope.int(hp.quniform('n_estimators_et', 50, 200, 1)),
        'max_depth': scope.int(hp.quniform('max_depth_et', 5, 20, 1)),
        'min_samples_split': scope.int(hp.quniform('min_samples_split_et', 2, 20, 1)),
        'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf_et', 1, 10, 1)),
        'max_features': hp.choice('max_features_et', ['auto', 'sqrt', 'log2'])
    },
    
    'gradient_boosting': {
        'n_estimators': scope.int(hp.quniform('n_estimators_gb', 50, 200, 1)),
        'max_depth': scope.int(hp.quniform('max_depth_gb', 5, 20, 1)),
        'learning_rate': hp.loguniform('learning_rate_gb', -3, 0),
        'min_samples_split': scope.int(hp.quniform('min_samples_split_gb', 2, 20, 1)),
        'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf_gb', 1, 10, 1)),
        'subsample': hp.uniform('subsample_gb', 0.6, 1.0)
    },
    
    'adaboost': {
        'n_estimators': scope.int(hp.quniform('n_estimators_ab', 50, 200, 1)),
        'learning_rate': hp.loguniform('learning_rate_ab', -3, 0),
    },
    
    'mlp': {
        'hidden_layer_sizes': hp.choice('hidden_layer_sizes_mlp', [(50,), (100,), (150,)]),
        'activation': hp.choice('activation_mlp', ['identity', 'logistic', 'tanh', 'relu']),
        'alpha': hp.loguniform('alpha_mlp', -5, 2),
        'learning_rate_init': hp.loguniform('learning_rate_init_mlp', -5, 0)
    },
    
    'xgboost': {
        'n_estimators': scope.int(hp.quniform('n_estimators_xgb', 50, 200, 1)),
        'max_depth': scope.int(hp.quniform('max_depth_xgb', 5, 20, 1)),
        'learning_rate': hp.loguniform('learning_rate_xgb', -3, 0),
        'subsample': hp.uniform('subsample_xgb', 0.6, 1.0),
        'gamma': hp.loguniform('gamma_xgb', -3, 3),
        'colsample_bytree': hp.uniform('colsample_bytree_xgb', 0.6, 1.0)
    }
}


if __name__ == '__main__':
    curr_dir = Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    hyperparameters_path = Path(home_dir.as_posix()+'/models/hyperparameters.pkl')
    pickle.dump(hyperparameters, open(hyperparameters_path, 'wb'))