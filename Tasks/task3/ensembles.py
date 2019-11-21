import time

import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error


def rmse(y_1, y_2):
    return mean_squared_error(y_1, y_2) ** 0.5


class RandomForestMSE:
    def __init__(
            self, n_estimators, max_depth=None, feature_subsample_size=None,
            base_estimator=DecisionTreeRegressor, **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.

        max_depth : int
            The maximum depth of the tree. If None then there is no limits.

        feature_subsample_size : float
            The size of feature set for each tree. If None then use recommendations.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.base_estimator = base_estimator
        self.trees_parameters = trees_parameters
        self.estimators = []
        self.indices = []

    def fit(self, x, y, x_test=None, y_test=None, verbose=False):
        """
        x : numpy ndarray
            Array of size n_objects, n_features

        y : numpy ndarray
            Array of size n_objects
        """
        feature_subsample_size = x.shape[1] // 3 if self.feature_subsample_size is None else self.feature_subsample_size
        rmse_train = []
        rmse_test = []
        times = []
        start_time = time.time()

        predictions_base = 0.
        predictions_base_test = 0.

        for idx in range(self.n_estimators):
            if verbose:
                if x_test is not None and idx > 0:
                    print('idx: ', idx, rmse_train[-1], rmse_test[-1])
                elif idx > 0:
                    print('idx: ', idx, rmse_train[-1])

            self.indices.append(np.random.choice(x.shape[1], feature_subsample_size, replace=False))

            self.estimators.append(
                self.base_estimator(
                    max_depth=self.max_depth,
                    **self.trees_parameters
                ).fit(x[:, self.indices[-1]], y)
            )

            predictions_base += self.estimators[-1].predict(x[:, self.indices[-1]])

            rmse_train.append(rmse(predictions_base / len(self.estimators), y))
            if x_test is not None:
                predictions_base_test += self.estimators[-1].predict(x_test[:, self.indices[-1]])
                rmse_test.append(rmse(predictions_base_test / len(self.estimators), y_test))
            times.append(time.time() - start_time)
        if x_test is not None:
            return rmse_train, rmse_test, times
        else:
            return rmse_train, times

    def predict(self, x):
        """
        x : numpy ndarray
            Array of size n_objects, n_features
            
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        return np.mean((tree.predict(x[:, indices]) for indices, tree in zip(self.indices, self.estimators)))


class GradientBoostingMSE:
    def __init__(
            self, n_estimators, learning_rate=0.1, max_depth=5, feature_subsample_size=None,
            base_estimator=DecisionTreeRegressor, **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.
        
        learning_rate : float
            Use learning_rate * gamma instead of gamma

        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        
        feature_subsample_size : float
            The size of feature set for each tree. If None then use recommendations.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.base_estimator = base_estimator
        self.trees_parameters = trees_parameters
        self.estimators = []
        self.weights = []
        self.indices = []

    def fit(self, x, y, x_test=None, y_test=None, verbose=False):
        """
        x : numpy ndarray
            Array of size n_objects, n_features
            
        y : numpy ndarray
            Array of size n_objects
        """
        feature_subsample_size = x.shape[1] // 3 if self.feature_subsample_size is None else self.feature_subsample_size
        rmse_train = []
        rmse_test = []
        times = []
        start_time = time.time()

        self.indices.append(np.random.choice(x.shape[1], feature_subsample_size, replace=False))
        self.estimators.append(
            self.base_estimator(
                max_depth=1,
                **self.trees_parameters
            ).fit(x[:, self.indices[-1]], y)
        )
        self.weights.append(1.)

        predictions_base = self.predict(x)
        rmse_train.append(rmse(predictions_base, y))

        predictions_base_test = None
        if x_test is not None:
            predictions_base_test = self.predict(x_test)
            rmse_test.append(rmse(predictions_base_test, y_test))
        times.append(time.time() - start_time)

        for idx in range(self.n_estimators - 1):
            if verbose:
                if x_test is not None:
                    print('idx: ', idx, rmse_train[-1], rmse_test[-1])
                else:
                    print('idx: ', idx, rmse_train[-1])

            self.indices.append(np.random.choice(x.shape[1], feature_subsample_size, replace=False))
            gradients = 2.0 * (y - predictions_base)

            self.estimators.append(
                self.base_estimator(
                    max_depth=self.max_depth,
                    **self.trees_parameters
                ).fit(x[:, self.indices[-1]], gradients)
            )

            prediction_i = self.estimators[-1].predict(x[:, self.indices[-1]])
            betta = (
                    (prediction_i * (y - predictions_base)).sum() /
                    (prediction_i ** 2).sum()
            )
            self.weights.append(self.learning_rate * betta)
            predictions_base += self.weights[-1] * self.estimators[-1].predict(x[:, self.indices[-1]])

            rmse_train.append(rmse(predictions_base, y))
            if x_test is not None:
                predictions_base_test += self.weights[-1] * self.estimators[-1].predict(x_test[:, self.indices[-1]])
                rmse_test.append(rmse(predictions_base_test, y_test))
            times.append(time.time() - start_time)
        if x_test is not None:
            return rmse_train, rmse_test, times
        else:
            return rmse_train, times

    def predict(self, x):
        """
        x : numpy ndarray
            Array of size n_objects, n_features
            
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        result = 0.
        for weight, indices, tree in zip(self.weights, self.indices, self.estimators):
            result += weight * tree.predict(x[:, indices])
        return result
