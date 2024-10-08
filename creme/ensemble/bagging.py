import collections
import copy
import statistics

from sklearn import utils

from .. import base


__all__ = ['BaggingClassifier', 'BaggingRegressor']


class BaseBagging(collections.UserList, base.Wrapper):

    def __init__(self, model, n_models=10, random_state=None):
        super().__init__()
        self.model = model
        self.extend([copy.deepcopy(model) for _ in range(n_models)])
        self.rng = utils.check_random_state(random_state)

    @property
    def _model(self):
        return self.model

    def fit_one(self, x, y):

        for model in self:
            for _ in range(self.rng.poisson(1)):
                model.fit_one(x, y)

        return self


class BaggingClassifier(BaseBagging, base.Classifier):
    """Bagging for classification.

    For each incoming observation, each model's ``fit_one`` method is called ``k`` times where
    ``k`` is sampled from a Poisson distribution of parameter 1. ``k`` thus has a 36% chance of
    being equal to 0, a 36% chance of being equal to 1, an 18% chance of being equal to 2, a 6%
    chance of being equal to 3, a 1% chance of being equal to 4, etc. You can do
    ``scipy.stats.poisson(1).pmf(k)`` to obtain more detailed values.

    Parameters:
        classifier (BinaryClassifier or MultiClassifier): The classifier to bag.

    Example:

        In the following example three logistic regressions are bagged together. The performance is
        slightly better than when using a single logistic regression.

        ::

            >>> from creme import compose
            >>> from creme import ensemble
            >>> from creme import linear_model
            >>> from creme import metrics
            >>> from creme import model_selection
            >>> from creme import optim
            >>> from creme import preprocessing
            >>> from creme import stream
            >>> from sklearn import datasets

            >>> X_y = stream.iter_sklearn_dataset(
            ...     dataset=datasets.load_breast_cancer(),
            ...     shuffle=True,
            ...     random_state=42
            ... )
            >>> model = ensemble.BaggingClassifier(
            ...     model=(
            ...         preprocessing.StandardScaler() |
            ...         linear_model.LogisticRegression()
            ...     ),
            ...     n_models=3,
            ...     random_state=42
            ... )
            >>> metric = metrics.F1()

            >>> model_selection.online_score(X_y, model, metric)
            F1: 0.970547

            >>> print(model)
            BaggingClassifier(StandardScaler | LogisticRegression)

    References:
        1. `Online Bagging and Boosting <https://ti.arc.nasa.gov/m/profile/oza/files/ozru01a.pdf>`_

    """

    def predict_proba_one(self, x):
        """Averages the predictions of each classifier."""

        y_pred = collections.defaultdict(float)

        # Sum the predictions
        for classifier in self:
            for label, proba in classifier.predict_proba_one(x).items():
                y_pred[label] += proba

        # Divide by the number of predictions
        for label in y_pred:
            y_pred[label] /= len(self)

        return dict(y_pred)


class BaggingRegressor(BaseBagging, base.Regressor):
    """Bagging for regression.

    For each incoming observation, each model's ``fit_one`` method is called ``k`` times where
    ``k`` is sampled from a Poisson distribution of parameter 1. ``k`` thus has a 36% chance of
    being equal to 0, a 36% chance of being equal to 1, an 18% chance of being equal to 2, a 6%
    chance of being equal to 3, a 1% chance of being equal to 4, etc. You can do
    ``scipy.stats.poisson(1).pmf(k)`` for more detailed values.

    Parameters:
        base_regressor (Regressor): The regressor to bag.

    Example:

        In the following example three logistic regressions are bagged together. The performance is
        slightly better than when using a single logistic regression.

        ::

            >>> from creme import compose
            >>> from creme import ensemble
            >>> from creme import linear_model
            >>> from creme import metrics
            >>> from creme import model_selection
            >>> from creme import optim
            >>> from creme import preprocessing
            >>> from creme import stream
            >>> from sklearn import datasets

            >>> X_y = stream.iter_sklearn_dataset(
            ...     dataset=datasets.load_boston(),
            ...     shuffle=True,
            ...     random_state=42
            ... )
            >>> model = preprocessing.StandardScaler()
            >>> model |= ensemble.BaggingRegressor(
            ...     model=linear_model.LinearRegression(intercept_lr=0.1),
            ...     n_models=3,
            ...     random_state=42
            ... )
            >>> metric = metrics.MAE()

            >>> model_selection.online_score(X_y, model, metric)
            MAE: 4.26101

    References:
        1. `Online Bagging and Boosting <https://ti.arc.nasa.gov/m/profile/oza/files/ozru01a.pdf>`_

    """

    def predict_one(self, x):
        """Averages the predictions of each regressor."""
        return statistics.mean((regressor.predict_one(x) for regressor in self))
