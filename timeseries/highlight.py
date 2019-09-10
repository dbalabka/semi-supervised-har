from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from .helper import NoFitMixin


class TimeseriesHighlightTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self._max = self._min = 0
        pass

    def highlight_function(self, pair):
        return (pair[0] - 1. if pair[1] < 0 else pair[0])

    def fit(self, X, y=None):
        self._max = np.max(X)
        self._min = np.min(X)
        return self

    def transform(self, timeseries):
        return np.apply_along_axis(
            self.highlight_function,
            axis=0,
            arr=np.array([timeseries, np.gradient(timeseries, axis=1)])
        )

    def plot(self, timeseries):
        pass