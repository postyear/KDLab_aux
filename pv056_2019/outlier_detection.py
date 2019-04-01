from __future__ import absolute_import

import numpy as np
import pandas as pd
from typing import Any, Dict
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from CL import CLMetric
from CLD import CLDMetric

DETECTORS: Dict[str, Any] = {}


class AbstractDetector:
    name: str
    data_type: str
    values: np.array

    def __init__(self, **settings):
        self.settings = settings

    def compute_scores(self, dataframe: pd.DataFrame, classes: np.array):
        raise NotImplementedError()


def detector(cls):
    DETECTORS.update({cls.name: cls})
    return cls


@detector
class LOF(AbstractDetector):
    name = "LOF"
    data_type = "REAL"

    def compute_scores(self, dataframe: pd.DataFrame, classes: np.array):
        bin_dataframe = dataframe._binarize_categorical_values()

        self.clf = LocalOutlierFactor(**self.settings)
        self.clf.fit(bin_dataframe.values)
        self.values = self.clf._decision_function(bin_dataframe.values)
        return self


@detector
class NN(AbstractDetector):
    name = "NearestNeighbors"
    data_type = "REAL"

    def compute_scores(self, dataframe: pd.DataFrame, classes: np.array):
        bin_dataframe = dataframe._binarize_categorical_values()
        if "n_neighbors" in self.settings:
            self.settings["n_neighbors"] = int(self.settings["n_neighbors"])
        self.clf = NearestNeighbors(**self.settings)
        self.clf.fit(bin_dataframe.values)
        distances, _ = self.clf.kneighbors()
        self.values = np.mean(distances, axis=1)
        return self


@detector
class IsoForest(AbstractDetector):
    name = "IsolationForest"
    data_type = "REAL"

    def compute_scores(self, dataframe: pd.DataFrame, classes: np.array):
        bin_dataframe = dataframe._binarize_categorical_values()

        self.clf = IsolationForest(**self.settings)
        self.clf.fit(bin_dataframe.values)
        self.values = self.clf.decision_function(bin_dataframe.values)
        return self


@detector
class CL(AbstractDetector):
    name = "ClassLikelihood"
    data_type = "REAL"

    def compute_scores(self, dataframe: pd.DataFrame, classes: np.array):

        self.clf = CLMetric(self.settings)
        self.values = self.clf.findLikelihood(dataframe, classes)
        return self


@detector
class CLD(AbstractDetector):
    name = "ClassLikelihoodDifference"
    data_type = "REAL"

    def compute_scores(self, dataframe: pd.DataFrame, classes: np.array):

        self.clf = CLDMetric(self.settings)
        self.values = self.clf.findLikelihood(dataframe, classes)
        return self
