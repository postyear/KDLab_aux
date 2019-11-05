import numpy as np

from sklearn.ensemble import RandomForestClassifier


class RFOEXMetric:
    def countRFOEX(self, df, classes, params):
        values = np.empty([0])
        """
        estimator = RandomForestClassifier(params)
        estimator.fit(df.values)
        trees = estimator.estimators_
        matrix = np.zeros((len(df), len(df)), float)

        for tree in trees:

            predicted = np.empty(len(df), float)
            for index, row in df.values:
                _, predicted[index] = tree.predict(row)





            
            value = 0.0
            for neighbor in neighbors:
                if classes[index] != classes[neighbor]:
                    value += 1.0
            values = np.append(values, np.full((1, 1), value / k))
        """

        return values
