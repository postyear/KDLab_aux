import time

import numpy as np

from sklearn.ensemble import RandomForestClassifier
import pyximport; pyximport.install()

class RFOEXMetric:

    def countRFOEX(self, df, classes, super):

        start = time.time()

        inst = len(df)
        values = np.empty(inst)

        estimator = RandomForestClassifier(**super.settings)
        estimator.fit(df.values, classes)
        print('fit complete')

        cdef int i, j

        trees = estimator.estimators_
        cdef short[:,:] matrix = np.empty((inst, inst), np.short)

        cdef short[:,:] predicted = np.empty((len(trees), inst), np.short)

        for i in range(len(trees)):
            pred = trees[i].predict(df)
            for j in range(inst):
                #aux = trees[i].predict(df)
                predicted[i][j] = pred[j]
        print('Trees predicted')

        for i in range(inst):
            for j in range(i, inst):
                sim = 0
                for t in range(len(trees)):
                    if predicted[t][i] == predicted[t][j]:
                        sim += 1
                matrix[i, j] = sim
                matrix[j, i] = sim
            if i % 100 == 0:
                print('Intance %d done' % i)
        print('Matrix produced')

        _, cls_num = np.unique(classes, return_inverse=True)
        cdef int[:] clss = cls_num.astype(int)

        #cdef int cl
        cls_indices = {}
        noncls_indices = {}
        """
        for cls in np.unique(clss):
            cls_indices[cls] = np.empty(0)
            noncls_indices[cls] = np.empty(0)
        for i in range(len(inst)):
            cl = clss[i]
            for cls in np.unique(clss):
                if cl != cls:
                    noncls_indices:
        """

        for cls in np.unique(clss):
            cls_indices[cls] = [i for i in range(len(df)) if clss[i] == cls]
            noncls_indices[cls] = [i for i in range(len(df)) if clss[i] != cls]\

        print('indices split')

        cdef double[:] proxsuminv = np.empty(inst)
        cdef double[:] wrong_cl = np.empty(inst)
        cdef double[:] general_out = np.empty(inst)

        cdef double[:] row
        cdef int s

        for i in range(inst):

            point1 = time.process_time_ns()

            row = matrix.base[i] / len(trees)

            point2 = time.process_time_ns()

            # ProxSumInverse
            indices = cls_indices[clss[i]]
            proxsuminv[i] = 1 / sum([row[j] for j in indices])

            # TopC instances
            num_inst = len(cls_indices[clss[i]])
            top_c = np.argsort(np.multiply(row, -1))[num_inst:]

            # P_wrong_cl
            s = 0
            for j in top_c:
                if cls_num[i] != cls_num[j]:
                    s += 1
            wrong_cl[i] = s / num_inst

            # P_general_out
            general_out[i] = (num_inst - sum(row[j] for j in top_c)) / num_inst
            #print('Base calculated for instance %d' % i)

            if i % 100 == 0:
                print('Intance %d done' % i)

            """
            if i == 0:
                print('1 to 2', point2 - point1)
                print('2 to 3', point3 - point2)
                print('3 to 4', point4 - point3)
                print('4 to 5', point5 - point4)
                print('5 to 6', point6 - point5)
            """

        medians = {}
        means = {}
        for cls in np.unique(clss):
            medians[cls] = np.median(proxsuminv.base[cls_indices[cls]])
            means[cls] = np.mean(abs(proxsuminv.base[cls_indices[cls]] - medians[cls]))
        const = np.max(proxsuminv) / 4

        fo1 = [(proxsuminv[i] - medians[clss[i]]) / means[clss[i]] for i in range(inst)]
        fo2 = wrong_cl * const
        fo3 = general_out * const

        values = np.add(np.add(fo1, fo2), fo3)

        end = time.time()
        print('Total time:')
        print(end - start)

        return values
