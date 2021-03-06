import copy

import numpy as np

import itertools


class MulticlassStrategy:
    def __init__(self, classifier, mode, **kwargs):
        print(mode)
        """
        Инициализация мультиклассового классификатора
        
        classifier - базовый бинарный классификатор
        
        mode - способ решения многоклассовой задачи,
        либо 'one_vs_all', либо 'all_vs_all'
        
        **kwargs - параметры классификатор
        """
        self.n_classes = None
        self.classifier = classifier
        self.mode = mode
        self.kwargs = kwargs
        self.classifiers = []

    def fit(self, x, y):
        """
        Обучение классификатора
        """
        self.n_classes = np.max(y) + 1
        if self.mode == 'one_vs_all':
            for n_class in range(self.n_classes):
                print("Class {0} vs All".format(str(n_class)))
                y_tmp = copy.deepcopy(y)
                y_tmp[y != n_class] = -1
                y_tmp[y == n_class] = 1
                self.classifiers.append(self.classifier(**self.kwargs))
                self.classifiers[-1].fit(x, y_tmp)
        else:
            for (idx, jdx) in itertools.combinations(range(self.n_classes), 2):
                print("Class {0} vs Class {1}".format(str(idx), str(jdx)))
                y_tmp = y[(y == idx) + (y == jdx)]
                y_tmp[y_tmp == idx] = -1
                y_tmp[y_tmp == jdx] = 1
                self.classifiers.append(self.classifier(**self.kwargs))
                self.classifiers[-1].fit(x[(y == idx) + (y == jdx)], y_tmp)

    def predict(self, x, y=None):
        """
        Выдача предсказаний классификатором
        """
        if self.mode == 'one_vs_all':
            predictions = np.concatenate(
                [classifier.predict_proba(x)[:, 1].reshape([-1, 1]) for classifier in self.classifiers],
                axis=1
            )
        else:
            predictions = np.zeros([x.shape[0], self.n_classes])
            for kdx, (idx, jdx) in enumerate(itertools.combinations(range(self.n_classes), 2)):
                idx_class = self.classifiers[kdx].predict(x) == -1
                predictions[idx_class, idx] += 1
                predictions[~idx_class, jdx] += 1
        return np.argmax(predictions, axis=1)
