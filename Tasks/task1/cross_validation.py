import numpy as np
from collections import defaultdict
from nearest_neighbors import KNNClassifier


def accuracy(y, y_predict):
    return float(np.sum(y == y_predict)) / float(y.shape[0])


def kfold(n, n_folds):
    result = []
    q = n // n_folds
    r = n % n_folds
    left_border = 0
    for idx in range(n_folds):
        chunk_size = q + 1 if idx < r else q
        right_border = left_border + chunk_size
        result.append(
            (
                np.concatenate((np.arange(0, left_border), np.arange(right_border, n))),
                np.arange(left_border, right_border)
            )
        )
        left_border += chunk_size
    return result


def knn_cross_val_score(x, y, k_list, score='accuracy', cv=None, evaluate_time=False, **kwargs):
    result_list = defaultdict(list)
    result_array = dict()

    n_classes = np.max(y) + 1

    if score == 'accuracy':
        scorer = accuracy
    else:
        raise NotImplementedError

    if cv is None:
        cv = kfold(x.shape[0], 3)

    use_weights = kwargs['weights'] if 'weights' in kwargs.keys() else True
    classifier = KNNClassifier(
        k=k_list[-1],
        strategy=kwargs['strategy'] if 'strategy' in kwargs.keys() else 'my_own',
        metric=kwargs['metric'] if 'metric' in kwargs.keys() else 'euclidean',
        weights=use_weights,
        test_block_size=kwargs['test_block_size'] if 'test_block_size' in kwargs.keys() else 100
    )

    for train_idxs, test_idxs in cv:
        classifier.fit(x[train_idxs], y[train_idxs])
        if use_weights:
            distances, idx_neighbors = classifier.find_kneighbors(x[test_idxs], return_distance=True)
        else:
            idx_neighbors = classifier.find_kneighbors(x[test_idxs], return_distance=False)

        weights = 1. / (distances + 1e-5) if use_weights else None

        class_scores = np.zeros([test_idxs.shape[0], n_classes], dtype=np.float64)
        neighbor_classes = y[train_idxs][idx_neighbors]
        for k in range(k_list[-1]):
            for idx in range(test_idxs.shape[0]):
                class_scores[idx, neighbor_classes[idx, k]] += (weights[idx, k] if use_weights else 1.)
            y_predict = np.argmax(class_scores, axis=1)
            if k + 1 in k_list:
                result_list[k + 1].append(scorer(y[test_idxs], y_predict))

    for key in result_list.keys():
        result_array[key] = np.array(result_list[key], dtype=np.float64)

    return result_array
