import numpy as np
from utils import *
from sklearn.neighbors import NearestNeighbors
from distances import euclidean_distance, cosine_distance


class KNNClassifier:
    def __init__(
            self, k=1, strategy='brute', metric='euclidean',
            weights=False, test_block_size=None, epsilon=1e-5,
            augment_test_data=False, angle=None, x_shift=None, y_shift=None, sigma=None
    ):
        self.k = k
        self.strategy = strategy
        self.metric = metric
        self.weights = weights
        self.test_block_size = test_block_size
        self.epsilon = epsilon

        self.augment_test_data = augment_test_data
        self.angle = angle
        self.x_shift = x_shift
        self.y_shift = y_shift
        self.sigma = sigma

        self.model = None

        self.x = None
        self.y = None
        self.n_classes = None

        if self.metric == 'euclidean':
            self.pairwise_dist = euclidean_distance
        elif self.metric == 'cosine':
            self.pairwise_dist = cosine_distance
        else:
            raise NotImplementedError

        self._find_kneighbors_chunk = self._find_kneighbors_chunk

    def fit(self, x, y):
        self.y = y.astype(np.int)
        self.n_classes = np.max(y) + 1

        if self.strategy != 'my_own':
            self.model = NearestNeighbors(
                n_neighbors=self.k,
                algorithm=self.strategy,
                metric=self.metric,
            )
            self.model.fit(x)
            self._find_kneighbors_chunk = self.model.kneighbors
        else:
            self.x = x

    def find_kneighbors(self, x, return_distance):
        distances = np.empty([x.shape[0], self.k], dtype=np.float64) if return_distance else None
        idx_neighbors = np.empty([x.shape[0], self.k], dtype=np.int)

        if self.test_block_size is None:
            real_test_block_size = x.shape[0]
        else:
            real_test_block_size = min(self.test_block_size, x.shape[0])
        r = (real_test_block_size - x.shape[0] % real_test_block_size) % real_test_block_size
        n = x.shape[0] // real_test_block_size + (1 if r else 0)
        left_border = 0
        for idx in range(n):
            chunk_size = real_test_block_size - 1 if idx < r else real_test_block_size
            right_border = left_border + chunk_size

            if return_distance:
                (distances_chunk, idx_neighbors_chunk) = \
                    self._find_kneighbors_chunk(x[left_border:right_border], return_distance=True)
            else:
                (distances_chunk, idx_neighbors_chunk) = \
                    None, self._find_kneighbors_chunk(x[left_border:right_border], return_distance=False)

            idx_neighbors[left_border:right_border] = idx_neighbors_chunk
            if return_distance:
                distances[left_border:right_border] = distances_chunk
            del idx_neighbors_chunk, distances_chunk

            left_border += chunk_size
        if return_distance:
            return distances, idx_neighbors
        else:
            return idx_neighbors

    def _find_kneighbors_chunk(self, x, return_distance):
        pairwise_dist = self.pairwise_dist(x, self.x)
        idx_neighbors = np.argsort(pairwise_dist, axis=1)[:, :self.k]
        if return_distance:
            return np.sort(pairwise_dist, axis=1)[:, :self.k], idx_neighbors
        else:
            return idx_neighbors

    def predict(self, x):
        y_predict = np.empty([x.shape[0]], dtype=np.int)

        if self.test_block_size is None:
            real_test_block_size = x.shape[0]
        else:
            real_test_block_size = min(self.test_block_size, x.shape[0])
        r = (real_test_block_size - x.shape[0] % real_test_block_size) % real_test_block_size
        n = x.shape[0] // real_test_block_size + (1 if r else 0)

        left_border = 0
        for idx in range(n):
            chunk_size = real_test_block_size - 1 if idx < r else real_test_block_size
            right_border = left_border + chunk_size

            y_predict_chunk = self._predict_chunk(x[left_border:right_border])
            y_predict[left_border:right_border] = y_predict_chunk
            del y_predict_chunk

            left_border += chunk_size
        return y_predict

    def _predict_chunk(self, x):
        if self.weights:
            if self.augment_test_data:
                x_augmented = augment_data(
                    x, angle=self.angle, x_shift=self.x_shift, y_shift=self.y_shift, sigma=self.sigma
                )

                if self.sigma is None:
                    distances_aug = np.empty([x.shape[0], 3 * self.k], dtype=np.float64)
                    idx_neighbors_aug = np.empty([x.shape[0], 3 * self.k], dtype=np.int)

                    (distances_aug[:, 2 * self.k:],
                     idx_neighbors_aug[:, 2 * self.k:]) = \
                        self._find_kneighbors_chunk(x_augmented[2 * x.shape[0]:], return_distance=True)
                else:
                    distances_aug = np.empty([x.shape[0], 2 * self.k], dtype=np.float64)
                    idx_neighbors_aug = np.empty([x.shape[0], 2 * self.k], dtype=np.int)

                (distances_aug[:, :self.k],
                 idx_neighbors_aug[:, :self.k]) = \
                    self._find_kneighbors_chunk(x_augmented[:x.shape[0]], return_distance=True)

                (distances_aug[:, self.k:2 * self.k],
                 idx_neighbors_aug[:, self.k:2 * self.k]) = \
                    self._find_kneighbors_chunk(x_augmented[x.shape[0]:2 * x.shape[0]], return_distance=True)

                distances = np.sort(distances_aug, axis=1)[:, :self.k]
                idx_neighbors = np.empty([x.shape[0], self.k], dtype=np.int)
                idx_quasi_neighbors = np.argsort(distances_aug, axis=1)
                for idx in range(x.shape[0]):
                    idx_neighbors[idx] = idx_neighbors_aug[idx, idx_quasi_neighbors[idx]][:self.k]
            else:
                distances, idx_neighbors = self._find_kneighbors_chunk(x, return_distance=True)
        else:
            idx_neighbors = self._find_kneighbors_chunk(x, return_distance=False)
            distances = None

        if self.weights:
            distances += self.epsilon
            np.divide(1., distances, out=distances)
            weights = distances
        else:
            weights = None

        neighbor_classes = self.y[idx_neighbors]
        if not self.weights:
            class_scores = np.apply_along_axis(np.bincount, axis=1, arr=neighbor_classes)
        else:
            class_scores = np.empty([x.shape[0], self.n_classes], dtype=np.float64)
            for idx in range(x.shape[0]):
                class_scores[idx] = np.bincount(neighbor_classes[idx], weights=weights[idx], minlength=self.n_classes)
        y_predict = np.argmax(class_scores, axis=1)
        return y_predict
