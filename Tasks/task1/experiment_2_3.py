import csv
import time
from utils import *
from cross_validation import *

from sklearn.datasets import fetch_mldata


mldata = fetch_mldata("MNIST original")
x, y = mldata['data'][:, :].astype(np.float64), mldata['target'].astype(np.int)
x_train, _, y_train, _ = x[:60000], x[60000:], y[:60000], y[60000:]
x_train, y_train = shuffle_data(x_train, y_train)

k_list = range(1, 11)

times = []


st = time.time()
results_euclidean_no_weights = knn_cross_val_score(
    x_train, y_train, k_list,
    score='accuracy', strategy='brute', metric='euclidean', weights=False
)
times.append(time.time() - st)
results_euclidean_no_weights_list = [list_mean(l) for _, l in sorted(results_euclidean_no_weights.items())]


st = time.time()
results_euclidean_use_weights = knn_cross_val_score(
    x_train, y_train, k_list,
    score='accuracy', strategy='brute', metric='euclidean', weights=True
)
times.append(time.time() - st)
results_euclidean_use_weights_list = [list_mean(l) for _, l in sorted(results_euclidean_use_weights.items())]


st = time.time()
results_cosine_no_weights = knn_cross_val_score(
    x_train, y_train, k_list,
    score='accuracy', strategy='brute', metric='cosine', weights=False
)
times.append(time.time() - st)
results_cosine_no_weights_list = [list_mean(l) for _, l in sorted(results_cosine_no_weights.items())]


st = time.time()
results_cosine_use_weights = knn_cross_val_score(
    x_train, y_train, k_list,
    score='accuracy', strategy='brute', metric='cosine', weights=True
)
times.append(time.time() - st)
results_cosine_use_weights_list = [list_mean(l) for _, l in sorted(results_cosine_use_weights.items())]


head_line = ['K'] + ['Euclidean; No weights', 'Euclidean; Use weights', 'Cosine; No weights', 'Cosine; Use weights']
with open('./experiment_2_3/scores.csv', 'w', newline='') as file:
    writer = csv.writer(file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(head_line)
    for idx, k in enumerate(k_list):
        writer.writerow([k] + [
            results_euclidean_no_weights_list[idx],
            results_euclidean_use_weights_list[idx],
            results_cosine_no_weights_list[idx],
            results_cosine_use_weights_list[idx]
        ])

head_line = ['Euclidean; No weights', 'Euclidean; Use weights', 'Cosine; No weights', 'Cosine; Use weights']
with open('./experiment_2_3/times.csv', 'w', newline='') as file:
    writer = csv.writer(file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(head_line)
    writer.writerow([str(times[0]), str(times[1]), str(times[2]), str(times[3])])
