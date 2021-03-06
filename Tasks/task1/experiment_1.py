import csv
import time
from utils import *
from cross_validation import *
from nearest_neighbors import KNNClassifier

from sklearn.datasets import fetch_mldata


mldata = fetch_mldata("MNIST original")
x, y = mldata['data'][:, :].astype(np.float64), mldata['target'].astype(np.int)
x_train, x_test, y_train, _ = x[:60000], x[60000:], y[:60000], y[60000:]


fea_subsample_sizes = (10, 20, 100)
strategies = ('my_own', 'brute', 'kd_tree', 'ball_tree')
metrics = ('euclidean', )

times_fit = defaultdict(list)
times_predict = defaultdict(list)
times_fit_predict = defaultdict(list)
for fea_subsample_size in fea_subsample_sizes:
    fea_idxs = np.random.randint(0, x.shape[1], [fea_subsample_size])
    x_train_tmp, x_test_tmp = x_train[:, fea_idxs], x_test[:, fea_idxs]
    for strategy in strategies:
        for metric in metrics:
            delta_1 = []
            delta_2 = []
            for _ in range(1):
                model = KNNClassifier(
                    k=5, strategy=strategy, metric=metric
                )
                st = time.time()
                model.fit(x_train_tmp, y_train)
                st_predict = time.time()
                model.find_kneighbors(x_test_tmp, return_distance=False)
                en = time.time()
                delta_1.append(st_predict - st)
                delta_2.append(en - st_predict)
            print("Stratery: ", strategy, " Metric: ", metric)
            print("Fit: ", list_mean(delta_1))
            print("Predict: ", list_mean(delta_2))
            print("Fit + Predict: ", list_mean(delta_1) + list_mean(delta_2))
            times_fit[fea_subsample_size].append(list_mean(delta_1))
            times_predict[fea_subsample_size].append(list_mean(delta_2))
            times_fit_predict[fea_subsample_size].append(list_mean(delta_1) + list_mean(delta_2))


head_line = ['N features'] + [strategy for strategy in strategies]

with open('./experiment_1/time_fit.csv', 'w', newline='') as file:
    writer = csv.writer(file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(head_line)
    for fea_subsample_size in fea_subsample_sizes:
        writer.writerow([fea_subsample_size] + times_fit[fea_subsample_size])

with open('./experiment_1/time_predict.csv', 'w', newline='') as file:
    writer = csv.writer(file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(head_line)
    for fea_subsample_size in fea_subsample_sizes:
        writer.writerow([fea_subsample_size] + times_predict[fea_subsample_size])

with open('./experiment_1/time_fit_predict.csv', 'w', newline='') as file:
    writer = csv.writer(file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(head_line)
    for fea_subsample_size in fea_subsample_sizes:
        writer.writerow([fea_subsample_size] + times_fit_predict[fea_subsample_size])
