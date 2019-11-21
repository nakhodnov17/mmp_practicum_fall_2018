# import time
# from utils import *
# from cross_validation import *
# from sklearn.datasets import fetch_mldata
# from nearest_neighbors import KNNClassifier
# from sklearn.neighbors import KNeighborsClassifier
#
# if __name__ == '__main__':
#     model = KNNClassifier(k=5, strategy='brute', metric='cosine', weights=False, test_block_size=10000)
#     # model = KNeighborsClassifier(
#     #     n_neighbors=5, algorithm='brute',
#     #     metric='euclidean', weights=lambda w: 1. / (1e-5 + w)
#     # )
#     mldata = fetch_mldata("MNIST original")
#     x_, y_ = mldata['data'][:, :].astype(np.float64), mldata['target'].astype(np.int)
#     x_train, x_test, y_train, y_test = x_[:1000], x_[60000:], y_[:1000], y_[60000:]
#
#     # x_train, y_train = augment_data(x_train, y_train, rotate_img, 15, True)
#     # x_train, y_train = shuffle_data(x_train, y_train)
#     x_train, y_train = x_train, y_train
#
#     # st = time.time()
#     # res = knn_cross_val_score(x_train, y_train, range(10, 11), cv=kfold(60000, 2), strategy='my_own')
#     # print(res)
#     # print("cross_val_score time: ", time.time() - st)
#
#     st = time.time()
#     model.fit(x_train, y_train)
#     print("fit time: ", time.time() - st)
#
#     st = time.time()
#     y_predict = model.predict(x_test)
#     print("predict time: ", time.time() - st)
#
#     print(y_test)
#     print(y_predict)
#     print(np.sum(y_predict != y_test))
#
#     # 274
#     # 309

import numpy as np

np.array()