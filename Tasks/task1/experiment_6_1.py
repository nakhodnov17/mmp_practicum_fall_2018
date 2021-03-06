import csv
from utils import *
from cross_validation import *

from sklearn.datasets import fetch_mldata


mldata = fetch_mldata("MNIST original")
x, y = mldata['data'][:, :].astype(np.float64), mldata['target'].astype(np.int)
x_train, x_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]
x_train, y_train = shuffle_data(x_train, y_train)


angles = (5, 10, 15)
shifts = (1, 2, 3)
sigmas = (0.5, 1., 1.5)


rotation_scores = []
for angle in angles:
    model = KNNClassifier(
        k=4, strategy='brute', metric='cosine', weights=True,
        augment_test_data=True, angle=angle
    )
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    rotation_scores.append(accuracy(y_test, y_predict))
print(rotation_scores)


shift_scores = []
for shift in shifts:
    model = KNNClassifier(
        k=4, strategy='brute', metric='cosine', weights=True,
        augment_test_data=True, x_shift=shift, y_shift=shift
    )
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    shift_scores.append(accuracy(y_test, y_predict))
print(shift_scores)


blur_scores = []
for sigma in sigmas:
    model = KNNClassifier(
        k=4, strategy='brute', metric='cosine', weights=True,
        augment_test_data=True, sigma=sigma
    )
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    blur_scores.append(accuracy(y_test, y_predict))
print(blur_scores)


head_line = [str(angle) for angle in angles]
with open('./experiment_6/rotation_scores.csv', 'w', newline='') as file:
    writer = csv.writer(file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(head_line)
    writer.writerow(str(score) for score in rotation_scores)

head_line = [str(shift) for shift in shifts]
with open('./experiment_6/shift_scores.csv', 'w', newline='') as file:
    writer = csv.writer(file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(head_line)
    writer.writerow(str(score) for score in shift_scores)

head_line = [str(sigma) for sigma in sigmas]
with open('./experiment_6/blur_scores.csv', 'w', newline='') as file:
    writer = csv.writer(file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(head_line)
    writer.writerow(str(score) for score in blur_scores)
