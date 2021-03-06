import csv
from utils import *
from cross_validation import *

from sklearn.metrics import confusion_matrix
from sklearn.datasets import fetch_mldata

# Accuracy measured on cross validation
accuracy_cross_val = 0.9755166666666666
# SOTA accuracy
accuracy_SOTA = 0.9979

mldata = fetch_mldata("MNIST original")
x, y = mldata['data'][:, :].astype(np.float64), mldata['target'].astype(np.int)
x_train, x_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]


model = KNNClassifier(k=4, strategy='brute', metric='cosine', weights=True)
model.fit(x_train, y_train)
y_predict = model.predict(x_test)

confusion_mtx = confusion_matrix(y_test, y_predict)
print("Test accuracy: ", accuracy(y_test, y_predict))
print("SOTA accuracy: ", accuracy_SOTA)
print("Cross validation accuracy: ", accuracy_cross_val)

head_line = ['Test', 'SOTA', 'Cross validation']
with open('./experiment_4/accuracies.csv', 'w', newline='') as file:
    writer = csv.writer(file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(head_line)
    writer.writerow([str(accuracy(y_test, y_predict)), str(accuracy_SOTA), str(accuracy_cross_val)])

head_line = ['-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
with open('./experiment_4/confusion_matrix.csv', 'w', newline='') as file:
    writer = csv.writer(file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(head_line)
    for idx in range(confusion_mtx.shape[0]):
        writer.writerow([str(idx)] + [str(confusion_mtx[idx, jdx]) for jdx in range(confusion_mtx.shape[1])])


h, w = 5, 5
x_wrong_classified = x_test[y_test != y_predict]
y_test_wrong_classified = y_test[y_test != y_predict]
y_predict_wrong_classified = y_predict[y_test != y_predict]
subsample = np.random.randint(0, x_wrong_classified.shape[0], [h * w])
fig, axes = plt.subplots(h, w, figsize=(10, 10))
for idx in range(h):
    for jdx in range(w):
        ax = axes[idx][jdx]
        ax.imshow(x_wrong_classified[subsample[idx * h + jdx]].reshape(28, 28), cmap="Greys")
        ax.axis("off")
        ax.set_title(
            label='{0} ({1})'.format(
                y_test_wrong_classified[subsample[idx * h + jdx]],
                y_predict_wrong_classified[subsample[idx * h + jdx]]
            ),
            fontdict={'fontsize': 22}
        )
plt.savefig('./experiment_4/wrong_classifies.png')
