import csv
from utils import *
from cross_validation import *

from sklearn.metrics import confusion_matrix
from sklearn.datasets import fetch_mldata

mldata = fetch_mldata("MNIST original")
x, y = mldata['data'][:, :].astype(np.float64), mldata['target'].astype(np.int)
x_train, x_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]


angle, shift, sigma = 10, 1, 1.

model_rotation = KNNClassifier(
    k=4, strategy='brute', metric='cosine', weights=True
)
model_rotation.fit(*augment_data(x_train, y_train, angle=angle))
y_predict_rotation = model_rotation.predict(x_test)
rotation_score = accuracy(y_test, y_predict_rotation)


model_shift = KNNClassifier(
    k=4, strategy='brute', metric='cosine', weights=True
)
model_shift.fit(*augment_data(x_train, y_train, x_shift=shift, y_shift=shift))
y_predict_shift = model_shift.predict(x_test)
shift_score = accuracy(y_test, y_predict_shift)


blur_scores = []
model_blur = KNNClassifier(
    k=4, strategy='brute', metric='cosine', weights=True
)
model_blur.fit(*augment_data(x_train, y_train, sigma=sigma))
y_predict_blur = model_blur.predict(x_test)
blur_score = accuracy(y_test, y_predict_blur)


confusion_mtx_rotation = confusion_matrix(y_test, y_predict_rotation)
confusion_mtx_shift = confusion_matrix(y_test, y_predict_shift)
confusion_mtx_blur = confusion_matrix(y_test, y_predict_blur)

print("Test accuracy (rotation): ", rotation_score)
print("Test accuracy (shift): ", shift_score)
print("Test accuracy (blur): ", blur_score)


def save_results(y_predict, confusion_mtx, suffics):
    head_line = ['-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    with open('./experiment_5/confusion_matrix_' + suffics + '.csv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(head_line)
        for idx in range(confusion_mtx.shape[0]):
            writer.writerow(
                [str(idx)] + [str(confusion_mtx[idx, jdx]) for jdx in range(confusion_mtx.shape[1])]
            )

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
    plt.savefig('./experiment_5/wrong_classifies_' + suffics + '.png')


save_results(y_predict_rotation, confusion_mtx_rotation, 'rotation')
save_results(y_predict_shift, confusion_mtx_shift, 'shift')
save_results(y_predict_blur, confusion_mtx_blur, 'blur')
