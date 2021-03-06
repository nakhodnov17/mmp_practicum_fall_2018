import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, shift
from scipy.ndimage.filters import gaussian_filter


def plot_img(x):
    if len(x.shape) > 1:
        plt.imshow(x, cmap='Greys')
    else:
        h, w = 2 * [int(sqrt(x.shape[0]))]
        plt.imshow(x.reshape([h, w]), cmap='Greys')
    plt.show()


def rotate_img(x, angle):
    h, w = 2 * [int(sqrt(x.shape[0]))]
    result = rotate(x.reshape([h, w]), -angle, reshape=False).reshape(-1)
    return result


def shift_img(x, x_shift, y_shift):
    h, w = 2 * [int(sqrt(x.shape[0]))]
    result = shift(x.reshape([h, w]), [x_shift, y_shift]).reshape(-1)
    return result


def blur_img(x, sigma):
    h, w = 2 * [int(sqrt(x.shape[0]))]
    result = gaussian_filter(x.reshape([h, w]), sigma).reshape(-1)
    return result


def augment_data(x, y=None, angle=None, x_shift=None, y_shift=None, sigma=None):
    if angle is not None:
        x_augmented_1 = np.apply_along_axis(lambda arr: rotate_img(arr, -angle), axis=1, arr=x)
        x_augmented_2 = np.apply_along_axis(lambda arr: rotate_img(arr, angle), axis=1, arr=x)
        if y is None:
            return np.concatenate([x, x_augmented_1, x_augmented_2], axis=0)
        else:
            return np.concatenate([x, x_augmented_1, x_augmented_2], axis=0), np.concatenate([y, y, y], axis=0)
    if x_shift is not None and y_shift is not None:
        x_augmented_1 = np.apply_along_axis(lambda arr: shift_img(arr, -x_shift, -y_shift), axis=1, arr=x)
        x_augmented_2 = np.apply_along_axis(lambda arr: shift_img(arr, x_shift, y_shift), axis=1, arr=x)
        if y is None:
            return np.concatenate([x, x_augmented_1, x_augmented_2], axis=0)
        else:
            return np.concatenate([x, x_augmented_1, x_augmented_2], axis=0), np.concatenate([y, y, y], axis=0)
    if sigma is not None:
        x_augmented = np.apply_along_axis(lambda arr: blur_img(arr, sigma), axis=1, arr=x)
        if y is None:
            return np.concatenate([x, x_augmented], axis=0)
        else:
            return np.concatenate([x, x_augmented], axis=0), np.concatenate([y, y], axis=0)


def shuffle_data(x, y):
    permutation = np.arange(x.shape[0])
    np.random.shuffle(permutation)
    return x[permutation], y[permutation]


def list_mean(l):
    return sum(l) / len(l)
