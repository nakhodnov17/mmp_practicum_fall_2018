import scipy.special
import time

from oracles import BinaryLogistic, MulticlassLogistic

from collections import defaultdict

import numpy as np


class GDClassifier:
    """
    Реализация метода градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """

    def __init__(self, loss_function, step_alpha=1, step_beta=0,
                 tolerance=1e-5, max_iter=1000, **kwargs):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора.
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия
        - 'multinomial_logistic' - многоклассовая логистическая регрессия

        step_alpha - float, параметр выбора шага из текста задания

        step_beta- float, параметр выбора шага из текста задания

        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию.
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если (f(x_{k+1}) - f(x_{k})) < tolerance: то выход

        max_iter - максимальное число итераций

        **kwargs - аргументы, необходимые для инициализации
        """
        self.loss_function = loss_function
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.w_0 = None

        self.w = None

        if loss_function == 'binary_logistic':
            self.objective = BinaryLogistic(**kwargs)
        else:
            self.objective = MulticlassLogistic(**kwargs)

    def fit(self, x, y, w_0=None, trace=False, x_test=None, y_test=None):
        """
        Обучение метода по выборке x с ответами y

        x - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w_0 - начальное приближение в методе

        trace - переменная типа bool

        Если trace = True, то метод должен вернуть словарь history, содержащий информацию
        о поведении метода. Длина словаря history = количество итераций + 1 (начальное приближение)

        history['time']: list of floats, содержит интервалы времени между двумя итерациями метода
        history['func']: list of floats, содержит значения функции на каждой итерации
        (0 для самой первой точки)
        """
        if w_0 is not None:
            self.w_0 = w_0
        else:
            if self.loss_function == 'binary_logistic':
                self.w_0 = np.zeros([x.shape[1]], dtype=np.float)
            else:
                self.w_0 = np.zeros([np.max(y) + 1, x.shape[1]], dtype=np.float)
        self.w = self.w_0
        history = defaultdict(list)

        # history['func'].append(self.get_objective(x, y))
        history['func_train'].append(self.get_objective(x, y))
        history['accuracy_train'].append(self.get_accuracy(x, y))
        if x_test is not None:
            history['func_test'].append(self.get_objective(x_test, y_test))
            history['accuracy_test'].append(self.get_accuracy(x_test, y_test))
        else:
            history['func_test'].append(0.)
            history['accuracy_test'].append(0.)

        history['time'].append(0)

        n_iter = 0
        while (
                n_iter < self.max_iter and
                (n_iter == 0 or np.abs(history['func_train'][-2] - history['func_train'][-1]) > self.tolerance)
        ):
            start = time.time()

            self.w -= self.step_alpha / (n_iter + 1) ** self.step_beta * self.get_gradient(x, y)

            history['time'].append(time.time() - start)
            # history['func'].append(self.get_objective(x, y))
            history['func_train'].append(self.get_objective(x, y))
            history['accuracy_train'].append(self.get_accuracy(x, y))
            if x_test is not None:
                history['func_test'].append(self.get_objective(x_test, y_test))
                history['accuracy_test'].append(self.get_accuracy(x_test, y_test))
            print(
                'Loss (Train/Test): {0:.4f}/{1:.4f}'.format(
                    history['func_train'][-1], history['func_test'][-1]
                )
            )
            print(
                'Accuracy (Train/Test): {0:.4f}/{1:.4f}\n'.format(
                    history['accuracy_train'][-1], history['accuracy_test'][-1]
                )
            )
            n_iter += 1
        if trace:
            return history

    def predict(self, x):
        """
        Получение меток ответов на выборке x

        x - scipy.sparse.csr_matrix или двумерный numpy.array

        return: одномерный numpy array с предсказаниями
        """
        if self.loss_function == 'binary_logistic':
            return np.sign(x.dot(self.w))
        else:
            return np.argmax(x.dot(self.w.T), axis=1)

    def predict_proba(self, x):
        """
        Получение вероятностей принадлежности x к классу k

        x - scipy.sparse.csr_matrix или двумерный numpy.array

        return: двумерной numpy array, [i, k] значение соответветствует вероятности
        принадлежности i-го объекта к классу k
        """
        if self.loss_function == 'binary_logistic':
            probas = np.zeros((x.shape[0], 2))
            probas[:, 1] = scipy.special.expit(x.dot(self.w))
            probas[:, 0] = 1 - probas[:, 1]
            return probas
        else:
            classes = np.argmax(x.dot(self.w.T))
            logits = x.dot(self.w.T)
            max_logits = np.max(logits, axis=1)
            denominator = np.sum(np.exp(logits - max_logits.reshape([-1, 1])), axis=1)
            numerator = np.exp(logits[range(x.shape[0]), classes] - max_logits)
            return numerator / denominator

    def get_objective(self, x, y):
        """
        Получение значения целевой функции на выборке x с ответами y

        x - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array

        return: float
        """
        return self.objective.func(x, y, self.w)

    def get_gradient(self, x, y):
        """
        Получение значения градиента функции на выборке x с ответами y

        x - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array

        return: numpy array, размерность зависит от задачи
        """
        return self.objective.grad(x, y, self.w)

    def get_accuracy(self, x, y):
        return np.mean(y == self.predict(x))

    def get_weights(self):
        """
        Получение значения весов функционала
        """
        return self.w


class SGDClassifier(GDClassifier):
    """
    Реализация метода стохастического градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """

    def __init__(self, loss_function, batch_size=128, step_alpha=1, step_beta=0,
                 tolerance=1e-5, max_iter=1000, random_seed=153, **kwargs):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора.
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия
        - 'multinomial_logistic' - многоклассовая логистическая регрессия

        batch_size - размер подвыборки, по которой считается градиент

        step_alpha - float, параметр выбора шага из текста задания

        step_beta- float, параметр выбора шага из текста задания

        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если (f(x_{k+1}) - f(x_{k})) < tolerance: то выход


        max_iter - максимальное число итераций

        random_seed - в начале метода fit необходимо вызвать np.random.seed(random_seed).
        Этот параметр нужен для воспроизводимости результатов на разных машинах.

        **kwargs - аргументы, необходимые для инициализации
        """
        super(SGDClassifier, self).__init__(loss_function, step_alpha, step_beta, tolerance, max_iter, **kwargs)
        self.batch_size = batch_size
        self.random_seed = random_seed

    def fit(self, x, y, w_0=None, trace=False, log_freq=1., x_test=None, y_test=None):
        """
        Обучение метода по выборке x с ответами y

        x - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w_0 - начальное приближение в методе

        Если trace = True, то метод должен вернуть словарь history, содержащий информацию
        о поведении метода. Если обновлять history после каждой итерации, метод перестанет
        превосходить в скорости метод GD. Поэтому, необходимо обновлять историю метода лишь
        после некоторого числа обработанных объектов в зависимости от приближённого номера эпохи.
        Приближённый номер эпохи:
            {количество объектов, обработанных методом SGD} / {количество объектов в выборке}

        log_freq - float от 0 до 1, параметр, отвечающий за частоту обновления.
        Обновление должно проиходить каждый раз, когда разница между двумя значениями приближённого номера эпохи
        будет превосходить log_freq.

        history['epoch_num']: list of floats, в каждом элементе списка будет записан приближённый номер эпохи:
        history['time']: list of floats, содержит интервалы времени между двумя соседними замерами
        history['func']: list of floats, содержит значения функции после текущего приближённого номера эпохи
        history['weights_diff']: list of floats, содержит квадрат нормы разности векторов весов с соседних замеров
        (0 для самой первой точки)
        """
        np.random.seed(self.random_seed)

        if w_0 is not None:
            self.w_0 = w_0
        else:
            if self.loss_function == 'binary_logistic':
                self.w_0 = np.zeros([x.shape[1]], dtype=np.float)
            else:
                self.w_0 = np.zeros([np.max(y) + 1, x.shape[1]], dtype=np.float)
        self.w = self.w_0
        history = defaultdict(list)

        history['epoch_num'].append(0.)
        history['time'].append(0)
        # history['func'].append(self.get_objective(x, y))
        history['func_train'].append(self.get_objective(x, y))
        history['accuracy_train'].append(self.get_accuracy(x, y))
        if x_test is not None:
            history['func_test'].append(self.get_objective(x_test, y_test))
            history['accuracy_test'].append(self.get_accuracy(x_test, y_test))
        else:
            history['func_test'].append(0.)
            history['accuracy_test'].append(0.)

        n_iter = 0
        while (
                history['epoch_num'][-1] < self.max_iter and
                (
                        len(history['func_train']) < 2 or
                        np.abs(history['func_train'][-2] - history['func_train'][-1]) > self.tolerance
                )
        ):
            start = time.time()

            indices = np.random.randint(0, x.shape[0], [self.batch_size])
            epoch_num = (n_iter + 1.) * self.batch_size / x.shape[0]
            self.w -= self.step_alpha / (epoch_num + 1) ** self.step_beta * self.get_gradient(x[indices], y[indices])

            if epoch_num - history['epoch_num'][-1] > log_freq:
                history['epoch_num'].append(epoch_num)
                history['time'].append(time.time() - start)
                # history['func'].append(self.get_objective(x, y))
                history['func_train'].append(self.get_objective(x, y))
                history['accuracy_train'].append(self.get_accuracy(x, y))
                if x_test is not None:
                    history['func_test'].append(self.get_objective(x_test, y_test))
                    history['accuracy_test'].append(self.get_accuracy(x_test, y_test))
                print(
                    'Loss (Train/Test): {0:.4f}/{1:.4f}'.format(
                        history['func_train'][-1], history['func_test'][-1]
                    )
                )
                print(
                    'Accuracy (Train/Test): {0:.4f}/{1:.4f}\n'.format(
                        history['accuracy_train'][-1], history['accuracy_test'][-1]
                    )
                )
            n_iter += 1
        if trace:
            return history
