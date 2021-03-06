import numpy as np
import scipy
import scipy.sparse
import scipy.special


def sigmoid(x):
    # type: (Union[scipy.sparse.csr_matrix, np.ndarray]) -> np.ndarray
    return scipy.special.expit(x)


class BaseSmoothOracle:
    """
    Базовый класс для реализации оракулов.
    """
    def func(self, x, y, w):
        """
        Вычислить значение функции в точке w.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x, y, w):
        """
        Вычислить значение градиента функции в точке w.
        """
        raise NotImplementedError('Grad oracle is not implemented.')

        
class BinaryLogistic(BaseSmoothOracle):
    """
    Оракул для задачи двухклассовой логистической регрессии.
    
    Оракул должен поддерживать l2 регуляризацию.
    """

    def __init__(self, l2_coef=0.):
        """
        Задание параметров оракула.
        
        l2_coef - коэффициент l2 регуляризации
        """
        self.l2_coef = l2_coef

    def func(self, x, y, w):
        """
        Вычислить значение функционала в точке w на выборке x с ответами y.

        x - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w - одномерный numpy array
        """

        return np.sum(
            np.logaddexp(0., -y * x.dot(w))
        ) / float(x.shape[0]) + self.l2_coef * np.linalg.norm(w) ** 2 / 2.

    def grad(self, x, y, w):
        """
        Вычислить градиент функционала в точке w на выборке x с ответами y.
        
        x - scipy.sparse.csr_matrix или двумерный numpy.array
        
        y - одномерный numpy array
        
        w - одномерный numpy array
        """
        if isinstance(x, np.ndarray):
            return -np.sum(
                np.multiply(
                    np.multiply(
                        x,
                        y.reshape([-1, 1])
                    ),
                    sigmoid(-y * x.dot(w)).reshape([-1, 1])
                ),
                axis=0
            ) / float(x.shape[0]) + self.l2_coef * w
        else:
            return np.asarray(
                -np.sum(
                    x.multiply(
                        y.reshape([-1, 1])
                    ).multiply(
                        sigmoid(-y * x.dot(w)).reshape([-1, 1])
                    ),
                    axis=0
                )
            ).reshape([-1]) / float(x.shape[0]) + self.l2_coef * w


class MulticlassLogistic(BaseSmoothOracle):
    """
    Оракул для задачи многоклассовой логистической регрессии.
    
    Оракул должен поддерживать l2 регуляризацию.
    
    w в этом случае двумерный numpy array размера (class_number, d),
    где class_number - количество классов в задаче, d - размерность задачи
    """
    
    def __init__(self, class_number=0, l2_coef=0.):
        """
        Задание параметров оракула.
        
        class_number - количество классов в задаче
        
        l2_coef - коэффициент l2 регуляризации
        """
        self.class_number = class_number
        self.l2_coef = l2_coef
     
    def func(self, x, y, w):
        """
        Вычислить значение функционала в точке w на выборке x с ответами y.
        
        x - scipy.sparse.csr_matrix или двумерный numpy.array
        
        y - одномерный numpy array
        
        w - двумерный numpy array
        """
        logits = x.dot(w.T)
        max_logits = np.max(logits, axis=1)
        log_denominator = np.log(np.sum(np.exp(logits - max_logits.reshape([-1, 1])), axis=1))
        log_numerator = logits[range(x.shape[0]), y] - max_logits
        return -np.sum(log_numerator - log_denominator) / float(x.shape[0]) + self.l2_coef * np.linalg.norm(w) ** 2 / 2.
        
    def grad(self, x, y, w):
        """
        Вычислить значение функционала в точке w на выборке x с ответами y.
        
        x - scipy.sparse.csr_matrix или двумерный numpy.array
        
        y - одномерный numpy array
        
        w - двумерный numpy array
        """
        logits = x.dot(w.T)
        max_logits = np.max(logits, axis=1).reshape([-1, 1])
        indicator = y.reshape(-1, 1) == np.arange(0, w.shape[0]).reshape(1, -1)
        probas = np.exp(logits - max_logits) / np.sum(np.exp(logits - max_logits), axis=1).reshape([-1, 1])
        b = (indicator - probas).reshape([x.shape[0], w.shape[0], 1])
        x = x if isinstance(x, np.ndarray) else x.toarray()
        return -np.sum(x.reshape([x.shape[0], 1, -1]) * b, axis=0) / float(x.shape[0]) + self.l2_coef * w
