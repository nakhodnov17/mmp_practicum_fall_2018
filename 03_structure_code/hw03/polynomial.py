class Polynomial:
    def __init__(self, *coeffs):
        self.coeffs = coeffs

    def __call__(self, x):
        return sum((self.coeffs[i] * x**i for i in range(len(self.coeffs))))
