import numpy as np
from matplotlib import pyplot as plt
from scipy import stats


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h


def generate_uniform_data(a=.0, b=1.0, size=10):
    from numpy.random.mtrand import uniform
    return uniform(a, b, size)


def generate_exp_data(lam=1.0, size=10):
    from numpy.random.mtrand import exponential
    return exponential(lam, size)


def generate_triangular_data(a, b, peak, size=10):
    from numpy.random.mtrand import triangular
    return triangular(a, peak, b, size)


def mse(x):
    return np.sqrt(np.var(x))


def prep_for_output(data):
    return ", ".join(str(np.round(x, 2)) for x in data)


if __name__ == '__main__':
    alfa = 20
    lam = 1 / 200
    a = 190
    size = 100

    uniform_data = generate_uniform_data(b=alfa, size=size)
    exp_data = generate_exp_data(lam=lam, size=size)
    triang_data = generate_triangular_data(0, a, 0, size=size)

    print(f'Uniform: {prep_for_output(uniform_data)}')
    estim_data_for_uniform = mean_confidence_interval(uniform_data)
    print(f'Mean value estimation for uniform: {estim_data_for_uniform}')
    print(f'MSE value estimation for uniform: {mse(uniform_data)}')
    print()

    print(f'Exponential : {prep_for_output(exp_data)}')
    estim_data_for_exp = mean_confidence_interval(exp_data)
    print(f'Mean value estimation for uniform: {estim_data_for_exp}')
    print(f'MSE value estimation for uniform: {mse(exp_data)}')
    print()

    print(f'Triangular: {prep_for_output(triang_data)}')
    estim_data_for_triang = mean_confidence_interval(triang_data)
    print(f'Mean value estimation for uniform: {estim_data_for_triang}')
    print(f'MSE value estimation for uniform: {mse(triang_data)}')
    print()

    plt.hist(uniform_data)
    plt.title('Uniform distribution')
    plt.grid(True)
    plt.box(True)
    plt.show()
    plt.hist(exp_data)
    plt.title('Exponential distribution')
    plt.grid(True)
    plt.box(True)
    plt.show()
    plt.hist(triang_data)
    plt.title('Triangular distribution')
    plt.grid(True)
    plt.box(True)
    plt.show()
