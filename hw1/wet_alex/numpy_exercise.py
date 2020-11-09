import numpy as np
from scipy.misc import face
from scipy.stats import norm
import scipy as sp
import matplotlib.pyplot as plt
import timeit


def one_hot_5_of_10():
    """
    Return a zero vector of size 10 but the fifth value which is 1
    """
    my_arr = np.zeros([10])
    my_arr[5] = 1

    return my_arr


def negate_3_to_8(x):
    """
    Given a 1D array, negate all elements which are between 3 and 8
    """
    return np.where( (x>3) & (x<8), -x, x)


def get_size_properties(x):
    """
    Given an array x, return a tuple with the following properties:
    (num_rows, num_cols, num_elements, num_dimensions)
    """

    num_rows, num_cols = x.shape
    num_elements = x.size
    num_dimensions = x.ndims

    return num_rows, num_cols, num_elements, num_dimensions


def append_vector_to_matrix(x, y):
    """
    Append row vector y to the end (bottom) of matrix x.
    Result may be a new matrix (rather than the input matrix itself)
    """
    num_rows, num_cols = x.shape
    new_mat = np.zeros([num_rows+1, num_cols])
    new_mat[0:num_rows, :] = x
    new_mat[num_rows, :] = y

    return new_mat


def column_sum(x):
    """
    Return a vector containing the sum of each column of x
    """
    return np.sum(x, axis = 0)


def multiplication_table():
    """
    print the multiplication table ("lu'ach ha'kefel") using Python's broadcasting
    """
    min = 1
    max = 10

    arr = np.arange(min, max+1)
    arr_row = arr.reshape(max-min+1, 1)
    arr_col = arr.reshape(1, max - min +1)
    mult_table = arr_row * arr_col

    print(mult_table)



def view_face():
    """
    View the face image using Scipy's scipy.misc.face() and display the image
    """
    face = sp.misc.face()
    plt.imshow(face)

    # that's all?



def q1():
    a = np.arange(4)
    b = a[2:4]
    b[0] = 10
    return a


def q2():
    a, b = np.meshgrid(np.arange(4), np.arange(0, 30, 10))
    mesh = a + b
    return mesh


def plot_samples(sample, x):
    """
    Fill in the missing lines to match the titles of the subplots
    """
    plt.figure()

    plt.subplot(2,2,1)
    plt.title('Normal Random Variable')
    plt.plot(sample)

    plt.subplot(2,2,2)
    plt.title('Probability Distribution Function')

    # calculate the mean and variance
    mean = np.mean(sample)
    var = np.var(sample)
    pdf = sp.stats.norm.pdf(x, loc=mean, scale=var)

    plt.plot(x, pdf)

    plt.subplot(2,2,3)
    plt.title('Cummulative Distribution Function')
    cdf = sp.stats.norm.cdf(x, loc=mean, scale=var)
    plt.plot(x, cdf)

    plt.subplot(2,2,4)
    plt.title('Percent Point Function')
    ppf = sp.stats.norm.ppf(x, loc=mean, scale=var)
    plt.plot(x, ppf)
    plt.show(block=True)


def seed_zero():
    """
    Seed numpy's random generator with the value 0
    """
    np.random.seed(0)


def test(got, expected):
    """
    Simple provided test() function used in main() to print
    what each function returns vs. what it's supposed to return.
    """
    if got == expected:
        prefix = ' OK '
    else:
        prefix = '  X '
    print('%s got: %s expected: %s' % (prefix, repr(got), repr(expected)))


def test_array(got, expected):
    if np.array_equal(got, expected):
        prefix = ' OK '
    else:
        prefix = '  X '
    print('%s got:\n    %s\n expected:\n    %s' % (prefix, repr(got), repr(expected)))


def mat_mul_pure_python(x, y):
    result = [[0] * len(x)] * len(y[0])
    # iterate through rows of X
    for i in range(len(x)):
        # iterate through columns of Y
        for j in range(len(y[0])):
            # iterate through rows of Y
            for k in range(len(y)):
                result[i][j] += x[i][k] * y[k][j]


# Calls the above functions with interesting inputs.
def main():
    # Numpy
    seed_zero()

    x = np.array([[0, 1, 2, 3],
                 [10, 11, 12, 13]])
    y = np.array([20, 21, 22, 23])
    z = np.arange(10)

    # Implement function a-?
    test_array(one_hot_5_of_10(), np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]))
    test_array(negate_3_to_8(z), [0, 1, 2, 3, -4, -5, -6, -7, 8, 9])
    test(get_size_properties(x), (2, 4, 8, 2))
    test_array(append_vector_to_matrix(x, y), np.array([[ 0,  1,  2,  3],
                                                        [10, 11, 12, 13],
                                                        [20, 21, 22, 23]]))
    test_array(column_sum(x), np.array([10, 12, 14, 16]))

    # TODO  - what the f do i do here
    # TODO -------------------

    # Fill in the expected value of the functions Q1-Q2.
    # Yes, we're aware you can print the value and copy-paste it.
    # Please try to think about it before you do so.
    a1 = np.array([0, 1, 10, 3])
    test_array(a1, q1())
    a2 = [[ 0,  1,  2,  3],
          [10, 11, 12, 13],
          [20, 21, 22, 23]]
    test_array(a2, q2())
    # -----------

    multiplication_table()

    # SciPy
    view_face()

    sample = norm.rvs(size=100)
    x = sp.r_[-5:5:100j]
    plot_samples(sample, x)

    # Compare the execution speed of matrix multiplication using pure python and using SciPy.
    # No need to submit this section, only to be impressed by the incredible gap in performance.
    setup1 = \
"""from __main__ import mat_mul_pure_python
from scipy.stats import norm
import numpy as np
X = np.random.random((100, 100))
Y = np.random.random((100, 100))
x_list = X.tolist()
y_list = Y.tolist()"""
    print(timeit.timeit("mat_mul_pure_python(x_list, y_list)", setup=setup1, number=10))

    setup2 = \
"""from scipy.stats import norm
import numpy as np
X = np.random.random((100, 100))
Y = np.random.random((100, 100))
"""
    print(timeit.timeit("np.matmul(X, Y)", setup=setup1, number=10))


if __name__ == '__main__':
    main()
