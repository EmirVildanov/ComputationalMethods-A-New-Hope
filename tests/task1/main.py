from tasks.task1.main import *
import unittest
from numpy.testing import *


class TestTask1Methods(unittest.TestCase):
    def test_default_example_works(self):
        A_matrix_rows = [[1, 0.99], [0.99, 0.98]]
        b = np.array([1.99, 1.97])
        A = np.array(A_matrix_rows)

        inverse_A = np.linalg.inv(A)
        x = inverse_A.dot(b)
        assert_allclose(x, np.array([1., 1.]), rtol=1e-5, atol=0)

    def test_default_varied_example_works(self):
        A_matrix_rows = [[1, 0.99], [0.99, 0.98]]
        b = np.array([2, 2])
        A = np.array(A_matrix_rows)

        inverse_A = np.linalg.inv(A)
        x = inverse_A.dot(b)
        assert_allclose(x, np.array([200., -200.]), rtol=1e-5, atol=0)

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)


if __name__ == '__main__':
    unittest.main()
