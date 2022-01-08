import numpy as np
import unittest as ut
import numpy.linalg as linalg
from skdiveMove.imutools import vector


class TestSequenceFunctions(ut.TestCase):
    def setUp(self):
        self.delta = 1e-5

    def test_normalize(self):
        result = vector.normalize([3, 0, 0])
        correct = np.array([[1.0, 0.0, 0.0]])
        error = linalg.norm(result - correct)
        self.assertAlmostEqual(error, 0)

        # Ensure that 'normalize' does not modify the input values
        data = 10 * np.ones(5)
        vector.normalize(data[:3])
        self.assertEqual(data[2], 10)

    def test_vangle(self):
        v1 = np.array([[1, 0, 0],
                       [1, 1, 0]])
        v2 = np.array([[1, 0, 0],
                       [2, 0, 0]])
        result = vector.vangle(v1, v2)
        correct = np.array([0, np.pi / 4])
        self.assertAlmostEqual(np.linalg.norm(correct - result), 0)
        self.assertAlmostEqual(vector.vangle(v1[0], v1[1]), np.pi / 4)

        result = vector.vangle(v1, v2[0])
        correct = np.array([0, np.pi / 4])
        self.assertAlmostEqual(np.linalg.norm(correct - result), 0)


if __name__ == "__main__":
    ut.main()
