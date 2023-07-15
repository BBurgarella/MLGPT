import unittest
from Functions import softmax
import numpy as np


class Testsoftmax(unittest.TestCase):

    def test_single_value(self):
        val = 0
        softVal = softmax(val)
        self.assertEqual(softVal, 1)

    def test_vector(self):
        val = np.array([1,3,4,5])
        softVal = softmax(val)
        Expected = np.array([0.01203764271194, 0.088946817297404, 0.2417825171588, 0.65723302283186])
        diff = softVal - Expected
        sum = np.sum(diff)
        self.assertAlmostEqual(sum,0)

if __name__ == '__main__':
    unittest.main()