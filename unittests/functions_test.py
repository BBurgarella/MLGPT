import unittest
from Functions import softmax, scaledDotProductSelfAttention
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

class TestSelfAttention(unittest.TestCase):

    def testSelfAttentionNoMask(self):
        """
        for this test, I generated keys, values and queries from the
        google collab notebook that is linked under the video,
        this allows me to validate the fact that we get consistent resutls
        """
        Queries = np.array([[0.29917916,0.32222616,0.44944202,0.40087818],
                            [0.42742966,0.75929284,0.94481094,0.16952069],
                            [0.66734589,0.18528671,0.72916473,0.01845809]])
        
        Keys = np.array([[0.36584701,0.43562185,0.4480778 ,0.90873129],
                         [0.31221214,0.07409408,0.83602278,0.04485871],
                         [0.87830161,0.9997026 ,0.95931107,0.09707838]])
        
        Values = np.array([[0.35764971,0.1772979 ,0.60785344,0.87649764],
                           [0.07441695,0.98663028,0.18366487,0.38199649],
                           [0.10817892,0.45987829,0.05264802,0.53601417]])

        ExpectedOutput = np.array([[0.18201628,0.5167383,0.27626652,0.6057456 ],
                                   [0.1683729 ,0.52207  ,0.24163088,0.5893614 ],
                                   [0.16761667,0.5406556,0.24742973,0.5844119 ]])
        
        ExpectedWeights = np.array([[0.33489865,0.28760365,0.3774977 ],
                                    [0.277405  ,0.26688248,0.4557126 ],
                                    [0.27928528,0.30317473,0.41753995]])

        Output, weights = scaledDotProductSelfAttention(Queries, Keys, Values)

        diffOut = np.sum(ExpectedOutput - Output)
        diffWeights = np.sum(ExpectedWeights - weights)

        self.assertAlmostEqual(diffOut, 0, 5)
        self.assertAlmostEqual(diffWeights, 0, 5)



if __name__ == '__main__':
    unittest.main()