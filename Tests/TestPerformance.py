import unittest
import numpy as np
import tensorflow as tf
from Utils.performance_funcs import calculate_inception_score


class TestPerformance(unittest.TestCase):

    def test_calculate_inception_score_with_manual_example(self):
        ς = 1.e-20
        py = np.array([0.33, 0.33, 0.33])
        pyx = np.array([[0.05, 0.05, 0.9],
                        [0.1, 0.2, 0.7],
                        [0.7, 0.15, 0.15],
                        [0.1, 0.8, 0.1]])
        ops_matrix = pyx * (np.log(pyx + ς) - np.log(py + ς))
        inception_score_ans = np.exp(np.mean(np.sum(ops_matrix, axis=1), axis=0))
        py = tf.constant(py, dtype=tf.float32)
        pyx = tf.constant(pyx, dtype=tf.float32)
        inception_score0 = calculate_inception_score(pyx=pyx, py=py)

        py = np.array([0.1, 0.3, 0.6])
        pyx = np.array([[0.2, 0.3, 0.5],
                        [0.5, 0.4, 0.1],
                        [0.33, 0.33, 0.33],
                        [0.3, 0.4, 0.3]])
        py = tf.constant(py, dtype=tf.float32)
        pyx = tf.constant(pyx, dtype=tf.float32)
        inception_score1 = calculate_inception_score(pyx=pyx, py=py)

        self.assertTrue(expr=np.isclose(inception_score_ans, inception_score0.numpy()))
        self.assertTrue(expr=inception_score0 > inception_score1)


if __name__ == '__main__':
    unittest.main()
