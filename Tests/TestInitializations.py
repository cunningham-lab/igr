# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ===========================================================================================================
# Imports
# ===========================================================================================================
import unittest
import numpy as np
from Utils.initializations import get_uniform_mix_probs
# ===========================================================================================================


class TestSBDist(unittest.TestCase):

    def test_get_uniform_mix_probs(self):
        #    Global parameters  #######
        test_tolerance = 1.e-7
        probs_ans = np.array([0.3, 0.3, 0.3, 0.1/4, 0.1/4, 0.1/4, 0.1/4, 0, 0, 0])
        initial_point, middle_point, final_point, mass_in_beginning, max_size = 0, 2, 6, 0.9, 10
        probs = get_uniform_mix_probs(initial_point=initial_point, middle_point=middle_point,
                                      final_point=final_point,
                                      mass_in_beginning=mass_in_beginning, max_size=max_size)
        relative_diff = np.linalg.norm(probs - probs_ans) / np.linalg.norm(probs_ans)
        self.assertTrue(expr=relative_diff < test_tolerance)


if __name__ == '__main__':
    unittest.main()
