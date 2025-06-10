import unittest
import numpy as np

def create_test_agg_fn(lowerbound, upperbound):
    def agg_fn(array):
        if np.isnan(array).all():
            return float("nan")
        else:
            values = np.copy(array[~np.isnan(array)])
            if not np.isnan(lowerbound):
                values = values[values >= lowerbound]
            if not np.isnan(upperbound):
                values = values[values <= upperbound]
            if len(values) > 0:
                return np.mean(values)
            else:
                return float("nan")
    return agg_fn

class TestAggFn(unittest.TestCase):

    def setUp(self):
        self.agg_fn = create_test_agg_fn(lowerbound=0, upperbound=100)

    def test_all_nan(self):
        array = np.array([np.nan, np.nan])
        self.assertTrue(np.isnan(self.agg_fn(array)))

    def test_within_bounds(self):
        array = np.array([10, 20, 30])
        self.assertEqual(self.agg_fn(array), 20.0)

    def test_above_upper_bound(self):
        array = np.array([10, 20, 150])
        self.assertEqual(self.agg_fn(array), 15.0)

    def test_below_lower_bound(self):
        array = np.array([-10, 20, 30])
        self.assertEqual(self.agg_fn(array), 25.0)

    def test_all_out_of_bounds(self):
        array = np.array([-10, 150])
        self.assertTrue(np.isnan(self.agg_fn(array)))

    def test_mixed_nans(self):
        array = np.array([np.nan, 50, 80])
        self.assertEqual(self.agg_fn(array), 65.0)

if __name__ == '__main__':
    unittest.main()
