import unittest

from src.decisioning import (
    ACTION_COSTS,
    ACTION_DISCOUNT_CALL,
    ACTION_EMAIL,
    ACTION_NONE,
    estimate_clv,
    expected_net_gain,
    recommended_action,
)


class TestDecisioning(unittest.TestCase):
    def test_estimate_clv_proxy(self):
        features = {"Balance": 1000.0, "Tenure": 5.0, "EstimatedSalary": 2000.0}
        clv = estimate_clv(features)
        self.assertAlmostEqual(clv, 1600.0, places=2)

    def test_recommended_action_thresholds(self):
        self.assertEqual(recommended_action(0.10), ACTION_NONE)
        self.assertEqual(recommended_action(0.30), ACTION_EMAIL)
        self.assertEqual(recommended_action(0.59), ACTION_EMAIL)
        self.assertEqual(recommended_action(0.60), ACTION_DISCOUNT_CALL)

    def test_expected_net_gain(self):
        clv = 2000.0
        p_churn = 0.4
        action = ACTION_EMAIL
        net_gain = expected_net_gain(p_churn, clv, ACTION_COSTS[action])
        self.assertAlmostEqual(net_gain, 795.0, places=2)


if __name__ == "__main__":
    unittest.main()
