from src.decisioning import (
    ACTION_COSTS,
    ACTION_DISCOUNT_CALL,
    ACTION_EMAIL,
    ACTION_NONE,
    estimate_clv,
    expected_net_gain,
    recommended_action,
)

def test_estimate_clv_proxy():
    features = {"Balance": 1000.0, "Tenure": 5.0, "EstimatedSalary": 2000.0}
    clv = estimate_clv(features)
    assert clv == 1600.0


def test_recommended_action_thresholds():
    assert recommended_action(0.10) == ACTION_NONE
    assert recommended_action(0.30) == ACTION_EMAIL
    assert recommended_action(0.59) == ACTION_EMAIL
    assert recommended_action(0.60) == ACTION_DISCOUNT_CALL


def test_expected_net_gain():
    clv = 2000.0
    p_churn = 0.4
    action = ACTION_EMAIL
    net_gain = expected_net_gain(p_churn, clv, ACTION_COSTS[action])
    assert round(net_gain, 2) == 795.0
