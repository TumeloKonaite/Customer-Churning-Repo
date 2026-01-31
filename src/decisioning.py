"""Lightweight retention decision engine utilities.

All functions are deterministic and rely on simple, explainable heuristics.
Assumptions are documented per function and in module-level constants.
"""

from typing import Mapping

LOW_RISK_THRESHOLD = 0.30
HIGH_RISK_THRESHOLD = 0.60

ACTION_NONE = "No action"
ACTION_EMAIL = "Retention email"
ACTION_DISCOUNT_CALL = "Discount or retention call"

ACTION_COSTS = {
    ACTION_NONE: 0.0,
    ACTION_EMAIL: 5.0,
    ACTION_DISCOUNT_CALL: 50.0,
}


def estimate_clv(customer_features: Mapping[str, float]) -> float:
    """Estimate customer lifetime value (CLV) using a simple proxy.

    Assumptions:
    - Balance is a stand-in for account value.
    - Tenure increases value linearly with a small multiplier.
    - Salary is used only as a modest stabilizer to avoid zero CLV.

    Proxy formula:
        clv = max(0, balance) * (1 + tenure / 10) + 0.05 * max(0, salary)

    Expected inputs include keys like "Balance", "Tenure", and "EstimatedSalary".
    Missing values default to 0.
    """
    balance = float(customer_features.get("Balance", 0.0) or 0.0)
    tenure = float(customer_features.get("Tenure", 0.0) or 0.0)
    salary = float(customer_features.get("EstimatedSalary", 0.0) or 0.0)

    balance = max(0.0, balance)
    tenure = max(0.0, tenure)
    salary = max(0.0, salary)

    tenure_factor = 1.0 + (tenure / 10.0)
    clv = (balance * tenure_factor) + (0.05 * salary)
    return float(clv)


def recommended_action(p_churn: float) -> str:
    """Map churn probability to a retention action.

    Thresholds:
    - p_churn < LOW_RISK_THRESHOLD -> no action
    - LOW_RISK_THRESHOLD <= p_churn < HIGH_RISK_THRESHOLD -> retention email
    - p_churn >= HIGH_RISK_THRESHOLD -> discount or retention call
    """
    if p_churn < LOW_RISK_THRESHOLD:
        return ACTION_NONE
    if p_churn < HIGH_RISK_THRESHOLD:
        return ACTION_EMAIL
    return ACTION_DISCOUNT_CALL


def expected_net_gain(p_churn: float, clv: float, action_cost: float) -> float:
    """Compute expected net gain from a retention action.

    value_saved = p_churn * clv
    net_gain = value_saved - action_cost
    """
    value_saved = p_churn * clv
    net_gain = value_saved - action_cost
    return float(net_gain)
