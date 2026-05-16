import unittest

import pandas as pd

from main_2 import find_cheapest_way_to_make_customer_happy


class RuleBasedModel:
    def __init__(self, is_happy):
        self.is_happy = is_happy
        self.seen_inputs = []

    def predict(self, X):
        self.seen_inputs.append(X)

        if isinstance(X, pd.DataFrame):
            state = tuple(X.iloc[0].tolist())
        else:
            state = tuple(X[0])

        return [1 if self.is_happy(state) else 0]


class FindCheapestWayToMakeCustomerHappyTest(unittest.TestCase):
    def test_returns_zero_cost_when_customer_is_already_happy(self):
        current_state = pd.Series(
            [5, "fast", "yes"],
            index=["rating", "service_speed", "parking_available"],
        )
        model = RuleBasedModel(lambda state: state[0] >= 4 and state[1] == "fast")

        result = find_cheapest_way_to_make_customer_happy(
            current_state=current_state,
            cost_to_change_state={
                "rating": {
                    5: [(1, 10), (3, 5)],
                },
                "service_speed": {
                    "fast": [("slow", 2)],
                },
                "parking_available": {
                    "yes": [("no", 1)],
                },
            },
            model=model,
        )

        self.assertEqual(result["cost"], 0)
        self.assertEqual(result["path"], [(5, "fast", "yes")])
        self.assertIsInstance(model.seen_inputs[0], pd.DataFrame)
        self.assertEqual(
            list(model.seen_inputs[0].columns),
            ["rating", "service_speed", "parking_available"],
        )

    def test_finds_cheapest_multi_step_path(self):
        model = RuleBasedModel(
            lambda state: (
                state[0] == "high" and state[1] == "short"
            ) or state[2] == "vip"
        )

        result = find_cheapest_way_to_make_customer_happy(
            current_state=("low", "long", "regular"),
            cost_to_change_state={
                "rating": {
                    "low": [("medium", 1), ("high", 12)],
                    "medium": [("high", 4)],
                },
                "queue": {
                    "long": [("short", 4)],
                },
                "status": {
                    "regular": [("vip", 20)],
                },
            },
            model=model,
        )

        self.assertEqual(result["cost"], 9)
        self.assertEqual(result["path"][0], ("low", "long", "regular"))
        self.assertEqual(result["path"][-1], ("high", "short", "regular"))
        self.assertLess(result["cost"], 20)

        for previous_state, next_state in zip(result["path"], result["path"][1:]):
            changed_features = sum(
                previous_value != next_value
                for previous_value, next_value in zip(previous_state, next_state)
            )
            self.assertEqual(changed_features, 1)

    def test_returns_none_when_no_happy_state_is_reachable(self):
        model = RuleBasedModel(lambda state: state == ("premium", "perfect"))

        result = find_cheapest_way_to_make_customer_happy(
            current_state=("basic", "bad"),
            cost_to_change_state={
                "tier": {
                    "basic": [("standard", 3)],
                },
                "service": {
                    "bad": [("good", 4)],
                },
            },
            model=model,
        )

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
