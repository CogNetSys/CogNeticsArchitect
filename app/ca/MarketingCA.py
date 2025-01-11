# app/ca/MarketingCA.py

from app.ca.CellularAutomaton import CellularAutomaton

class MarketingCA(CellularAutomaton):
    def __init__(self):
        super().__init__("Marketing CA")

    def execute_action(self, rule_data):
        """
        Perform actions based on the distributed rule.

        Args:
            rule_data (dict): The distributed rule data.
        """
        if "marketing" in rule_data["rule"].lower():
            print(f"[Marketing CA] Adjusting marketing strategy based on rule: {rule_data['rule']}")
        else:
            print(f"[Marketing CA] No relevant action for this rule.")