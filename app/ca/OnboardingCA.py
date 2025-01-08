from app.ca.CellularAutomaton import CellularAutomaton

class OnboardingCA(CellularAutomaton):
    def __init__(self):
        super().__init__(context="onboarding process", name="Onboarding CA")

    def execute_action(self, rule_data):
        print(f"[{self.name}] Applying rule: {rule_data['rule']}")
