class CellularAutomaton:
    def __init__(self, context, name="Unnamed CA"):
        self.context = context  # Initialize the context attribute
        self.name = name  # Add the name attribute

    def receive_rule(self, rule_data):
        """
        Process the received rule.

        Args:
            rule_data (dict): The distributed rule data.
        """
        print(f"{self.name} received the rule: {rule_data['rule']}")

    def execute_action(self, rule_data):
        """
        Executes the given rule action.

        Args:
            rule_data (dict): The rule data containing information about the action to execute.
        """
        print(f"[{self.context}] Executing action: {rule_data['rule']}")