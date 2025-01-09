from app.ca.MarketingCA import MarketingCA
from app.ca.ResourceAllocationCA import ResourceAllocationCA
from app.ca.OnboardingCA import OnboardingCA

# Instantiate CAs
marketing_ca = MarketingCA()
resource_allocation_ca = ResourceAllocationCA()
onboarding_ca = OnboardingCA()

# CA registry (maps rule context to CAs)
ca_registry = {
    "marketing": marketing_ca,
    "resource_allocation": resource_allocation_ca,
    "onboarding process": onboarding_ca,
}

def distribute_rule(rule_data):
    """
    Distribute the rule to the relevant CA based on its context.

    Args:
        rule_data (dict): The rule data to be distributed.
    """
    context = rule_data.get("context", "general").lower()

    # Find the appropriate CA
    ca = ca_registry.get(context)

    if ca:
        ca.receive_rule(rule_data)
        ca.execute_action(rule_data)
    else:
        print(f"[RuleDistributorAgent] No CA found for context '{context}'")

# Example Usage
if __name__ == "__main__":
    example_rule_data = {
        "rule": "If demand is high, allocate additional onboarding resources and notify stakeholders.",
        "class": "high",
        "context": "marketing",
    }
    distribute_rule(example_rule_data)