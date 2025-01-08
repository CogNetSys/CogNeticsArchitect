import json
from RuleDistributorAgent import distribute_rule

# Rule templates with context-aware actions
rule_templates = {
    "low": "If demand is low, increase marketing efforts and review onboarding for improvements.",
    "normal": "If demand is normal, maintain current workflows with continuous optimization checks.",
    "high": "If demand is high, allocate additional onboarding resources and notify stakeholders.",
}


def classify_goal(goal_text):
    """
    Classify the goal text into a specific category.

    Args:
        goal_text (str): The text describing the goal.

    Returns:
        str: The classified goal category.
    """
    if "increase customer retention" in goal_text.lower():
        return "increase_customer_retention"
    elif "reduce churn" in goal_text.lower():
        return "reduce_churn"
    elif "reduce cost" in goal_text.lower():
        return "reduce_cost"
    else:
        return "unknown_goal"


def generate_rule(structured_data):
    """
    Generate a rule based on the structured data.

    Args:
        structured_data (dict): The structured data containing goal, metric, constraint, timeline.

    Returns:
        dict: The generated rule and related metadata.
    """
    goal_text = structured_data.get("goal", "Unknown goal")
    classified_goal = classify_goal(goal_text)

    if classified_goal not in rule_templates:
        raise ValueError(
            f"Invalid goal: {goal_text}. No rule available for this type of goal."
        )

    rule = rule_templates[classified_goal]
    rule_data = {
        "rule": rule,
        "class": classified_goal,
        "context": structured_data.get("context", "general"),
    }

    print(f"Generated Rule: {rule}")
    print(f"Rule Data for Distribution: {json.dumps(rule_data, indent=4)}")
    return rule_data


# RuleAgent function to generate rules based on NN output
def process_nn_output(predicted_class):
    """
    Generate a rule based on the demand class index.

    Args:
        predicted_class (int): The index predicted by the NN for "low", "normal", or "high".

    Returns:
        dict: The generated rule data for distribution.
    """
    # Class index-to-label mapping
    demand_classes = ["low", "normal", "high"]

    if predicted_class not in [0, 1, 2]:
        raise ValueError(f"Invalid predicted class index: {predicted_class}")

    demand_label = demand_classes[predicted_class]  # Convert class index to label
    print(f"Demand Class Received from NN: {demand_label}")

    # Generate a rule based on the NN output
    rule = rule_templates[demand_label]

    # Prepare rule data for distribution
    rule_data = {"rule": rule, "class": demand_label, "context": "onboarding process"}

    print(f"Generated Rule: {rule_data['rule']}")
    print(f"Rule Data for Distribution: {json.dumps(rule_data, indent=4)}")
    return rule_data


if __name__ == "__main__":
    # Example structured data
    structured_data_example = {
        "goal": "increase customer retention by 15%",
        "metric": "15%",
        "constraint": "Budget cap: $500K",
        "timeline": "Q1 2025",
        "context": "onboarding process",
    }
    process_nn_output(structured_data_example)
