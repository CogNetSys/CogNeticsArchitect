import numpy as np
import pandas as pd
import random


# Function to generate synthetic onboarding data
def generate_training_data(num_years=5, base_requests=1000, seasonal_spike=0.3):
    data = []
    quarters = [1, 2, 3, 4]

    for year in range(2020, 2020 + num_years):
        for quarter in quarters:
            # Simulate seasonal spike in Q1 and Q4
            if quarter in [1, 4]:
                demand = base_requests + int(
                    base_requests * seasonal_spike * random.uniform(0.8, 1.2)
                )
            else:
                demand = base_requests + int(base_requests * random.uniform(-0.2, 0.2))

            # Classify demand level
            if demand > base_requests * 1.2:
                demand_class = "high"
            elif demand < base_requests * 0.8:
                demand_class = "low"
            else:
                demand_class = "normal"

            data.append(
                {
                    "Year": year,
                    "Quarter": quarter,
                    "Requests": demand,
                    "Class": demand_class,
                }
            )

    # Convert to DataFrame for easy saving and viewing
    df = pd.DataFrame(data)
    return df


# Generate synthetic data
df = generate_training_data(num_years=10, base_requests=1000, seasonal_spike=0.4)

# Save the data to a CSV file for reference
df.to_csv("synthetic_training_data.csv", index=False)
print("Synthetic training data saved as 'synthetic_training_data.csv'")
print(df.head(10))
