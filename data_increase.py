import pandas as pd
import random

# Load your existing dataset
df = pd.read_csv('instagram_data.csv')

# Function to generate random data variations
def generate_variation(row):
    new_row = row.copy()
    new_row['followers'] = random.randint(2000, 25000)
    new_row['likes'] = random.randint(50, 2000)
    new_row['comments'] = random.randint(0, 200)
    new_row['hashtags_count'] = random.randint(1, 20)
    new_row['reach'] = new_row['followers'] * random.uniform(0.3, 0.7)  # Random reach
    return new_row

# Increase dataset by generating 2000 new samples
new_rows = [generate_variation(row) for _, row in df.iterrows() for _ in range(100)]  # Add 100 variations per row to reach 2000 new rows

# Add synthetic data to the original dataframe
expanded_df = pd.DataFrame(new_rows)
expanded_df = pd.concat([df, expanded_df], ignore_index=True)

# Save the new dataset
expanded_df.to_csv('expanded_instagram_data.csv', index=False)




