import pandas as pd

# Replace 'your_file.csv' with the path to your CSV file.
csv_file = "/home/ubuntu/shaotingf/LMCache/serve/results/Apr_1/1.csv"
df = pd.read_csv(csv_file)

# Display the loaded DataFrame
print("Loaded DataFrame:")
print(df)

# Group by 'context' and collect the corresponding 'index_in_dataset' values as lists.
grouped = df.groupby("context")["index_in_dataset"].apply(list).reset_index()

# Display the grouped result.
print("\nGrouped by context:")
print(grouped)

# Save the grouped DataFrame to a new CSV file.
output_file = "grouped_output.csv"  # Change this name or path as desired.
grouped.to_csv(output_file, index=False)

print(f"\nThe grouped data has been saved to {output_file}.")

