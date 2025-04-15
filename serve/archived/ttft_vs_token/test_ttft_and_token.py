import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

# Load the tokenizer from the specified model.
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# Manually specify the list of CSV filenames.
csv_files = [
    "/home/ubuntu/shaotingf/LMCache/serve/results/Apr_1/test/test_ttft_10_0.csv",
    "/home/ubuntu/shaotingf/LMCache/serve/results/Apr_1/test/test_ttft_10_1.csv",
    "/home/ubuntu/shaotingf/LMCache/serve/results/Apr_1/test/test_ttft_10_02.csv",
    "/home/ubuntu/shaotingf/LMCache/serve/results/Apr_1/test/test_ttft_10_03.csv",
    "/home/ubuntu/shaotingf/LMCache/serve/results/Apr_1/test/test_ttft_10_06.csv",
]

# Dictionary to store the linear regression coefficients for each file.
linear_params = {}

# Process each CSV file.
for file in csv_files:
    # Read the CSV file into a DataFrame.
    df = pd.read_csv(file)
    
    # Lists to store token counts and ttft values.
    token_counts = []
    ttft_values = []
    
    # Process each row.
    for idx, row in df.iterrows():
        # Convert context to string (to handle NaNs or non-string types).
        context_text = str(row['context'])
        
        # Tokenize the context using the model's tokenizer.
        tokenized = tokenizer(context_text, return_tensors="pt")
        # Count tokens (assuming tokenized.input_ids shape is [1, num_tokens]).
        token_count = tokenized.input_ids.shape[1]
        token_counts.append(token_count)
        
        # Get the ttft value; convert it to float if possible.
        try:
            ttft = float(row['ttft'])
        except (ValueError, TypeError):
            ttft = np.nan
        ttft_values.append(ttft)
    
    # Add the computed values as new columns.
    df['token_count'] = token_counts
    df['ttft_numeric'] = ttft_values  # Clarify these are numeric values.
    
    # Optionally, print out token count and ttft for each row.
    print(f"\nFile: {file}")
    print(df[['token_count', 'ttft_numeric']])
    
    # Plot token count vs ttft.
    plt.figure()
    plt.scatter(df['token_count'], df['ttft_numeric'], label='Data Points')
    
    # Perform linear regression (一次函数: a first-degree function: y = ax + b).
    valid_data = df[['token_count', 'ttft_numeric']].dropna()
    if len(valid_data) > 1:
        # Use numpy's polyfit for a degree-1 polynomial fit.
        coeffs = np.polyfit(valid_data['token_count'], valid_data['ttft_numeric'], 1)
        a, b = coeffs  # a is the slope and b is the intercept.
        # Store the coefficients.
        linear_params[file] = {"a": a, "b": b}
        print(f"Linear fit for {file}: a (slope) = {a:.2f}, b (intercept) = {b:.2f}")
        
        poly1d_fn = np.poly1d(coeffs)
        # Generate x values for plotting the regression line.
        x_vals = np.linspace(valid_data['token_count'].min(), valid_data['token_count'].max(), 100)
        plt.plot(x_vals, poly1d_fn(x_vals), color='red', label=f'Linear Fit: y = {a:.2f}x + {b:.2f}')
    else:
        print(f"Not enough valid data for linear regression in {file}.")
        linear_params[file] = {"a": np.nan, "b": np.nan}
    
    plt.xlabel("Token Count")
    plt.ylabel("TTFT")
    plt.title(f"Token Count vs TTFT for {file}")
    plt.legend()
    # Save the plot as an image file.
    plt.savefig(f"{file}_token_ttft_plot.png")
    plt.close()

# Save the linear coefficients for each file to a CSV file.
params_df = pd.DataFrame.from_dict(linear_params, orient="index")
params_df.index.name = "filename"
params_df.reset_index(inplace=True)
params_df.to_csv("linear_coefficients_updated.csv", index=False)
print("\nSaved linear coefficients for each file to linear_coefficients_updated.csv")

