# Import necessary libraries
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# Load the datasets
customers = pd.read_csv('Customers.csv')
transactions = pd.read_csv('Transactions.csv')

# Print columns to check the names
print("Customers Columns:", customers.columns)

# Prepare the data: Merge Customers and Transactions on CustomerID to get customer profiles
customer_transactions = transactions.groupby('CustomerID')['TotalValue'].sum().reset_index()

# Merge the customer profile (Customers) with the transaction data
customer_data = pd.merge(customers, customer_transactions, on='CustomerID', how='left')

# Fill any NaN values in TotalValue (e.g., for customers who have not made any transactions)
customer_data['TotalValue'] = customer_data['TotalValue'].fillna(0)

# Check if the necessary columns are present
required_columns = ['Age', 'Income', 'TotalValue']
missing_columns = [col for col in required_columns if col not in customer_data.columns]

if missing_columns:
    print(f"Warning: Missing columns in customer data: {missing_columns}")
    # If Age and Income are missing, use only TotalValue
    features = customer_data[['TotalValue']]  # Only use TotalValue if Age and Income are missing
else:
    features = customer_data[['Age', 'Income', 'TotalValue']]  # Use Age, Income, and TotalValue

# Standardize the features (important for distance/similarity calculations)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Compute the cosine similarity matrix between all customers
similarity_matrix = cosine_similarity(features_scaled)

# Function to get the top 3 lookalike customers for a given customer
def get_lookalikes(customer_idx, similarity_matrix, top_n=3):
    # Get the similarity scores for the given customer (excluding self)
    similarity_scores = similarity_matrix[customer_idx]
    similar_indices = similarity_scores.argsort()[-(top_n + 1):-1]  # Exclude self
    similar_customers = customer_data.iloc[similar_indices]
    
    if 'Age' in customer_data.columns and 'Income' in customer_data.columns:
        # Use Age, Income, and TotalValue if available
        return similar_customers[['CustomerID', 'Age', 'Income', 'TotalValue']], similarity_scores[similar_indices]
    else:
        # Use only TotalValue if Age and Income are missing
        return similar_customers[['CustomerID', 'TotalValue']], similarity_scores[similar_indices]

# Generate lookalikes for the first 20 customers (C0001 to C0020)
lookalike_dict = {}
for i in range(20):  # For customers C0001 - C0020
    lookalike_customers, scores = get_lookalikes(i, similarity_matrix)
    lookalike_dict[customer_data.iloc[i]['CustomerID']] = list(zip(lookalike_customers['CustomerID'], scores))

# Prepare the data to export to CSV
lookalike_list = []
for customer_id, similar_customers in lookalike_dict.items():
    for idx, (similar_customer_id, score) in enumerate(similar_customers):
        lookalike_list.append([customer_id, idx + 1, similar_customer_id, score])

# Create a DataFrame for the lookalikes
lookalike_df = pd.DataFrame(lookalike_list, columns=['CustomerID', 'LookalikeRank', 'LookalikeCustomerID', 'SimilarityScore'])

# Save to CSV
lookalike_df.to_csv('FirstName_LastName_Lookalike.csv', index=False)

# Output the first few rows of the lookalike data
print(lookalike_df.head())


