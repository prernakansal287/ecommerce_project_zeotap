# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the datasets
customers = pd.read_csv('Customers.csv')
products = pd.read_csv('Products.csv')
transactions = pd.read_csv('Transactions.csv')

# Debugging - Check the first few rows of each dataset
print("Customers Dataset:")
print(customers.head())
print(customers.info())
print("\nProducts Dataset:")
print(products.head())
print(products.info())
print("\nTransactions Dataset:")
print(transactions.head())
print(transactions.info())

# Parse dates in Transactions.csv
transactions['TransactionDate'] = pd.to_datetime(transactions['TransactionDate'], errors='coerce')

# Check for missing or invalid values in Transactions
print("\nTransactions - Check for missing values:")
print(transactions.isnull().sum())

# Drop rows with invalid or missing TransactionDate
transactions = transactions.dropna(subset=['TransactionDate'])

# Ensure Quantity and TotalValue are numeric
transactions['Quantity'] = pd.to_numeric(transactions['Quantity'], errors='coerce')
transactions['TotalValue'] = pd.to_numeric(transactions['TotalValue'], errors='coerce')

# Drop rows with invalid Quantity or TotalValue
transactions = transactions.dropna(subset=['Quantity', 'TotalValue'])

# Add a new column for the month of each transaction
transactions['TransactionMonth'] = transactions['TransactionDate'].dt.to_period('M').astype(str)

# Group data by month to analyze monthly sales
monthly_sales = transactions.groupby('TransactionMonth', as_index=False)['TotalValue'].sum()

# Debugging - Check the cleaned and grouped data
print("\nMonthly Sales Data:")
print(monthly_sales.head())
print(monthly_sales.info())

# Plot monthly sales trends
sns.lineplot(data=monthly_sales, x='TransactionMonth', y='TotalValue', marker='o')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Show the plot
plt.title("Monthly Sales Trend")
plt.xlabel("Transaction Month")
plt.ylabel("Total Sales Value (USD)")
plt.show()
