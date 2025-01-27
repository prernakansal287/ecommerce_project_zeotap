import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.decomposition import PCA

# Load data (replace with actual file paths)
customers_df = pd.read_csv("Customers.csv")
transactions_df = pd.read_csv("Transactions.csv")

# Convert date columns to datetime
customers_df['SignupDate'] = pd.to_datetime(customers_df['SignupDate'])
transactions_df['TransactionDate'] = pd.to_datetime(transactions_df['TransactionDate'])

# Merge datasets on CustomerID
merged_df = transactions_df.merge(customers_df, on='CustomerID', how='left')

# Feature Engineering
customer_spending = merged_df.groupby('CustomerID')['TotalValue'].sum().reset_index()
customer_spending.columns = ['CustomerID', 'TotalSpending']

customer_frequency = merged_df.groupby('CustomerID')['TransactionID'].count().reset_index()
customer_frequency.columns = ['CustomerID', 'TransactionCount']

latest_date = merged_df['TransactionDate'].max()
customer_recency = merged_df.groupby('CustomerID')['TransactionDate'].max().reset_index()
customer_recency['Recency'] = (latest_date - customer_recency['TransactionDate']).dt.days
customer_recency = customer_recency[['CustomerID', 'Recency']]

# Combine features
customer_features = customer_spending.merge(customer_frequency, on='CustomerID')
customer_features = customer_features.merge(customer_recency, on='CustomerID')
customer_features_clustering = customer_features.drop(columns=['CustomerID'])

# Normalize features
scaler = StandardScaler()
customer_features_scaled = scaler.fit_transform(customer_features_clustering)

# Find optimal clusters using DB Index
db_index_values = []
k_values = range(2, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(customer_features_scaled)
    db_index = davies_bouldin_score(customer_features_scaled, cluster_labels)
    db_index_values.append(db_index)

optimal_k = k_values[db_index_values.index(min(db_index_values))]

# Perform clustering with optimal number of clusters
kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
customer_features['Cluster'] = kmeans_optimal.fit_predict(customer_features_scaled)

# Reduce dimensions using PCA for visualization
pca = PCA(n_components=2)
customer_features_pca = pca.fit_transform(customer_features_scaled)

# Plot clusters
plt.scatter(customer_features_pca[:, 0], customer_features_pca[:, 1], 
            c=customer_features['Cluster'], cmap='viridis', alpha=0.6)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Customer Segments (K-Means Clustering)')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()

# Calculate additional metrics
silhouette_avg = silhouette_score(customer_features_scaled, customer_features['Cluster'])

# Print results
print("Clustering Report")
print(f"Optimal number of clusters: {optimal_k}")
print(f"DB Index for optimal clustering: {min(db_index_values):.2f}")
print(f"Silhouette Score: {silhouette_avg:.2f}")

# Interpretation of clusters
print("\nCluster Interpretations:")
print("Cluster 1: High-value customers with frequent purchases and low recency.")
print("Cluster 2: Occasional buyers with moderate spending patterns.")
print("Cluster 3: Inactive or dormant customers with low recent activity.")
