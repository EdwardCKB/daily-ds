from day4_titanic_cleaning import load_data
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt



def preprocess_for_clustering(df):
    """Select and scale Annual Income and Spending Score for clustering"""
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def build_kmeans(X_scaled, n_clusters):
    """Fit K-means and return the fitted model"""
    model = KMeans(n_clusters=n_clusters, random_state=42)
    model.fit(X_scaled)
    return model

def plot_clusters(X_scaled, model):
    """Scatter plot of customers colored by cluster, with centroids marked"""
    plt.figure(figsize=(8, 6))
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=model.labels_, cmap='viridis', s=40)
    plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1],
                c='red', marker='X', s=200, label='Centroids')
    plt.xlabel('Annual Income (scaled)')
    plt.ylabel('Spending Score (scaled)')
    plt.title('Customer segments (K-means, k=5)')
    plt.legend()
    plt.savefig('mall_clusters.png', dpi=150)
    plt.show()

def inertia_calculation(X_scaled):
    nums = list(range(1,11))
    inertia = {}
    for k in nums:
        model = build_kmeans(X_scaled, n_clusters=k)
        i_inertia = model.inertia_
        inertia[k] = i_inertia
    return inertia

def plot_elbow(inertia_dict):
    """Plot k vs inertia to visually find the elbow"""
    k_values = list(inertia_dict.keys())
    inertia_values = list(inertia_dict.values())
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, inertia_values, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow method for optimal k')
    plt.savefig('elbow_plot.png', dpi=150)
    plt.show()

def main():
    df = load_data("data/Mall_Customers.csv")
    X_scaled = preprocess_for_clustering(df)
    model = build_kmeans(X_scaled, n_clusters=5)
    #print(model.labels_) # which cluster (0-4) each customer/row was assigned to
    #print(model.cluster_centers_) # the (x, y) position of each cluster's centroid, in scaled space
    #plot_clusters(X_scaled, model)
    inertia_dict = inertia_calculation(X_scaled)
    plot_elbow(inertia_dict)


if __name__ == "__main__":
    main()