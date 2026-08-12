from day4_titanic_cleaning import load_data
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering


def preprocess_for_clustering(df):
    """Select and scale Annual Income and Spending Score for clustering"""
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled


def preprocess_full_features(df):
    """Encode Gender and scale all 4 relevant features"""
    df = df.copy()
    df['Gender_encoded'] = df['Gender'].map({'Male': 0, 'Female': 1})
    features = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Gender_encoded']]
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    return features_scaled
    
#========
#K means
#========
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

def plot_clusters_2d_slice(X_scaled, model):
    """Cluster on all 4 features, but visualize only Income vs Spending Score"""
    plt.figure(figsize=(8, 6))
    plt.scatter(X_scaled[:, 1], X_scaled[:, 2], c=model.labels_, cmap='viridis', s=40)
    plt.xlabel('Annual Income (scaled)')
    plt.ylabel('Spending Score (scaled)')
    plt.title('Full 4-feature clustering, viewed as Income vs Spending Score')
    plt.savefig('mall_clusters_4feature_slice.png', dpi=150)
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


#========================
#Hierarchical Clustering
#========================
def plot_dendrogram(X_scaled):
    """Build and plot a dendrogram using ward linkage"""
    # compute the full merge history (every step from 200 individual
    # points down to 1 giant cluster), using Ward linkage
    merge_sequence = linkage(X_scaled, method='ward')
    plt.figure(figsize=(10, 6))
    dendrogram(merge_sequence, color_threshold=0)
    plt.xlabel('Customers')
    plt.ylabel('Distance')
    plt.title('Dendrogram (Ward linkage)')
    plt.savefig('dendrogram.png', dpi=150)
    plt.show()
    # this plot is only for DECIDING n_clusters — find the biggest
    # gap, count how many vertical lines a cut through that gap
    # crosses, that's the number to use below

def build_heirarchical(X_scaled, n_clusters):
    """Fit Agglomerative Clustering and return the fitted model"""
    # same bottom-up merging as above, but this time it stops the
    # moment it reaches exactly n groups (chosen from the dendrogram)
    # instead of merging all the way down to 1
    model = AgglomerativeClustering(n_clusters=n_clusters)
    model.fit(X_scaled)
    return model

def plot_hierarchical_clusters(X_scaled, model):
    """Scatter plot of customers colored by hierarchical cluster assignment"""
    # model already stopped merging and assigned each point a final
    # label (0 to n-1) — no tree left to draw, just plot like K-means:
    # real coordinates, colored by that final label
    plt.figure(figsize=(8, 6))
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=model.labels_, cmap='viridis', s=40)
    plt.xlabel('Annual Income (scaled)')
    plt.ylabel('Spending Score (scaled)')
    plt.title('Customer segments (Hierarchical, n_clusters=5)')
    plt.savefig('hierarchical_clusters.png', dpi=150)
    plt.show()

def main():
    df = load_data("data/Mall_Customers.csv")
    X_scaled = preprocess_for_clustering(df)
    model = build_heirarchical(X_scaled, 5)
    #model = build_kmeans(features_scaled, n_clusters=5)
    #print(model.labels_) # which cluster (0-4) each customer/row was assigned to
    #print(model.cluster_centers_) # the (x, y) position of each cluster's centroid, in scaled space
    #plot_clusters(features_scaled, model)
    #plot_clusters_2d_slice(features_scaled, model)
    #inertia_dict = inertia_calculation(X_scaled)
    #plot_elbow(inertia_dict)
    plot_hierarchical_clusters(X_scaled, model)


if __name__ == "__main__":
    main()