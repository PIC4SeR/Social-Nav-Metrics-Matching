from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import silhouette_score, adjusted_rand_score, confusion_matrix

def cluster_comparison_using_ARI(labels_to_analyze : np.array, name='array'):
    """
    Given the labels fed as input, computes the adjusted rand score
    Computes ARI which shows how closely the cluster labels match the preassigned labels
    """

    manual_labels=["Bad","Intermediate","Good","Good","Intermediate","Bad","Intermediate","Bad","Good","Good","Intermediate","Bad",
                   "Bad","Intermediate","Good","Good","Intermediate","Bad","Intermediate","Bad","Good","Intermediate","Good","Bad"]
    manual_encoded = pd.Categorical(manual_labels, categories=["Bad", "Intermediate", "Good"]).codes

    ari_labels = adjusted_rand_score(manual_encoded, labels_to_analyze)
    print(f"Adjusted Rand Index k-means for {name}: {ari_labels:.2f}")

    return ari_labels

def cluster_K_means(arr : np.array, K=3):
    """
    Computes the k-means clustering in K-clusters, output is as an array with the assigned labels
    Input: array of samples to cluster
    Output: array of labels assigned
    """
    kmeans= KMeans(n_clusters=K, random_state=42, init='k-means++')
    cluster_labels = kmeans.fit_predict(arr)
    return cluster_labels

def cluster_hierarchical_K_means(arr : np.array, K=3):
    """
    Computes the hierarchical clustering in K-clusters, output is as an array with the assigned labels

    Input: array of samples to cluster

    Output: array of labels assigned
    """
    Z = linkage(arr, method='complete')  # or 'complete', 'average', etc.
    cluster_labels = fcluster(Z, t=K, criterion='maxclust')
    return cluster_labels

def run_TSNE(arr : np.array, n=2, p=6):
    """
    Reduces the dimensionality of input array to n-dimensions using t-SNE

    Input: array of samples to be considered

    Output: array of samples with n-features
    """
    tsne_2d = TSNE(n_components = n, random_state=42, perplexity=p)
    tsne_results_2d = tsne_2d.fit_transform(arr)
    return tsne_results_2d

def run_PCA(arr: np.array, n_components: int = 2):
    """
    Reduces the dimensionality of the input array to n_components using PCA.
    
    Input:
      - arr: array of samples to be considered
      - n_components: the number of principal components to keep

    Output: array of samples with n_components features
    """
    pca = PCA(n_components= n_components, random_state=42)
    pca_results = pca.fit_transform(arr)
    return pca_results