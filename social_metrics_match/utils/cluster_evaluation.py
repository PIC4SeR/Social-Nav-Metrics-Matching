from sklearn.decomposition import PCA
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import silhouette_score, adjusted_rand_score, confusion_matrix

def cluster_comparison_using_ARI(array_of_features_,use_pca=True,n_clusters=3, name='array'):
    """
    Given the array of features, executes clusering using two techniques: k-means and hierarchical clustering.
    Computes ARI which shows how closely the cluster labels match the preassigned labels
    """
    if use_pca:
        pca = PCA(n_components=0.95)
        array_of_features = pca.fit_transform(array_of_features_)
    else:
        array_of_features = array_of_features_
    cluster_labels_k_means = cluster_K_means(array_of_features, n_clusters=n_clusters )    
    cluster_labels_linkage = cluster_hierarchical_K_means(array_of_features, method='complete')  # or 'complete', 'average', etc.

    manual_labels=["Bad","Intermediate","Good","Good","Intermediate","Bad","Intermediate","Bad","Good","Intermediate","Good","Bad",
                   "Bad","Intermediate","Good","Good","Intermediate","Bad","Intermediate","Bad","Good","Intermediate","Good","Bad"]
    manual_encoded = pd.Categorical(manual_labels, categories=["Bad", "Intermediate", "Good"]).codes

    ari_linkage = adjusted_rand_score(manual_encoded, cluster_labels_linkage)
    print(f"Adjusted Rand Index hierarchical for {name}: {ari_linkage:.2f}")
    ari_k_means = adjusted_rand_score(manual_encoded, cluster_labels_k_means)
    print(f"Adjusted Rand Index k-means for {name}: {ari_k_means:.2f}")
    return ari_k_means, ari_linkage

def cluster_K_means(arr, K=3):
    kmeans= KMeans(n_clusters=K, random_state=42, init='k-means++')
    cluster_labels = kmeans.fit_predict(arr)
    return cluster_labels

def cluster_hierarchical_K_means(arr, K=3):
    Z = linkage(arr, method='complete')  # or 'complete', 'average', etc.
    cluster_labels = fcluster(Z, t=K, criterion='maxclust')
    return cluster_labels

def run_TSNE(arr, n=2, p=6):
    tsne_2d = TSNE(n_components = n, random_state=42, perplexity=p)
    tsne_results_2d = tsne_2d.fit_transform(arr)
    return tsne_results_2d