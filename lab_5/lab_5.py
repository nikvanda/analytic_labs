from sklearn.cluster import AffinityPropagation
from sklearn.random_projection import GaussianRandomProjection
from sklearn import datasets, preprocessing, metrics
from sklearn.manifold import Isomap, MDS, SpectralEmbedding, TSNE, LocallyLinearEmbedding
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import random

S = 5
C = [0, 2, 3, 4, 5, 7]
N_COMPONENTS = range(3, 13)


def prepared_data(df):
    filtered_data, target = [], df.target
    for c in C:
        class_data = df[target == c].iloc[S:S + 100]
        filtered_data.append(class_data)
    new_df = pd.concat(filtered_data).reset_index(drop=True)
    return new_df, new_df.target


def visualize_initial_data(df, target):
    for c in C:
        class_data = df[target == c].drop('target', axis=1)
        for _ in range(2):
            random_number = random.randint(0, 99)
            image_data = class_data.iloc[random_number].values.reshape(8, 8)
            plt.title(f'Class {c}, Sample {random_number}')
            plt.imshow(image_data, cmap='gray')
            plt.show()


def plot_data(data, labels, title, n):
    if n == 2:
        plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', edgecolor='k')
        plt.colorbar(label="Classes")
        plt.title(title)
        plt.show()
    elif n == 3:
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap='viridis', edgecolor='k')
        fig.colorbar(scatter, label="Classes")
        ax.set_title(title)
        plt.show()


def pca(norm_arr, n, labels):
    pca_model = PCA(n_components=n)
    data = pca_model.fit_transform(norm_arr)
    plot_data(data, labels, 'PCA', n)
    return data


def gaussian_random(norm_arr, n, labels):
    gaussian_random_model = GaussianRandomProjection(n_components=n)
    data = gaussian_random_model.fit_transform(norm_arr)
    plot_data(data, labels, 'Gaussian Random Projection', n)
    return data


def mds(norm_arr, n, labels):
    mds_model = MDS(n_components=n)
    data = mds_model.fit_transform(norm_arr)
    plot_data(data, labels, 'MDS', n)
    return data


def tsne(norm_arr, n, labels):
    tsne_model = TSNE(n_components=n)
    data = tsne_model.fit_transform(norm_arr)
    plot_data(data, labels, 'tSNE', n)
    return data


def isomap(norm_arr, n, labels):
    isomap_model = Isomap(n_components=n)
    data = isomap_model.fit_transform(norm_arr)
    plot_data(data, labels, 'Isomap', n)
    return data


def spectral_embedding(norm_arr, n, labels):
    spectral_embedding_model = SpectralEmbedding(n_components=n)
    data = spectral_embedding_model.fit_transform(norm_arr)
    plot_data(data, labels, 'Spectral Embedding', n)
    return data


def locally_linear_embedding(norm_arr, n, labels):
    loc_lin_embed_model = LocallyLinearEmbedding(n_components=n)
    data = loc_lin_embed_model.fit_transform(norm_arr)
    plot_data(data, labels, 'Locally Linear Embedding', n)
    return data


def affinity_clustering(norm_arr, labels, methods):
    results = []
    model = AffinityPropagation(damping=0.75)

    for n_component in N_COMPONENTS:
        for method in methods:
            method_name = method.__name__
            data = method(norm_arr, n_component, labels)
            y_pred = model.fit_predict(data)

            n_clusters = len(model.cluster_centers_indices_)
            fowlkes = metrics.fowlkes_mallows_score(labels, y_pred)
            results.append((method_name, n_component, fowlkes, n_clusters))

    df_results = pd.DataFrame(results, columns=['Method', 'Dimensions', 'Fowlkes Mallows Score', 'Num Clusters'])
    df_results.to_excel("Сlustering results.xlsx", index=False)
    return df_results


def plot_results(df_results):
    for method in df_results['Method'].unique():
        method_data = df_results[df_results['Method'] == method]
        plt.plot(method_data['Dimensions'], method_data['Fowlkes Mallows Score'], marker='o', label=method)

    plt.xlabel('Dimensions')
    plt.ylabel('Fowlkes Mallows Score')
    plt.legend(title="Methods")
    plt.title("Dependence of Clustering Quality on Dimensionality")
    plt.show()


def main():
    digits = datasets.load_digits(as_frame=True)['frame']
    new_digits, labels = prepared_data(digits)
    visualize_initial_data(new_digits, new_digits.target)

    new_digits = new_digits.drop('target', axis=1)
    normalize_arr = preprocessing.Normalizer().fit_transform(new_digits)

    methods_5_3 = [pca, gaussian_random, mds, tsne]
    dimensions = [2, 3]
    for method in methods_5_3:
        for dim in dimensions:
            method(normalize_arr, dim, labels)

    methods_5_4 = [pca, gaussian_random, mds, isomap, spectral_embedding, locally_linear_embedding]
    df_results = affinity_clustering(normalize_arr, labels, methods_5_4)
    print(df_results)
    plot_results(df_results)


if __name__ == '__main__':
    main()
