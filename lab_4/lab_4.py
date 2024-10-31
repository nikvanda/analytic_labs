import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, preprocessing, metrics
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans, MeanShift, DBSCAN, AffinityPropagation

S1, S2, S3 = 9, 2, 1
METHODS = ['single', 'complete', 'average', 'ward']
N_CLUSTERS_RANGE = range(1, 11)
BANDWIDTH_RANGE = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05]
EPS_RANGE = [0.005, 0.006, 0.007, 0.008, 0.009, 0.01]
MIN_SAMPLES_RANGE = range(1, 11)
DAMPING_RANGE = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]


# Функция для добавления n_clusters в кортежи
def add_n_clusters_if_present(results, key, is_dbscan=False):
    if results['n_clusters']:
        if is_dbscan:
            return [(tup[0], tup[1], round(float(tup[2]), 3), results['n_clusters'][i]) for i, tup in enumerate(results[key])]
        else:
            return [(tup[0], round(float(tup[1]), 3), results['n_clusters'][i]) for i, tup in enumerate(results[key])]
    else:
        return [(tup[0], round(float(tup[1]), 3)) if len(tup) == 2 else (tup[0], tup[1], round(float(tup[2]), 3)) for tup in results[key]]


def table_1(results, filename, is_dbscan=False):
    # Проверяем и добавляем n_clusters, если они есть
    results['adjusted_rand'] = add_n_clusters_if_present(results, 'adjusted_rand', is_dbscan)
    results['jaccard'] = add_n_clusters_if_present(results, 'jaccard', is_dbscan)
    results['fowlkes_mallows'] = add_n_clusters_if_present(results, 'fowlkes_mallows', is_dbscan)

    df = pd.DataFrame({
        'Adjusted Rand': results['adjusted_rand'],
        'Jaccard': results['jaccard'],
        'Fowlkes Mallows': results['fowlkes_mallows']
    })
    df.to_excel(filename, index=False)


def compare_plot(results_dict, metrics):
    for metric in metrics:
        plt.figure(figsize=(10, 7))
        for alg_name, result in results_dict.items():
            if 'n_clusters' in result and result['n_clusters']:
                x = result['n_clusters']
            else:
                x = [tup[0] for tup in result[metric]]
            y = [tup[-1] for tup in result[metric]]
            plt.scatter(x, y, label=f'{alg_name} - {metric}')

        plt.title(f'{metric.capitalize()} score vs. Number of Clusters')
        plt.xlabel('Number of Clusters')
        plt.ylabel(f'{metric.capitalize()} score')
        plt.legend()
        plt.grid(True)
        plt.show()


# Пример использования:
algorithms = ['MeanShift', 'DBSCAN', 'Affinity']
metrics = ['adjusted_rand', 'jaccard', 'fowlkes_mallows']


def plot_agglomerate(results, metric_name, n_clusters=None):
    plt.figure(figsize=(10, 6))
    for method in METHODS:
        method_results = [(n_clusters, score) for n_clusters, m, score in results if m == method]
        n_clusters_values, scores = zip(*method_results)
        plt.plot(n_clusters_values, scores, label=f'{method} linkage', marker='o')

    plt.title(f'{metric_name} for Different Linkage Methods')
    plt.xlabel('Number of Clusters')
    plt.ylabel(metric_name)
    plt.grid()
    plt.legend()
    plt.show()


def plot_k_means(results, metric_name, n_clusters=None):
    n_clusters, score = zip(*results)
    plt.plot(n_clusters, score, marker='o')
    plt.title(f'{metric_name} for KMeans')
    plt.xlabel('Number of Clusters')
    plt.ylabel(metric_name)
    plt.grid()
    plt.show()


def plot_mean_shift(results, metric_name, n_clusters=None):
    bandwidth, score = zip(*results)
    plt.plot(bandwidth, score, marker='o')
    for i in range(len(bandwidth)):
        plt.text(bandwidth[i], score[i], f'n_clusters={n_clusters[i]}', fontsize=9, ha='right')
    plt.title(f'{metric_name} for Mean Shift')
    plt.xlabel('Bandwidth')
    plt.ylabel(metric_name)
    plt.grid()
    plt.show()


def plot_dbscan(results, metric_name, n_clusters=None):
    eps_vals, min_samples_vals, scores = zip(*results)
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(eps_vals, min_samples_vals, c=scores, cmap='viridis', s=100)
    plt.colorbar(sc, label=metric_name)
    for i in range(len(eps_vals)):
        plt.text(eps_vals[i], min_samples_vals[i], f'{scores[i]:.2f}', fontsize=9,
                 ha='right', va='bottom')
    plt.xlabel('eps')
    plt.ylabel('min_samples')
    plt.title(f'{metric_name} for DBSCAN')
    plt.show()


def plot_affinity(results, metric_name, n_clusters=None):
    dampings, scores = zip(*results)
    plt.figure(figsize=(10, 6))
    plt.plot(dampings, scores, marker='o')
    for i in range(len(dampings)):
        plt.text(dampings[i], scores[i], f'n_clusters={n_clusters[i]}', fontsize=9, ha='right')
    plt.xlabel('Damping')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} for Affinity Propagation')
    plt.grid()
    plt.show()


def prepared_data(df):
    target = df.target
    new_df = pd.concat([
        df[target == 0][S1:S1 + 40],
        df[target == 1][S2:S2 + 40],
        df[target == 2][S3:S3 + 40]])
    return new_df, new_df['target']


def task_2(normalize_arr):
    for method in METHODS:
        linked = linkage(normalize_arr, method=method)
        plt.figure(figsize=(10, 7))
        dendrogram(linked)
        plt.title(f'Dendrogram using {method} linkage')
    plt.xlabel('Samples')
    plt.ylabel('Distance')
    plt.show()


def agglomerative_model(params):
    n_clusters, method = params
    return AgglomerativeClustering(n_clusters=n_clusters, linkage=method)


def k_means_model(n_clusters):
    return KMeans(n_clusters=n_clusters)


def mean_shift_model(bandwidth):
    return MeanShift(bandwidth=bandwidth)


def dbscan_model(params):
    eps, min_samples = params
    return DBSCAN(eps=eps, min_samples=min_samples)


def affinity_model(damping):
    return AffinityPropagation(damping=damping)


def perform_clustering(model, data, y_true, include_n_clusters=False):
    y_pred = model.fit_predict(data)
    adjusted_rand = metrics.adjusted_rand_score(y_true, y_pred)
    jaccard = metrics.jaccard_score(y_true, y_pred, average='weighted', zero_division=0)
    fowlkes_mallows = metrics.fowlkes_mallows_score(y_true, y_pred)
    if include_n_clusters:
        n_clusters = len(set(y_pred)) - (1 if -1 in y_pred else 0)
    else:
        n_clusters = None
    return adjusted_rand, jaccard, fowlkes_mallows, n_clusters


def evaluate_clustering(data, y_true, model_fn, param_range, include_n_clusters=False):
    results = {'adjusted_rand': [], 'jaccard': [], 'fowlkes_mallows': [],
               'n_clusters': []}
    for param in param_range:
        model = model_fn(param)
        adjusted_rand, jaccard, fowlkes_mallows, n_clusters = perform_clustering(model, data, y_true,
                                                                                 include_n_clusters)
        if isinstance(param, tuple):
            results['adjusted_rand'].append((*param, adjusted_rand))
            results['jaccard'].append((*param, jaccard))
            results['fowlkes_mallows'].append((*param, fowlkes_mallows))
        else:
            results['adjusted_rand'].append((param, adjusted_rand))
            results['jaccard'].append((param, jaccard))
            results['fowlkes_mallows'].append((param, fowlkes_mallows))
        if include_n_clusters and n_clusters is not None:
            results['n_clusters'].append(n_clusters)
    return results


def get_best_values(results, is_clusters=False):
    best_adjusted_rand = max(results['adjusted_rand'], key=lambda x: x[-1])
    best_jaccard = max(results['jaccard'], key=lambda x: x[-1])
    best_fowlkes_mallows = max(results['fowlkes_mallows'], key=lambda x: x[-1])
    if is_clusters:
        index_adjusted_rand = results['adjusted_rand'].index(best_adjusted_rand)
        index_jaccard = results['jaccard'].index(best_jaccard)
        index_fowlkes = results['fowlkes_mallows'].index(best_fowlkes_mallows)
    else:
        index_adjusted_rand = None
        index_jaccard = None
        index_fowlkes = None
    return best_adjusted_rand, best_jaccard, best_fowlkes_mallows, index_adjusted_rand, index_jaccard, index_fowlkes


def visualize_results(results, plot_fn, param_name):
    plot_fn(results['adjusted_rand'], f'Adjusted Rand Score VS {param_name}', results['n_clusters'])
    plot_fn(results['jaccard'], f'Jaccard Score VS {param_name}', results['n_clusters'])
    plot_fn(results['fowlkes_mallows'], f'Fowlkes VS Mallows Score{param_name}', results['n_clusters'])


def main():
    load_wine = datasets.load_wine(as_frame=True)['frame']
    wine_df, y_true = prepared_data(load_wine)
    wine_df = wine_df.drop('target', axis=1)
    normalize_arr = preprocessing.Normalizer().fit_transform(wine_df)
    task_2(normalize_arr)
    agglomerative_params = [(n_clusters, method) for method in METHODS for n_clusters in N_CLUSTERS_RANGE]
    agglomerative_results = evaluate_clustering(normalize_arr, y_true,
                                                agglomerative_model, agglomerative_params)
    best_rand, best_jaccard, best_fowlkes, _, _, _ = get_best_values(agglomerative_results)
    print(f'Best Adjusted Rand Score: {best_rand[2]:.4f} -----> n_clusters = {best_rand[0]}, linkage = {best_rand[1]}')
    print(
        f'Best Jaccard Score: {best_jaccard[2]:.4f} -----> n_clusters = {best_jaccard[0]}, linkage = {best_jaccard[1]}')
    print(
        f'Best Fowlkes Mallows Score: {best_fowlkes[2]:.4f} -----> n_clusters = {best_fowlkes[0]}, linkage = {best_fowlkes[1]}\n')
    kmeans_results = evaluate_clustering(normalize_arr, y_true, k_means_model, N_CLUSTERS_RANGE)
    best_rand, best_jaccard, best_fowlkes, _, _, _ = get_best_values(kmeans_results)
    print(f'Best Adjusted Rand Score: {best_rand[1]:.4f} -----> n_clusters = {best_rand[0]}')
    print(f'Best Jaccard Score: {best_jaccard[1]:.4f} -----> n_clusters = {best_jaccard[0]}')
    print(f'Best Fowlkes Mallows Score: {best_fowlkes[1]:.4f} -----> n_clusters = {best_fowlkes[0]}\n')
    meanshift_results = evaluate_clustering(normalize_arr, y_true,
                                            mean_shift_model, BANDWIDTH_RANGE, True)
    best_rand, best_jaccard, best_fowlkes, i_rand, i_jacc, i_fowl = get_best_values(meanshift_results, True)
    print(
        f'Best Adjusted Rand Score: {best_rand[1]:.4f} -----> bandwidth = {best_rand[0]}, 'f'n_cluster = {meanshift_results["n_clusters"][i_rand]}')
    print(
        f'Best Jaccard Score: {best_jaccard[1]:.4f} -----> bandwidth = {best_jaccard[0]}, 'f'n_cluster = {meanshift_results["n_clusters"][i_jacc]}')
    print(
        f'Best Fowlkes Mallows Score: {best_fowlkes[1]:.4f} -----> bandwidth = {best_fowlkes[0]}, 'f'n_cluster = {meanshift_results["n_clusters"][i_fowl]}\n')
    dbscan_params = [(eps, min_samples) for eps in EPS_RANGE for min_samples in MIN_SAMPLES_RANGE]
    dbscan_results = evaluate_clustering(normalize_arr, y_true, dbscan_model, dbscan_params, True)
    best_rand, best_jaccard, best_fowlkes, i_rand, i_jacc, i_fowl = get_best_values(dbscan_results, True)
    print(
        f'Best Adjusted Rand Score: {best_rand[2]:.4f} -----> eps = {best_rand[0]}, 'f'min_samples = {best_rand[1]}, n_clusters = {dbscan_results["n_clusters"][i_rand]}')
    print(
        f'Best Jaccard Score: {best_jaccard[2]:.4f} -----> eps = {best_jaccard[0]}, 'f'min_samples = {best_jaccard[1]}, n_clusters ={dbscan_results["n_clusters"][i_jacc]}')
    print(
        f'Best Fowlkes Mallows Score: {best_fowlkes[2]:.4f} -----> eps = {best_fowlkes[0]}, 'f'min_samples = {best_fowlkes[1]}, n_clusters ={dbscan_results["n_clusters"][i_fowl]}\n')
    affinity_results = evaluate_clustering(normalize_arr, y_true, affinity_model, DAMPING_RANGE, True)
    best_rand, best_jaccard, best_fowlkes, i_rand, i_jacc, i_fowl = get_best_values(affinity_results, True)
    print(
        f'Best Adjusted Rand Score: {best_rand[1]:.4f} -----> damping = {best_rand[0]} 'f'n_clusters = {affinity_results["n_clusters"][i_rand]}')
    print(
        f'Best Jaccard Score: {best_jaccard[1]:.4f} -----> damping = {best_jaccard[0]} 'f'n_clusters = {affinity_results["n_clusters"][i_jacc]}')
    print(f'Best Fowlkes Mallows Score: {best_fowlkes[1]:.4f} -----> damping= {best_fowlkes[0]} '
          f'n_clusters = {affinity_results["n_clusters"][i_fowl]}\n')
    visualize_results(agglomerative_results, plot_agglomerate, 'N_CLUSTERS')
    visualize_results(kmeans_results, plot_k_means, 'N_CLUSTERS')
    visualize_results(meanshift_results, plot_mean_shift, 'BANDWIDTH')
    visualize_results(dbscan_results, plot_dbscan, 'EPS and MIN_SAMPLES')
    visualize_results(affinity_results, plot_affinity, 'DAMPING')
    results_dict = {
        'Agglomerative': agglomerative_results,
        'KMeans': kmeans_results,
        'MeanShift': meanshift_results,
        'DBSCAN': dbscan_results,
        'Affinity': affinity_results
    }
    metrics_ = ['adjusted_rand', 'jaccard', 'fowlkes_mallows']
    compare_plot(results_dict, metrics_)


if __name__ == '__main__':
    main()
