import pandas as pd
from sklearn import datasets, preprocessing, metrics
from sklearn.cluster import AgglomerativeClustering, KMeans, MeanShift, DBSCAN, AffinityPropagation


class WineClustering:
    CLUSTERING_METHODS = ['single', 'complete', 'average', 'ward']
    CLUSTER_COUNT_RANGE = range(1, 11)
    BANDWIDTH_OPTIONS = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05]
    EPS_OPTIONS = [0.005, 0.006, 0.007, 0.008, 0.009, 0.01]
    MIN_SAMPLES_OPTIONS = range(1, 11)
    DAMPING_OPTIONS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    def __init__(self, wine_index1: int, wine_index2: int, wine_index3: int):
        wine_data = datasets.load_wine(as_frame=True)['frame']
        self.index1, self.index2, self.index3 = wine_index1, wine_index2, wine_index3

        wine_group0 = wine_data[wine_data.target == 0].iloc[self.index1:self.index1 + 40]
        wine_group1 = wine_data[wine_data.target == 1].iloc[self.index2:self.index2 + 40]
        wine_group2 = wine_data[wine_data.target == 2].iloc[self.index3:self.index3 + 40]

        combined_wine = pd.concat([wine_group0, wine_group1, wine_group2])
        self.wine_features, self.wine_target = combined_wine.drop('target', axis=1).values, combined_wine[
            'target'].values
        self.normalized_wine_data = preprocessing.Normalizer().fit_transform(self.wine_features)

    def evaluate_clustering(self, model_function, parameter_range, include_cluster_count=False):
        evaluation_results = {
            'adjusted_rand_index': [],
            'jaccard_score': [],
            'fowlkes_mallows_index': [],
            'cluster_count': []
        }
        for parameter in parameter_range:
            clustering_model = model_function(parameter)
            adjusted_rand, jaccard, fowlkes_mallows, cluster_count = self.perform_clustering(clustering_model,
                                                                                             include_cluster_count)
            if isinstance(parameter, tuple):
                evaluation_results['adjusted_rand_index'].append((*parameter, adjusted_rand))
                evaluation_results['jaccard_score'].append((*parameter, jaccard))
                evaluation_results['fowlkes_mallows_index'].append((*parameter, fowlkes_mallows))
            else:
                evaluation_results['adjusted_rand_index'].append((parameter, adjusted_rand))
                evaluation_results['jaccard_score'].append((parameter, jaccard))
                evaluation_results['fowlkes_mallows_index'].append((parameter, fowlkes_mallows))
            if include_cluster_count and cluster_count is not None:
                evaluation_results['cluster_count'].append(cluster_count)
        return evaluation_results

    def perform_clustering(self, model, include_cluster_count=False):
        predicted_labels = model.fit_predict(self.normalized_wine_data)
        adjusted_rand_index = metrics.adjusted_rand_score(self.wine_target, predicted_labels)
        jaccard_score = metrics.jaccard_score(self.wine_target, predicted_labels, average='weighted', zero_division=0)
        fowlkes_mallows_index = metrics.fowlkes_mallows_score(self.wine_target, predicted_labels)

        cluster_count = len(set(predicted_labels)) - (
            1 if -1 in predicted_labels else 0) if include_cluster_count else None
        return adjusted_rand_index, jaccard_score, fowlkes_mallows_index, cluster_count


def get_best_results(evaluation_results, is_cluster_count=False):
    best_adjusted_rand = max(evaluation_results['adjusted_rand_index'], key=lambda x: x[-1])
    best_jaccard = max(evaluation_results['jaccard_score'], key=lambda x: x[-1])
    best_fowlkes_mallows = max(evaluation_results['fowlkes_mallows_index'], key=lambda x: x[-1])

    if is_cluster_count:
        index_adjusted_rand = evaluation_results['adjusted_rand_index'].index(best_adjusted_rand)
        index_jaccard = evaluation_results['jaccard_score'].index(best_jaccard)
        index_fowlkes = evaluation_results['fowlkes_mallows_index'].index(best_fowlkes_mallows)
    else:
        index_adjusted_rand = None
        index_jaccard = None
        index_fowlkes = None

    return best_adjusted_rand, best_jaccard, best_fowlkes_mallows, index_adjusted_rand, index_jaccard, index_fowlkes


def main():
    s1, s2, s3 = 9, 2, 1
    wine_clustering = WineClustering(s1, s2, s3)
    agglomerative_parameters = [(n_clusters, method) for method in wine_clustering.CLUSTERING_METHODS for n_clusters in
                                wine_clustering.CLUSTER_COUNT_RANGE]
    agglomerative_results = wine_clustering.evaluate_clustering(AgglomerativeClustering, agglomerative_parameters)
    best_rand, best_jaccard, best_fowlkes, *_ = get_best_results(agglomerative_results)

    kmeans_results = wine_clustering.evaluate_clustering(KMeans, wine_clustering.CLUSTER_COUNT_RANGE)
    best_rand, best_jaccard, best_fowlkes, *_ = get_best_results(kmeans_results)

    meanshift_results = wine_clustering.evaluate_clustering(MeanShift, wine_clustering.BANDWIDTH_OPTIONS, True)
    best_rand, best_jaccard, best_fowlkes, i_rand, i_jacc, i_fowl = get_best_results(meanshift_results, True)

    dbscan_parameters = [(eps, min_samples) for eps in wine_clustering.EPS_OPTIONS for min_samples in
                         wine_clustering.MIN_SAMPLES_OPTIONS]
    dbscan_results = wine_clustering.evaluate_clustering(DBSCAN, dbscan_parameters, True)
    best_rand, best_jaccard, best_fowlkes, i_rand, i_jacc, i_fowl = get_best_results(dbscan_results, True)

    affinity_results = wine_clustering.evaluate_clustering(AffinityPropagation, wine_clustering.DAMPING_OPTIONS, True)
    best_rand, best_jaccard, best_fowlkes, i_rand, i_jacc, i_fowl = get_best_results(affinity_results, True)


if __name__ == '__main__':
    main()
