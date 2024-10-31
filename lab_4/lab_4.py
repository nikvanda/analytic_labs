import pandas as pd
from sklearn import datasets, preprocessing, metrics
from sklearn.cluster import AgglomerativeClustering, KMeans, MeanShift, DBSCAN, AffinityPropagation


class Exercise:
    METHODS = ['single', 'complete', 'average', 'ward']
    N_CLUSTERS_RANGE = range(1, 11)
    BANDWIDTH_RANGE = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05]
    EPS_RANGE = [0.005, 0.006, 0.007, 0.008, 0.009, 0.01]
    MIN_SAMPLES_RANGE = range(1, 11)
    DAMPING_RANGE = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    def __init__(self, s1: int, s2: int, s3: int):
        df = datasets.load_wine(as_frame=True)['frame']
        self.s1, self.s2, self.s3 = s1, s2, s3

        wine_0 = df[df.target == 0].iloc[self.s1:self.s1 + 40]
        wine_1 = df[df.target == 1].iloc[self.s2:self.s2 + 40]
        wine_2 = df[df.target == 2].iloc[self.s3:self.s3 + 40]

        wine = pd.concat([wine_0, wine_1, wine_2])
        self.wine, self.target = wine.drop('target', axis=1).values, wine['target'].values
        self.normalized_wine = preprocessing.Normalizer().fit_transform(self.wine)

    def evaluate_cluster(self, model_fn, param_range, include_n_clusters=False):
        results = {'adjusted_rand': [],
                   'jaccard': [],
                   'fowlkes_mallows': [],
                   'n_clusters': []}
        for param in param_range:
            model = model_fn(param)
            adjusted_rand, jaccard, fowlkes_mallows, n_clusters = self.clusterize(model, include_n_clusters)
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

    def clusterize(self, model, include_n_clusters=False):
        y_pred = model.fit_predict(self.normalized_wine)
        adjusted_rand = metrics.adjusted_rand_score(self.target, y_pred)
        jaccard = metrics.jaccard_score(self.target, y_pred, average='weighted', zero_division=0)
        fowlkes_mallows = metrics.fowlkes_mallows_score(self.target, y_pred)

        n_clusters = len(set(y_pred)) - (1 if -1 in y_pred else 0) if include_n_clusters else None
        return adjusted_rand, jaccard, fowlkes_mallows, n_clusters


def add_n_clusters_if_present(results, key, is_dbscan=False):
    if results['n_clusters']:
        if is_dbscan:
            return [(tup[0], tup[1], round(float(tup[2]), 3), results['n_clusters'][i])
                    for i, tup in enumerate(results[key])]
        else:
            return [(tup[0], round(float(tup[1]), 3), results['n_clusters'][i])
                    for i, tup in enumerate(results[key])]
    else:
        return [(tup[0], round(float(tup[1]), 3)) if len(tup) == 2
                else (tup[0], tup[1], round(float(tup[2]), 3))
                for tup in results[key]]


def table_1(results, filename, is_dbscan=False):
    results['adjusted_rand'] = add_n_clusters_if_present(results, 'adjusted_rand', is_dbscan)
    results['jaccard'] = add_n_clusters_if_present(results, 'jaccard', is_dbscan)
    results['fowlkes_mallows'] = add_n_clusters_if_present(results, 'fowlkes_mallows', is_dbscan)

    df = pd.DataFrame({
        'Adjusted Rand': results['adjusted_rand'],
        'Jaccard': results['jaccard'],
        'Fowlkes Mallows': results['fowlkes_mallows']
    })
    df.to_excel(filename, index=False)


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


def main():
    s1, s2, s3 = 9, 2, 1
    ex = Exercise(s1, s2, s3)
    agglomerative_params = [(n_clusters, method) for method in ex.METHODS for n_clusters in ex.N_CLUSTERS_RANGE]
    agglomerative_results = ex.evaluate_cluster(AgglomerativeClustering, agglomerative_params)
    best_rand, best_jaccard, best_fowlkes, *_ = get_best_values(agglomerative_results)

    kmeans_results = ex.evaluate_cluster(KMeans, ex.N_CLUSTERS_RANGE)
    best_rand, best_jaccard, best_fowlkes, *_ = get_best_values(kmeans_results)

    meanshift_results = ex.evaluate_cluster(MeanShift, ex.BANDWIDTH_RANGE, True)
    best_rand, best_jaccard, best_fowlkes, i_rand, i_jacc, i_fowl = get_best_values(meanshift_results, True)

    dbscan_params = [(eps, min_samples) for eps in ex.EPS_RANGE for min_samples in ex.MIN_SAMPLES_RANGE]
    dbscan_results = ex.evaluate_cluster(DBSCAN, dbscan_params, True)
    best_rand, best_jaccard, best_fowlkes, i_rand, i_jacc, i_fowl = get_best_values(dbscan_results, True)

    affinity_results = ex.evaluate_cluster(AffinityPropagation, ex.DAMPING_RANGE, True)
    best_rand, best_jaccard, best_fowlkes, i_rand, i_jacc, i_fowl = get_best_values(affinity_results, True)


if __name__ == '__main__':
    main()
