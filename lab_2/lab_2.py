import pandas as pd
import sklearn.datasets as skl
import matplotlib.pyplot as plt
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split

from collections import namedtuple

from sklearn.neighbors import KNeighborsClassifier

K1 = 2
K2 = 12
T = 0.4
ORIGINAL = 'Original'
STANDARDIZED = 'Standardized'
NORMALIZED = 'Normalized'

TrainSubsets = namedtuple('TrainSubsets', ['x_train', 'x_test', 'y_train', 'y_test'])


def get_subsets(subset: pd.DataFrame) -> TrainSubsets:
    x = subset.iloc[:, :3]
    y = subset.target
    subsets_to_train = TrainSubsets(*train_test_split(x, y, train_size=T, random_state=0))
    return subsets_to_train


def train(subset: TrainSubsets) -> list[KNeighborsClassifier]:
    return [neighbors.KNeighborsClassifier(k).fit(subset.x_train, subset.y_train)
            for k in range(K1, K2 + 1)]


def get_score_lists(subset: TrainSubsets, knns: list[KNeighborsClassifier]):
    trained_subset = [knn.score(subset.x_train, subset.y_train) for knn in knns]
    test_subset = [knn.score(subset.x_test, subset.y_test) for knn in knns]
    return trained_subset, test_subset


def display_scores(train_subset, test_subset, title):
    plt.plot(range(K1, K2 + 1), train_subset, label=f'Train {title}', marker='o')
    plt.plot(range(K1, K2 + 1), test_subset, label=f'Test {title}', marker='o')
    plt.legend()
    plt.grid(True)
    plt.xlabel('k')
    plt.ylabel('Quality')
    plt.title('KNN')
    plt.show()


def write_to_csv(name: str, **kwargs):
    df = pd.DataFrame(kwargs)
    df.set_index('knn', inplace=True)
    df.to_csv(f'{name}.csv')


def main():
    wine_ds = skl.load_wine(as_frame=True).frame
    keys = wine_ds.keys()
    standardized = pd.DataFrame(preprocessing.StandardScaler().fit_transform(wine_ds), columns=keys)
    normalized = pd.DataFrame(preprocessing.Normalizer().fit_transform(wine_ds), columns=keys)

    standardized.target = wine_ds.target
    normalized.target = wine_ds.target

    wine_ds = get_subsets(wine_ds)
    standardized = get_subsets(standardized)
    normalized = get_subsets(normalized)

    knn_wine_ds = train(wine_ds)
    knn_standardized = train(standardized)
    knn_normalized = train(normalized)

    wine_ds_train, wine_ds_test = get_score_lists(wine_ds, knn_wine_ds)
    write_to_csv(name=ORIGINAL, wine_ds_train=wine_ds_train, wine_ds_test=wine_ds_test, knn=range(K1, K2+1))

    standardized_train, standardized_test = get_score_lists(standardized, knn_standardized)
    write_to_csv(name=STANDARDIZED, wine_ds_train=standardized_train, wine_ds_test=standardized_test, knn=range(K1, K2+1))

    normalized_train, normalized_test = get_score_lists(normalized, knn_normalized)
    write_to_csv(name=NORMALIZED, wine_ds_train=standardized_train, wine_ds_test=standardized_test, knn=range(K1, K2+1))

    display_scores(wine_ds_train, wine_ds_test, ORIGINAL)
    display_scores(standardized_train, standardized_test, STANDARDIZED)
    display_scores(normalized_train, normalized_test, NORMALIZED)


if __name__ == '__main__':
    main()
