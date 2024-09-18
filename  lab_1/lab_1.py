import numpy as np
import pandas as pd
from sklearn import datasets, preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

rng = np.random.default_rng(12345)

N1 = 0
N1_COUNT = 2
N2 = 2
N2_COUNT = 3


class Exercise:
    __iris = datasets.load_iris(as_frame=True).frame
    __s = 9
    __df_size = 5

    @property
    def iris(self):
        return self.__iris

    @property
    def s(self):
        return self.__s

    @property
    def df_size(self):
        return self.__df_size

    def get_subset(self) -> pd.DataFrame:
        """Ex 1"""
        processed_data = self.iris.iloc[self.s:self.s + self.df_size]
        print(processed_data)
        self.iris.to_csv('ex1.csv')
        return processed_data

    @staticmethod
    def fill_random_nan(subset: pd.DataFrame, column: int, random_count: int) -> pd.DataFrame:
        """
        Set random values to nan.
        """
        idxes = rng.integers(0, subset.shape[0], random_count)
        print(idxes)
        for i in range(random_count):
            subset.iloc[i, column] = np.nan

        return subset

    def display_subset(self, subset: pd.DataFrame = iris) -> None:
        """Ex 2"""
        sns.scatterplot(data=subset, x='petal length (cm)', y='petal width (cm)', hue='target', palette='Greys')
        sns.pairplot(self.iris, hue='target')
        plt.show()

        confusion_matrix = subset.drop('target', axis=1).corr()
        sns.heatmap(confusion_matrix)
        plt.show()

    @staticmethod
    def standardize(subset: pd.DataFrame = iris) -> pd.DataFrame:
        """Ex 3"""
        standardized_subset = preprocessing.StandardScaler().fit_transform(subset)
        return standardized_subset

    @staticmethod
    def normalize(subset: pd.DataFrame = iris) -> pd.DataFrame:
        """Ex 3"""
        normalized_subset = preprocessing.Normalizer().fit_transform(subset)
        return normalized_subset

    @staticmethod
    def describe_subset(subset: pd.DataFrame | np.ndarray) -> None:
        """
        Show subset properties depending on its type
        """
        if isinstance(subset, pd.DataFrame):
            print(subset.describe())
        else:
            print(f'Mean: {subset.mean()}')
            print(f'Max: {subset.max()}')
            print(f'Min: {subset.min()}')

    @staticmethod
    def write_to_csv(subset: pd.DataFrame | np.ndarray, filename: str) -> None:
        """
        Write DataFrame to csv. If an argument is an array it converts it to DataFrame before writing
        """
        if isinstance(subset, np.ndarray):
            subset = pd.DataFrame(subset)
        subset.to_csv(filename)

    @staticmethod
    def fill_mean_nan(subset: pd.DataFrame) -> pd.DataFrame:
        """Ex 3"""
        mean_by_columns = {key: np.mean(column[[*map(lambda x: not x, np.isnan(column))]]) for key, column in zip(subset.keys(), subset.T.values)}
        return subset.fillna(value=mean_by_columns)

    @staticmethod
    def fill_imputer(subset: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
        return SimpleImputer(missing_values=np.nan, strategy=strategy).fit_transform(subset)


def main():
    ex = Exercise()
    subset = ex.get_subset()
    ex.describe_subset(subset)
    ex.display_subset(subset)

    standardized = ex.standardize(subset)
    ex.describe_subset(standardized)
    ex.write_to_csv(standardized, 'standardized.csv')

    normalized = ex.normalize(subset)
    ex.describe_subset(normalized)
    ex.write_to_csv(normalized, 'normalized.csv')

    print(subset)
    nan_subset = ex.fill_random_nan(subset, N1, N1_COUNT)
    nan_subset = ex.fill_random_nan(nan_subset, N2, N2_COUNT)

    ex.write_to_csv(nan_subset, 'nan_subset.csv')
    print(nan_subset)

    dropped_nan_subset = nan_subset.dropna()
    print(dropped_nan_subset)

    ex.write_to_csv(dropped_nan_subset, 'dropped_nan.csv')

    filled_mean_subset = ex.fill_mean_nan(nan_subset)
    ex.describe_subset(filled_mean_subset)
    ex.write_to_csv(filled_mean_subset, 'filled_mean_subset.csv')

    imputed_subset = ex.fill_imputer(subset)
    ex.describe_subset(imputed_subset)
    ex.write_to_csv(imputed_subset, 'imputer_mean.csv')


if __name__ == '__main__':
    main()
