import numpy as np
import pandas as pd
from sklearn import datasets, preprocessing
import seaborn as sns
import matplotlib.pyplot as plt


class Exercise:
    __iris = datasets.load_iris(as_frame=True).frame
    __s = 9
    __df_size = 5
    __n1 = 0
    __n2 = 2

    @property
    def iris(self):
        return self.__iris

    @property
    def s(self):
        return self.__s

    @property
    def n1(self):
        return self.__n1

    @property
    def n2(self):
        return self.__n2

    @property
    def df_size(self):
        return self.__df_size

    def get_subset(self) -> pd.DataFrame:
        """Ex 1"""
        processed_data = self.iris.iloc[self.s:self.s + self.df_size]
        print(processed_data)
        self.iris.to_csv('ex1.csv')
        return processed_data

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
        """Show subset properties depending on its type"""
        if isinstance(subset, pd.DataFrame):
            print(subset.describe())
        else:
            print(f'Mean: {subset.mean()}')
            print(f'Max: {subset.max()}')
            print(f'Min: {subset.min()}')

    @staticmethod
    def write_to_csv(subset: pd.DataFrame | np.ndarray, filename: str) -> None:
        if isinstance(subset, np.ndarray):
            subset = pd.DataFrame(subset)
        subset.to_csv(filename)


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


if __name__ == '__main__':
    main()
