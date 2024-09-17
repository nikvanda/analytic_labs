from sklearn import datasets, preprocessing
import seaborn as sns
import matplotlib.pyplot as plt


class Exercise:
    __iris = datasets.load_iris(as_frame=True).frame

    @property
    def iris(self):
        return self.__iris

    def ex1(self):
        processed_data = self.iris.iloc[10:15]
        print(processed_data)
        print(processed_data.describe())
        self.iris.to_csv('ex1.csv')

    def ex2(self):
        sns.scatterplot(data=self.iris, x='petal length (cm)', y='petal width (cm)', hue='target', palette='Greys')
        sns.pairplot(self.iris, hue='target')
        plt.show()

        confusion_matrix = self.iris.drop('target', axis=1).corr()
        sns.heatmap(confusion_matrix)
        plt.show()


def main():
    ex = Exercise()
    # ex.ex1()
    ex.ex2()


if __name__ == '__main__':
    main()
