import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn import datasets, neighbors


def main():
    k = 40
    iris = datasets.load_iris()
    # print(iris)
    print(type(iris))
    print(iris.keys())
    print(iris['data'])
    print(iris['target_names'])
    print(iris['target'])
    X = iris.data[:, 1:3]
    y = iris.target
    print(X)
    print(y)
    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)

    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    cmap = ['darkorange', 'c', 'darkblue']

    knn = neighbors.KNeighborsClassifier(k)
    knn.fit(x_train, y_train)
    _, ax = plt.subplots()

    sns.scatterplot(
        x=X[:, 0],
        y=X[:, 1],
        hue=iris.target_names[y],
        palette=cmap,
        alpha=1.0,
        edgecolor='black',
    )
    plt.title(f'3-Class classification (k = {k})')
    plt.show()

    score = knn.score(x_test, y_test)
    print(score)


if __name__ == '__main__':
    main()
