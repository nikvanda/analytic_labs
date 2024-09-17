from sklearn import datasets, preprocessing


def ex1(dataset):
    processed_data = dataset.iloc[10:15]
    print(processed_data)
    print(processed_data.describe())
    dataset.to_csv('ex1.csv')


def main():
    iris = datasets.load_iris(as_frame=True).frame
    ex1(iris)


if __name__ == '__main__':
    main()
