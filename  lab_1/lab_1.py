from sklearn import datasets, preprocessing


class Exercise:
    iris = datasets.load_iris(as_frame=True).frame

    def ex1(self):
        processed_data = self.iris.iloc[10:15]
        print(processed_data)
        print(processed_data.describe())
        self.iris.to_csv('ex1.csv')

    def ex2(self):
        pass


def main():
    ex = Exercise()
    ex.ex1()


if __name__ == '__main__':
    main()
