import math

import matplotlib.pyplot as plt
import numpy as np

# Пример данных
X = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Y = [100, 75, 55, 40, 30, 20, 15, 10, 10, 5, 5]
n = 11

def create_plot(x, y):
    # Создание точечного графика
    plt.scatter(x, y, color='blue', marker='o')

    # Добавление заголовка и меток осей
    plt.title('Точечный график')
    plt.xlabel('Ось X')
    plt.ylabel('Ось Y')

    # Показ графика
    plt.grid(True)
    plt.show()


def find_linear(x_input, y):
    A = ((n * sum([x * y for x, y in zip(X, Y)]) - sum(X) * sum(Y)) /
         (n * sum(map(lambda x1: x1 ** 2, X)) - (sum(X) ** 2)))
    print(A)
    create_plot(X, Y)
    B = (sum(Y) / n) - A * (sum(X) / n)
    print(B)

    Y_1 = [A * x + B for x in X]
    print(Y_1)
    create_plot(X, Y_1)

    b_1 = math.e ** B
    print(b_1)
    y_2 = [b_1 * (math.e ** (A * x)) for x in X]
    print(y_2)
    create_plot(X, y_2)


def main():
    find_linear(X, Y)


if __name__ == '__main__':
    main()
