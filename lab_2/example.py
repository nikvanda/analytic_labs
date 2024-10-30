import math

import matplotlib.pyplot as plt
import pandas as pd

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


def find_a(x_values, y_values, n_value):
    return ((n_value * sum([x * y for x, y in zip(x_values, y_values)]) - sum(x_values) * sum(y_values)) /
            (n_value * sum(map(lambda x1: x1 ** 2, x_values)) - (sum(x_values) ** 2)))


def find_b(x_values, y_values, a_value, n_value):
    return (sum(y_values) / n_value) - a_value * (sum(x_values) / n_value)


def find_mistake(new_y, old_y):
    return (math.sqrt(sum([(y - y_1) ** 2 for y, y_1 in zip(old_y, new_y)]))) / len(new_y)


def find_linear():
    a_value = find_a(X, Y, n)
    print(a_value)
    create_plot(X, Y)
    b_value = find_b(X, Y, a_value, n)
    print(b_value)

    y_1 = [a_value * x + b_value for x in X]
    print(y_1)
    create_plot(X, y_1)

    x_2 = [1 / x for x in X[1:]]
    a_2 = find_a(x_2, Y[1:], n - 1)
    b_2 = find_b(x_2, Y[1:],a_2, n - 1)

    y_2 = [a_2 * x + b_2 for x in x_2]
    print(y_2)
    create_plot(X[1:], y_2)

    mistake_1 = find_mistake(y_1, Y)
    mistake_2 = find_mistake(y_2, Y[1:])
    print(mistake_1)
    print(mistake_2)
    plt.scatter(X, Y, color='black', marker='o')
    plt.plot(X, y_1, label="Линейная: y = 2x + 1", color="blue")
    plt.plot(X[1:], y_2, label="Гиперболическая: y = 1/(x+0.01)", color="red")
    plt.grid(True)
    plt.show()

    df = pd.DataFrame({'xi': X,
                       'yi': Y,
                       'y1': y_1,
                       'y2': [None, *y_2]})
    print(df)

def main():
    find_linear()


if __name__ == '__main__':
    main()
