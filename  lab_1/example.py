import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets, preprocessing
from sklearn.impute import SimpleImputer

iris = datasets.load_iris()

# print(iris)

# print(iris.keys())

# print(iris['feature_names'])
# print(iris['data'])
#
# print(iris['target_names'])
#
# print(iris['target'])

x_data = iris.data
y_data = iris.target

# print(x_data)
# print(y_data)

setosa, versicolor, virginica = x_data[y_data == 0], x_data[y_data == 1], x_data[y_data == 2]

# print(setosa)
# print(versicolor)
# print(virginica)

iris_df = datasets.load_iris(as_frame=True).frame

print(iris_df.head())
print(iris_df.tail())

print(iris_df.index)
print(iris_df.columns)

print(iris_df.dtypes)

print(type(iris_df.to_numpy()))

print(iris_df.T)
print(iris_df)


print(iris_df.loc[:, 'target'])
print(iris_df.iloc[10:20])
print(iris_df.iloc[10:20, 1:3])

print(iris_df.describe())

print(iris_df.mean(axis=1))

print(iris_df.value_counts())

iris_df.to_csv('iris.csv')

iris_csv = pd.read_csv('iris.csv')

print(iris_csv)

iris_csv.to_excel('iris.xlsx')

iris_xlsx = pd.read_excel('iris.xlsx', 'Sheet1', index_col=None, na_values=['NA'])

print('EXCEL')
print(iris_xlsx)

sns.scatterplot(data=iris_df, x='petal length (cm)', y='petal width (cm)', hue='target', palette='Greys')

# sns.pairplot(iris_df, hue='target')

confusion_matrix = iris_df.drop('target', axis=1).corr()
sns.heatmap(confusion_matrix)

# plt.show()
print('STANDARD')
scaler = preprocessing.StandardScaler().fit(x_data)
print(scaler)
print(scaler.mean_)
print(scaler.var_)
print(scaler.scale_)

x_scaled = scaler.transform(x_data)
print('SCALED')
print(x_scaled)
print(np.mean(x_scaled))
print(np.std(x_scaled))

x_scaled_combo = preprocessing.StandardScaler().fit_transform(x_data)

x_normalized = preprocessing.normalize(x_data, norm='l2')
print('NORMALIZED')
print(x_normalized)

x_normalized_combo = preprocessing.Normalizer().fit_transform(x_data)

print(iris_df.size)
iris_df = iris_df.dropna()
print(iris_df.size)

iris_df = iris_df.fillna(value=5)

print('IMPUTER')
mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
x_data_corrupted = [[7, 2, 3, 1], [4, np.nan, 6, 1], [10, 5, 9, 1]]
mean_imputer.fit(x_data)
x_imputed = mean_imputer.transform(x_data_corrupted)
print(x_imputed)
print(x_data_corrupted)

