from collections import namedtuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


class Visualizer:

    @staticmethod
    def visualisation_task_2(matrix, title, x_labels, y_labels):
        sns.heatmap(matrix, annot=True, fmt=".4f", cmap="YlGnBu", xticklabels=x_labels,
                    yticklabels=y_labels)
        plt.title(title)
        plt.xlabel('max_features')
        plt.ylabel('max_depth')
        plt.show()

    @staticmethod
    def plot_graph(scores, label, title, n_range):
        plt.figure(figsize=(8, 6))
        plt.plot(n_range, scores, label=label, marker='o')
        plt.legend()
        plt.grid(True)
        plt.title(title)
        plt.xlabel('n_estimators')
        plt.ylabel('Score')
        plt.show()


TASK_RESPONSE = namedtuple('TASK_RESPONSE', ('accuracy', 'precision', 'recall'))


class Exercise:
    MAX_DEPTH_RANGE = range(2, 10)
    MAX_FEATURES_RANGE = range(2, 10)
    N_ESTIMATORS_RANGE = range(2, 101)

    def __init__(self, cv: int, rnd_st: int, c_param: str, p_param: str, s1: int, s2: int, s3: int):
        df = datasets.load_wine(as_frame=True)['frame']
        self.s1, self.s2, self.s3 = s1, s2, s3

        wine_0 = df[df.target == 0].iloc[self.s1:self.s1 + 40]
        wine_1 = df[df.target == 1].iloc[self.s2:self.s2 + 40]
        wine_2 = df[df.target == 2].iloc[self.s3:self.s3 + 40]

        wine = pd.concat([wine_0, wine_1, wine_2])
        self.wine, self.target = wine.drop('target', axis=1).values, wine['target'].values

        self.cv, self.rnd = cv, rnd_st
        self.c, self.p = c_param, p_param

    def task_2(self):
        results, indexes = [], []
        accuracy_mtx = np.zeros((len(self.MAX_DEPTH_RANGE),
                                 len(self.MAX_FEATURES_RANGE)))
        precision_mtx = np.zeros((len(self.MAX_DEPTH_RANGE),
                                  len(self.MAX_FEATURES_RANGE)))
        recall_mtx = np.zeros((len(self.MAX_DEPTH_RANGE),
                               len(self.MAX_FEATURES_RANGE)))
        for i, max_depth in enumerate(self.MAX_DEPTH_RANGE):
            for j, max_features in enumerate(self.MAX_FEATURES_RANGE):
                clf = DecisionTreeClassifier(max_depth=max_depth,
                                             max_features=max_features,
                                             random_state=self.rnd,
                                             criterion=self.c)
                accuracy = cross_val_score(clf, self.wine, self.target,
                                           cv=self.cv,
                                           scoring='accuracy').mean()
                precision = cross_val_score(clf, self.wine, self.target,
                                            cv=self.cv,
                                            scoring='precision_micro').mean()
                recall = cross_val_score(clf, self.wine, self.target,
                                         cv=self.cv,
                                         scoring='recall_micro').mean()
                accuracy_mtx[i, j] = accuracy
                precision_mtx[i, j] = precision
                recall_mtx[i, j] = recall
                indexes.append([max_depth, max_features])
                results.append(accuracy)

        Visualizer.visualisation_task_2(accuracy_mtx, 'Accuracy', self.MAX_FEATURES_RANGE, self.MAX_DEPTH_RANGE)
        Visualizer.visualisation_task_2(precision_mtx, 'Precision', self.MAX_FEATURES_RANGE, self.MAX_DEPTH_RANGE)
        Visualizer.visualisation_task_2(recall_mtx, 'Recall', self.MAX_FEATURES_RANGE, self.MAX_DEPTH_RANGE)

    def task_3(self):
        param_grid = {
            'max_depth': self.MAX_DEPTH_RANGE,
            'criterion': [self.c],
            'max_features': self.MAX_FEATURES_RANGE
        }
        clf = DecisionTreeClassifier(random_state=self.rnd, criterion=self.c)
        grid_search = GridSearchCV(clf, param_grid, cv=self.cv, scoring='accuracy').fit(self.wine, self.target)

    def task_4(self, classifier: AdaBoostClassifier | RandomForestClassifier):
        accuracy_scores, precision_scores, recall_scores = [], [], []
        classifier.fit(self.wine, self.target)
        for n_estimator in self.N_ESTIMATORS_RANGE:
            classifier.set_params(n_estimators=n_estimator)
            accuracy = cross_val_score(classifier, self.wine, self.target, cv=self.cv,
                                       scoring='accuracy').mean()
            precision = cross_val_score(classifier, self.wine, self.target, cv=self.cv,
                                        scoring='precision_macro').mean()
            recall = cross_val_score(classifier, self.wine, self.target, cv=self.cv,
                                     scoring='recall_macro').mean()
            accuracy_scores.append(accuracy)
            precision_scores.append(precision)
            recall_scores.append(recall)

        return TASK_RESPONSE(accuracy_scores, precision_scores, recall_scores)


def main():
    s1, s2, s3 = 9, 2, 1
    rnd, cv = 1, 5
    c, p = 'entropy', 'max_features'
    ex = Exercise(cv, rnd, c, p, s1, s2, s3)

    ex.task_2()
    ex.task_3()

    ada_boost = AdaBoostClassifier(random_state=ex.rnd, algorithm='SAMME')
    ada_accuracy, ada_precision, ada_recall = ex.task_4(ada_boost)

    title = 'AdaBoostClassifier'
    Visualizer.plot_graph(ada_accuracy, 'Accuracy', title, ex.N_ESTIMATORS_RANGE)
    Visualizer.plot_graph(ada_precision, 'Precision', title, ex.N_ESTIMATORS_RANGE)
    Visualizer.plot_graph(ada_recall, 'Recall', title, ex.N_ESTIMATORS_RANGE)

    forest_accuracy, forest_precision, forest_recall = ex.task_4(
        RandomForestClassifier(random_state=ex.rnd, criterion=ex.c))

    title = 'RandomForestClassifier'
    Visualizer.plot_graph(forest_accuracy, 'Accuracy', title, ex.N_ESTIMATORS_RANGE)
    Visualizer.plot_graph(forest_precision, 'Precision', title, ex.N_ESTIMATORS_RANGE)
    Visualizer.plot_graph(forest_recall, 'Recall', title, ex.N_ESTIMATORS_RANGE)


if __name__ == '__main__':
    main()
