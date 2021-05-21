# sources used
# https://stackoverflow.com/questions/52649568/use-of-np-where0
# https://stackoverflow.com/questions/20242479/printing-a-tree-data-structure-in-python
from collections import Counter

import numpy as np
from numpy import genfromtxt
import scipy.io
from scipy import stats

import random


def results_to_csv(y_test):
    y_test = y_test.astype(int)
    df = pd.DataFrame({'Category': y_test})
    df.index += 1  # Ensures that the index starts at 1.
    df.to_csv('submission.csv', index_label='Id')


random.seed(111)
np.random.seed(111)
epsilon = 1 / 1000000000
f = [
    "pain", "private", "bank", "money", "drug", "spam", "prescription",
    "creative", "height", "featured", "differ", "width", "other",
    "energy", "business", "message", "volumes", "revision", "path",
    "meter", "memo", "planning", "pleased", "record", "out",
    "semicolon", "dollar", "sharp", "exclamation", "parenthesis",
    "square_bracket", "ampersand", "http", "www", "free", "winner", "viagra"
]


class DecisionTree:

    def __init__(self, labels=None, depth=5, features=None, isForest=False):
        self.depth = depth
        self.labels = labels
        self.is_random_forest = isForest
        self.left = None
        self.right = None
        self.index = None
        self.thresh = None
        self.data = None
        self.pred = None
        self.features = features

    @staticmethod
    def entropy(y):
        if len(y) <= 1:
            return 0
        if np.unique(y).size == 1:
            return 0
        # counts = np.bincount(y)
        # probs = counts[np.where(counts>0)] / y.size
        num = 0
        for x in y:
            if (x == 0):
                num = num + 1
        p0 = num / y.size

        # p = probs[0]
        # if np.abs(p) < epsilon or np.abs(1 - p) < epsilon:
        #     return 0

        # return - np.sum(probs * np.log(probs)) / np.log(len(probs))
        return -p0 * np.log(p0) - (1 - p0) * np.log(1 - p0)

    @staticmethod
    def information_gain(X, y, thresh):
        parent = DecisionTree.entropy(y)
        y1 = y[np.where(X < thresh)[0]]
        p1 = y1.size / y.size
        y2 = y[np.where(X >= thresh)[0]]
        p2 = y2.size / y.size
        return parent - p1 * DecisionTree.entropy(y1) - p2 * DecisionTree.entropy(y2)

    @staticmethod
    def gini_impurity(y):
        if len(y) == 0:
            return 0
        counts = np.bincount(y)
        probs = counts[np.where(counts > 0)] / y.size
        p = probs[0]
        if np.unique(y).size == 1:
            return 0
        if np.abs(p) < epsilon or np.abs(1 - p) < epsilon:
            return 0
        return 1.0 - pow(p, 2) - pow(1 - p, 2)

    @staticmethod
    def gini_purification(X, y, thresh):
        parent = DecisionTree.gini_impurity(y)
        child1 = y[np.where(X < thresh)[0]]
        p1 = child1.size / y.size
        child2 = y[np.where(X >= thresh)[0]]
        p2 = child2.size / y.size
        return parent - p1 * DecisionTree.gini_impurity(child1) + p2 * DecisionTree.gini_impurity(child2)

    def split(self, X, y, idx, thresh):
        idx0 = np.where(X[:, idx] < thresh)[0]
        idx1 = np.where(X[:, idx] >= thresh)[0]
        X0, X1 = X[idx0, :], X[idx1, :]
        y0, y1 = y[idx0], y[idx1]
        return X0, y0, X1, y1

    def segmenter(self, X, y):
        if self.is_random_forest:
            feature_index = np.random.choice(X.shape[1], self.features)
            feature_index.sort()
            self.is_random_forest = False
            index, threshold = self.segmenter(X[:, feature_index], y)
            return feature_index[index], threshold
        else:
            gains = []
            thresholds = np.array([
                np.linspace(np.nan_to_num(np.min(X[:, i]) + epsilon), np.nan_to_num(np.max(X[:, i]) - epsilon),
                            num=X.shape[1])
                for i in range(X.shape[1])
            ])

            for i in range(X.shape[1]):
                gains.append([
                    self.information_gain(X[:, i], y, t) for t in thresholds[i, :]
                ])
            gains = np.nan_to_num(np.array(gains))
            index, thresh_idx = np.unravel_index(
                np.argmax(gains), gains.shape)
            return index, thresholds[index, thresh_idx]

    def fit(self, X, y):
        if self.depth > 0:
            if self.is_random_forest:
                self.index, self.thresh = self.segmenter(X, y)
            else:
                self.index, self.thresh = self.segmenter(X, y)
            X0, y0, X1, y1 = self.split(
                X, y, idx=self.index, thresh=self.thresh)
            if X0.size > 0 and X1.size > 0:
                self.left = DecisionTree(
                    depth=self.depth - 1, labels=self.features)
                self.left.fit(X0, y0)
                self.right = DecisionTree(
                    depth=self.depth - 1, labels=self.features)
                self.right.fit(X1, y1)
            else:
                self.depth = 0
                self.data, self.labels = X, y
                self.pred = stats.mode(y).mode[0]
        else:
            self.data, self.labels = X, y
            self.pred = stats.mode(y).mode[0]

        return self

    def predict(self, X):
        if self.depth != 0:
            idx0 = np.where(X[:, self.index] < self.thresh)[0]
            idx1 = np.where(X[:, self.index] >= self.thresh)[0]
            X0, X1 = X[idx0, :], X[idx1, :]
            yhat = np.zeros(X.shape[0])
            yhat[idx0] = self.left.predict(X0)
            yhat[idx1] = self.right.predict(X1)
            return yhat
        else:
            return self.pred * np.ones(X.shape[0])

    def __repr__(self, depth=0):

        if self.pred != None:
            str = '\t' * depth + 'leaf {}\n'.format(self.pred)
        else:
            str = '\t' * depth + 'depth {}, split  {}, thresh  < {}\n'.format(depth, f[self.index], self.thresh)
        if self.left != None:
            str += self.left.__repr__(depth + 1)
        if self.right != None:
            str += self.right.__repr__(depth + 1)
        return str


class RandomForest():
    def __init__(self, depth=None, labels=None, features=None, n_trees=100):
        self.depth = depth
        self.n_trees = n_trees
        self.trees = list()
        for i in range(n_trees):
            self.trees.append(DecisionTree(labels=labels, depth=10, features=1, isForest=True))

    def fit(self, X, y):

        for tree in self.trees:
            nX = np.random.choice(X.shape[0], X.shape[0])
            tree.fit(X[nX, :], y[nX])
        return self

    def predict(self, X):
        prediction = list()
        for tree in self.trees:
            prediction.append(tree.predict(X))
        return np.mean(prediction, axis=0)  # Todo

    def __repr__(self):
        """
        one way to visualize the decision tree is to write out a __repr__ method
        that returns the string representation of a tree. Think about how to visualize
        a tree structure. You might have seen this before in CS61A.
        """
        str = ""
        for _ in self.trees:
            str += self.trees.__repr__()
            str += "\n"
        return str


if __name__ == "__main__":
    # dataset = "titanic"
    dataset = "spam"

    if dataset == "titanic":
        import pandas as pd

        titanic_data = pd.read_csv('datasets/titanic/titanic_training.csv')
        means = {'pclass': titanic_data['pclass'].mode(), 'sex': titanic_data['sex'].mode(),
                 'age': titanic_data['age'].mean(),
                 'sibsp': titanic_data['sibsp'].mode(), 'parch': titanic_data['parch'].mode(),
                 'fare': titanic_data['fare'].mean(),
                 'embarked': titanic_data['embarked'].mode()}
        titanic_training = titanic_data[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']]
        titanic_training = pd.get_dummies(titanic_training, columns=['sex', 'embarked'])
        titanic_training = titanic_training.values
        titanic_labels = titanic_data['survived']
        titanic_labels = titanic_labels.values
        titanic_test = pd.read_csv('datasets/titanic/titanic_testing_data.csv')
        titanic_test = titanic_test[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']]
        titanic_test = pd.get_dummies(titanic_test, ['sex', 'embarked'])
        f = titanic_test.keys()
        print(f)
        titanic_test = titanic_test.values
        num = titanic_training.shape[0] // 10
        titanic_validation = titanic_training[:num, :]
        titanic_validation_labels = titanic_labels[0:num]
        titanic_training = titanic_training[num:, :]
        titanic_labels = titanic_labels[num:]

        dt = DecisionTree(labels=None, depth=10, features=f, isForest=False)
        rf = RandomForest(labels=None, depth=10, features=f, n_trees=100)

        dt.fit(titanic_training, titanic_labels)
        rf.fit(titanic_training, titanic_labels)

        testing_acc_dt = np.sum(dt.predict(titanic_training) == titanic_labels) / len(titanic_labels)
        testing_acc_rf = np.sum(dt.predict(titanic_validation) == titanic_validation_labels) / len(
            titanic_validation_labels)
        print('testing accuracy for decision tree:', testing_acc_dt)
        print('testing accuracy for random forest:', testing_acc_rf)

        # results_to_csv(dt.predict(titanic_test))

        # Todo
        # import csv
        # from sklearn.feature_extraction import DictVectorizer
        # from sklearn import preprocessing
        # import numpy as np
        # from sklearn.preprocessing import Imputer, OneHotEncoder
        #
        # test = csv.DictReader(open('titanic_testing_data.csv'))
        # training = csv.DictReader(open("titanic_training.csv"))
        # enc = OneHotEncoder(handle_unknown='ignore')
        # X = enc.fit(training)
        # y =  enc.fit(test)
        # v = DictVectorizer()
        # X = v.fit_transform(X).toarray()
        # y = v.fit_transform(test).toarray()
        #
        # le = preprocessing.LabelEncoder()
        # OneHotEncoder().fit_transform(X)
        # OneHotEncoder().fit_transform(y)
        #
        # imp = Imputer(missing_values='NaN', strategy='mode', axis=0)
        # X = imp.fit_transform(X)
        # y = imp.fit_transform(y)

    elif dataset == "spam":
        features = [
            "pain", "private", "bank", "money", "drug", "spam", "prescription",
            "creative", "height", "featured", "differ", "width", "other",
            "energy", "business", "message", "volumes", "revision", "path",
            "meter", "memo", "planning", "pleased", "record", "out",
            "semicolon", "dollar", "sharp", "exclamation", "parenthesis",
            "square_bracket", "ampersand", "http", "www", "free", "winner", "viagra"
        ]
        assert len(features) == 37

        # Load spam data
        path_train = 'datasets/spam-dataset/spam_data.mat'
        data = scipy.io.loadmat(path_train)
        X = data['training_data']
        y = np.squeeze(data['training_labels'])
        Z = data['test_data']
        class_names = ["Ham", "Spam"]

        dt = DecisionTree(labels=None, depth=10, features=features, isForest=False)

        from sklearn.model_selection import train_test_split

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)
        print("Features", features)
        print("Train/test size", X.shape, Z.shape)

        # dt.fit(X_train, y_train)
        # dt_predictions = dt.predict(X_train)
        # dt_predictions_val = dt.predict(X_val)
        # dt_predictions_test = dt.predict(Z)
        #
        # print('The training accuracy for tree is', np.sum(dt_predictions == y_train) / len(y_train))
        # print('The validation accuracy for tree is', np.sum(dt_predictions_val == y_val) / len(y_val))

        accuracies = list()

        for i in range(41):
            dt2 = DecisionTree(labels=None, depth=i, features=features, isForest=False)
            dt2.fit(X_train, y_train)
            dt_predictions_val = dt.predict(X_val)
            accuracies.append(np.sum(dt_predictions_val == y_val) / len(y_val))
        import matplotlib.pyplot as plt

        plt.figure()
        plt.legend()
        plt.xlabel("depth")
        plt.ylabel("accuracy")
        plt.plot(range(41), accuracies, 'r')

        plt.show()

        # rf = RandomForest(labels=None, depth=10, features=features , n_trees=100)
        # rf.fit(X_train, y_train)
        # rf_predictions = rf.predict(X_train)
        # rf_predictions_val = rf.predict(X_val)
        # print('The training accuracy for forest is', np.sum(rf_predictions == y_train) / len(y_train))
        # print('The validation accuracy for forest is', np.sum(rf_predictions_val == y_val) / len(y_val))

        accuraciesRF = list()

        # for i in range(41):
        #     dt2 = RandomForest(labels=None, depth=i, features=features)
        #     dt2.fit(X_train, y_train)
        #     dt2_predictions = dt2.predict(X_train)
        #     accuraciesRF.append(np.sum(dt2_predictions == y_train) / len(y_train))
        #
        # import matplotlib.pyplot as plt
        # plt.figure()
        #
        # plt.plot(accuraciesRF, range(41), 'r')
        #
        # plt.show()

        # print("Tree structure\n", dt.__repr__())
    else:
        raise NotImplementedError("Dataset %s not handled" % dataset)

