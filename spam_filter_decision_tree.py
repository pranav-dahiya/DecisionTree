import numpy as np
import pickle
import glob
from multiprocessing import Pool
import matplotlib.pyplot as plt
import gc


def read_file(filename):
    with open(filename, "rb") as f:
        return [pickle.load(f), int("spm" in filename)]


def read_data(folder):
    with open("vocabulary3.pickle", "rb") as f:
        data = [np.zeros((len(folder), len(pickle.load(f)))), np.zeros(len(folder))]
    pool = Pool()
    temp = [None for i in range(len(folder))]
    for i, filename in enumerate(folder):
        temp[i] = pool.apply_async(read_file, (filename, ))
    pool.close()
    pool.join()
    for i, val in enumerate(temp):
        data[0][i,:] = val.get()[0]
        data[1][i] = val.get()[1]
    return data


class Node:
    def __init__(self, label=-1, feature=-1, feature_splits=None, children=None):
        self.label = label
        self.feature = feature
        self.feature_splits = feature_splits
        self.children = children


def minority_class(p):
    return min(p, 1-p)


def gini_index(p):
    return 2*p*(1-p)


def entropy(p):
    return -(p*np.log2(p) + (1-p)*np.log2(1-p))


def impurity(splits, y, imp_func):
    I = 0
    n = 0
    for split in splits:
        if len(split) > 0:
            p = np.sum(y.take(split))/len(split)
        else:
            p = 0.5
        I += len(split)*imp_func(p)
        n += len(split)
    I /= n
    return I


def split_set(X, split, split_values):
    splits = [[] for i in range(len(split_values)+1)]
    j = 0
    for i in split:
        flag = True
        for j, split_value in enumerate(split_values):
            if X[i] < split_value:
                splits[j].append(i)
                flag = False
                break
        if flag:
            splits[-1].append(i)
    return splits


def best_split_feature(X, y, split, imp_func, n_bins):
    I_min = float('inf')
    f_best = None
    f_splits = None
    for f in range(X.shape[1]):
        max = np.max(X[:,f])
        min = np.min(X[:,f])
        split_values = [i*(max-min)/n_bins for i in range(1,n_bins)]
        I = impurity(split_set(X[:,f], split, split_values), y, imp_func)
        if I < I_min:
            I_min = I
            f_best = f
            f_splits = split_values
    return f_best, f_splits


def grow_tree(X, y, split, imp_func, n_bins):
    if len(split) < 0.83 * y.size:
        if len(split) == 0:
            return Node(label=int(np.random.rand()>0.83))
        return Node(label=int((np.sum(y.take(split))/len(split))+0.5))
    feature, feature_splits = best_split_feature(X, y, split, imp_func, n_bins)
    #print(len(split), feature)
    splits = split_set(X[:,feature], split, feature_splits)
    children = list()
    for split in splits:
        children.append(grow_tree(X, y, split, imp_func, n_bins))
    return Node(feature=feature, feature_splits=feature_splits, children=children)


def classify(root, X, y_, split):
    if root.label >= 0:
        for i in split:
            y_[i] = root.label
    else:
        splits = split_set(X[:,root.feature], split, root.feature_splits)
        for i, split in enumerate(splits):
            if len(split) > 0:
                classify(root.children[i], X, y_, split)


def prune(root, X, y, split):
    if root.feature > 0:
        return
    predictions = np.zeros(y.shape)
    classify(root, X, predictions, split)
    y_ = y.take(split)
    predictions = predictions.take(split)
    label = int((np.sum(y_/len(split))+0.5))
    if np.sum(np.abs((np.ones(y_.shape)*label)-y_)) <= np.sum(np.abs(predictions-y_)):
        root.label = label
        root.children = None
    else:
        splits = split_set(X[:,root.feature], split, root.feature_splits)
        for split, child in zip(splits, root.children):
            prune(child, X, y, split)


def test(root, X, y):
    predictions = np.zeros(y.shape)
    classify(root, X, predictions, list(range(y.size)))
    TP, FP, TN, FN = 0, 0, 0, 0
    for prediction, label in zip(predictions, y):
        if prediction == 1:
            if label == 1:
                TP += 1
            else:
                FP += 1
        else:
            if label == 1:
                FN += 1
            else:
                TN += 1
    print(",", TP, ",", FP, ",", TN, ",", FN)
    return (TP+TN)/(TP+TN+FP+FN)


def train_test_split(X, y, i):
    X_train, y_train = [], []
    for j in range(len(y)):
        if j != i:
            X_train.extend(X[j])
            y_train.extend(y[j])
    return np.array(X_train), np.array(y_train), np.array(X[i]), np.array(y[i])


def train_prune_split(X, y):
    splits = np.array(range(y.size))
    np.random.shuffle(splits)
    train_split = np.sort(splits[:-int(y.size/10)])
    prune_split = np.sort(splits[-int(y.size/10):])
    return train_split.tolist(), prune_split.tolist()


def cross_validate(X, y, i, imp_func):
    X_train, y_train, X_test, y_test = train_test_split(X, y, i)
    train_split, prune_split = train_prune_split(X_train, y_train)
    tree = grow_tree(X_train, y_train, train_split, imp_func, 4)
    prune(tree, X_train, y_train, prune_split)
    return test(tree, X_test, y_test)


if __name__ == '__main__':
    X, y = ([None for i in range(10)] for j in range(2))
    for i in range(10):
        X[i], y[i] = read_data(glob.glob("lingspam/part"+str(i+1)+"_tfidf/*.pickle"))
    pool = Pool(1)
    results = []
    for i in range(10):
        results.append(pool.apply_async(cross_validate, args=(X, y, i, gini_index)))
    pool.close()
    for result in results:
        result.get()
    '''
    bins = [2, 3, 4, 5, 6, 7]
    pool = Pool()
    trees = list()
    for n_bins in bins:
        trees.append(pool.apply_async(grow_tree, args=(X_train, y_train, list(range(y_train.size)), gini_index, n_bins)))
    for i, tree in enumerate(trees):
        trees[i] = tree.get()
    train_accuracy = [test(tree, X_train, y_train) for tree in trees]
    X_prune, y_prune = read_data(glob.glob("lingspam/part9_tfidf/*.pickle"))
    temp = list()
    for tree in trees:
        temp.append(pool.apply_async(prune, args=(tree, X_prune, y_prune, list(range(y_prune.size)))))
    for val in temp:
        val.get()
    pool.close()
    prune_accuracy = [test(tree, X_prune, y_prune) for tree in trees]
    X_test, y_test = read_data(glob.glob("lingspam/part10_tfidf/*.pickle"))
    test_accuracy = [test(tree, X_test, y_test) for tree in trees]
    plt.plot(bins, train_accuracy, label='train_accuracy')
    plt.plot(bins, prune_accuracy, label='prune_accuracy')
    plt.plot(bins, test_accuracy, label='test_accuracy')
    plt.ylabel("accuracy")
    plt.xlabel("n_bins")
    plt.legend()
    plt.show()
    print(gc.collect())
    root = grow_tree(X_train, y_train, list(range(y_train.size)), gini_index)
    print(test(root, X_train, y_train))
    prune(root, X_prune, y_prune, list(range(y_prune.size)))
    print(gc.collect())
    print(test(root, X_train, y_train))
    print(test(root, X_test, y_test))
    '''
