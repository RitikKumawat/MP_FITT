import numpy as np

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def _entropy(self, y):
        counts = np.bincount(y)
        probabilities = counts / len(y)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))  # Adding epsilon to avoid log(0)
        return entropy

    def _information_gain(self, y, y_split):
        parent_entropy = self._entropy(y)
        child_entropy = sum((len(child_y) / len(y)) * self._entropy(child_y) for child_y in y_split)
        info_gain = parent_entropy - child_entropy
        return info_gain

    def _split(self, X, y, feature_idx, split_value):
        left_mask = X[:, feature_idx] <= split_value
        right_mask = ~left_mask
        X_left, X_right = X[left_mask], X[right_mask]
        y_left, y_right = y[left_mask], y[right_mask]
        return X_left, X_right, y_left, y_right

    def _find_best_split(self, X, y):
        best_feature_idx, best_split_value, max_info_gain = None, None, -np.inf
        for feature_idx in range(X.shape[1]):
            unique_values = np.unique(X[:, feature_idx])
            for split_value in unique_values:
                y_split = [y[X[:, feature_idx] <= split_value], y[X[:, feature_idx] > split_value]]
                info_gain = self._information_gain(y, y_split)
                if info_gain > max_info_gain:
                    best_feature_idx, best_split_value, max_info_gain = feature_idx, split_value, info_gain
        return best_feature_idx, best_split_value

    def _build_tree(self, X, y, depth):
        if depth == 0 or len(np.unique(y)) == 1:
            return np.bincount(y).argmax()

        best_feature_idx, best_split_value = self._find_best_split(X, y)
        if best_feature_idx is None:
            return np.bincount(y).argmax()

        X_left, X_right, y_left, y_right = self._split(X, y, best_feature_idx, best_split_value)

        node = {}
        node['feature_idx'] = best_feature_idx
        node['split_value'] = best_split_value
        node['left'] = self._build_tree(X_left, y_left, depth - 1)
        node['right'] = self._build_tree(X_right, y_right, depth - 1)
        return node

    def fit(self, X, y):
        self.tree_ = self._build_tree(X, y, self.max_depth)

    def _predict_sample(self, x, tree):
        if isinstance(tree, dict):
            if x[tree['feature_idx']] <= tree['split_value']:
                return self._predict_sample(x, tree['left'])
            else:
                return self._predict_sample(x, tree['right'])
        else:
            return tree

    def predict(self, X):
        return np.array([self._predict_sample(x, self.tree_) for x in X])


X = np.array([
    [25, 55000, 0, 600],
    [30, 75000, 1, 700],
    [35, 40000, 0, 580],
    [45, 20000, 1, 300],
    [20, 10000, 0, 150],
    [35, 22000, 1, 350],
    [50, 80000, 0, 780],
    [40, 70000, 1, 750],
    [45, 45000, 0, 600],
    [55, 35000, 1, 450],
    [30, 65000, 0, 700],
    [20, 20000, 1, 400],
    [25, 35000, 0, 500],
    [35, 50000, 1, 630],
    [40, 60000, 0, 600]
])
y = np.array([1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1])

tree = DecisionTree(max_depth=3)
tree.fit(X, y)


new_data = np.array([
    [30, 60000, 1, 700],
    [45, 40000, 0, 600],
    [25, 30000, 1, 350]
])
predictions = tree.predict(new_data)
print("Predictions:", predictions)