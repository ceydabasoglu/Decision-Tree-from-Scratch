import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix


train_data = pd.read_csv("trainSet.csv")
X_train = train_data.iloc[:, :-1].values
y_train = train_data["class"].values

test_data = pd.read_csv("testSet.csv")
X_test = test_data.iloc[:, :-1].values
y_test = test_data["class"].values


class TreeNode:
    def __init__(self, data, attribute=None, threshold=None, left=None, right=None, result=None):
        self.data = data
        self.attribute = attribute
        self.threshold = threshold
        self.left = left
        self.right = right
        self.result = result

def decision_tree_train(data, target, max_depth=None):
    if len(set(target)) == 1 or (max_depth is not None and max_depth == 0):
        return TreeNode(data=data, result=Counter(target).most_common(1)[0][0])

    if len(data[0]) == 0:
        return TreeNode(data=data, result=Counter(target).most_common(1)[0][0])

    best_attribute, best_threshold = find_best_split(data, target)
    left_data, left_target, right_data, right_target = split_data(data, target, best_attribute, best_threshold)

    if not left_data.any() or not right_data.any():
        return TreeNode(data=data, result=Counter(target).most_common(1)[0][0])

    left_tree = decision_tree_train(left_data, left_target, max_depth=max_depth - 1 if max_depth is not None else None)
    right_tree = decision_tree_train(right_data, right_target, max_depth=max_depth - 1 if max_depth is not None else None)

    return TreeNode(data=data, attribute=best_attribute, threshold=best_threshold, left=left_tree, right=right_tree)



def find_best_split(data, target):
    best_gini = 1.0
    best_attribute = None
    best_threshold = None

    for col in range(len(data[0])):
        values = set(data[:, col])
        for value in values:
            left_indices = data[:, col] < value
            right_indices = data[:, col] >= value

            left_gini = gini_impurity(target[left_indices])
            right_gini = gini_impurity(target[right_indices])
            weighted_gini = (left_gini * sum(left_indices) + right_gini * sum(right_indices)) / len(data)

            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_attribute = col
                best_threshold = value

    return best_attribute, best_threshold


def split_data(data, target, attribute, threshold):
    left_indices = data[:, attribute] < threshold
    right_indices = data[:, attribute] >= threshold

    left_data = data[left_indices]
    left_target = target[left_indices]

    right_data = data[right_indices]
    right_target = target[right_indices]

    return left_data, left_target, right_data, right_target

def gini_impurity(target):
    if len(target) == 0:
        return 0
    if isinstance(target[0], str):
        p = (target == "good").sum() / len(target)
    else:
        p = (target == 1).sum() / len(target)
    return 1 - (p**2 + (1-p)**2)


def predict(tree, instance):
    if tree.result is not None:
        return tree.result
    if instance[tree.attribute] < tree.threshold:
        return predict(tree.left, instance)
    else:
        return predict(tree.right, instance)


def accuracy(tree, data, target):
    predictions = [predict(tree, instance) for instance in data]
    correct = sum(1 for p, t in zip(predictions, target) if p == t)
    return correct / len(target)



decision_tree = decision_tree_train(X_train, y_train, max_depth=6)


test_accuracy = accuracy(decision_tree, X_test, y_test)

print("Test Accuracy: {:.3f}".format(test_accuracy))

# Training set performance metrics
train_predictions = [predict(decision_tree, instance) for instance in X_train]
train_accuracy = accuracy_score(y_train, train_predictions)
train_recall = recall_score(y_train, train_predictions, pos_label="good")
train_precision = precision_score(y_train, train_predictions, pos_label="good")
train_f1_score = f1_score(y_train, train_predictions, pos_label="good")
train_confusion_matrix = confusion_matrix(y_train, train_predictions, labels=["good", "bad"])

print("Training Results:")
print("Accuracy: {:.3f}".format(train_accuracy))
print("True Positive Rate (Recall): {:.3f}".format(train_recall))
print("True Negative Rate: {:.3f}".format(train_confusion_matrix[1, 1] / (train_confusion_matrix[1, 0] + train_confusion_matrix[1, 1])))
print("Precision: {:.3f}".format(train_precision))
print("F1-Score: {:.3f}".format(train_f1_score))
print("Total number of TP:", train_confusion_matrix[0, 0])
print("Total number of TN:", train_confusion_matrix[1, 1])

# Test set performance metrics
test_predictions = [predict(decision_tree, instance) for instance in X_test]
test_accuracy = accuracy_score(y_test, test_predictions)
test_recall = recall_score(y_test, test_predictions, pos_label="good")
test_precision = precision_score(y_test, test_predictions, pos_label="good")
test_f1_score = f1_score(y_test, test_predictions, pos_label="good")
test_confusion_matrix = confusion_matrix(y_test, test_predictions, labels=["good", "bad"])

print("\nTest Results:")
print("Accuracy: {:.3f}".format(test_accuracy))
print("True Positive Rate (Recall): {:.3f}".format(test_recall))
print("True Negative Rate: {:.3f}".format(test_confusion_matrix[1, 1] / (test_confusion_matrix[1, 0] + test_confusion_matrix[1, 1])))
print("Precision: {:.3f}".format(test_precision))
print("F1-Score: {:.3f}".format(test_f1_score))
print("Total number of TP:", test_confusion_matrix[0, 0])
print("Total number of TN:", test_confusion_matrix[1, 1])


import matplotlib.pyplot as plt

def plot_tree_simple(tree, feature_names, class_names, parent_name=None, ax=None, pos=None, level=0, width=1., vertical_gap=0.2, xcenter=0.5):
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.axis('off')

    if pos is None:
        pos = {parent_name: (xcenter, 1 - level * vertical_gap)}

    dx = 1 / 2.0 
    nextx = xcenter - width/2 - dx/2

    if tree.result is not None:
        ax.text(pos[parent_name][0], pos[parent_name][1], str(tree.result), color='k', size=14, ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
    else:
        for child_name, child in [('True', tree.left), ('False', tree.right)]:
            nextx += dx
            pos[child_name] = (nextx, 1 - (level + 1) * vertical_gap)
            
            if isinstance(tree.threshold, (int, float)):  # Eğer threshold sayısal bir değerse
                text = f'{feature_names[tree.attribute]} <= {float(tree.threshold):.2f}'  # Sayısal değere dönüştürüldü
            else:
                text = f'{feature_names[tree.attribute]} <= {tree.threshold}'  # Sayısal değilse direkt olarak string olarak kullanıldı
            
            ax.text(pos[parent_name][0], pos[parent_name][1], text, color='k', size=14, ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
            ax.text(pos[child_name][0], pos[child_name][1], child_name, color='k', size=14, ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
            ax.plot([pos[parent_name][0], pos[child_name][0]], [pos[parent_name][1], pos[child_name][1]], color='k', linestyle='-', linewidth=2)
            plot_tree_simple(child, feature_names, class_names, parent_name=child_name, ax=ax, pos=pos, level=level + 1, width=width/2, vertical_gap=vertical_gap, xcenter=nextx - dx/2)


decision_tree = decision_tree_train(X_train, y_train, max_depth=6)

# Visualize the decision tree
plot_tree_simple(decision_tree, feature_names=train_data.columns[:-1], class_names=['bad', 'good'])
plt.show()





#decision tree nin her bir özniteliğin sınıflandırmadaki katkısını gösteren bir çubuk grafik.
#Bu, modelin hangi özniteliklere daha fazla önem verdiğini gösterir.
def calculate_feature_importance(tree):
    feature_importance = np.zeros(len(train_data.columns) - 1)

    def traverse(node, importance):
        if node.attribute is not None:
            importance[node.attribute] += 1
            traverse(node.left, importance)
            traverse(node.right, importance)

    traverse(tree, feature_importance)
    total_nodes = len([1 for value in feature_importance if value > 0])
    feature_importance /= total_nodes  # Normalize to get the average usage

    return feature_importance

importances = calculate_feature_importance(decision_tree)
features = train_data.columns[:-1]

plt.bar(features, importances)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.show()


#decision tree derinliği ile doğruluk arasındaki ilişkiyi gösteren bir çizgi grafik.
#Bu, ağacın ne kadar derinleşirse modelin doğruluğunun nasıl değiştiğini anlamak için. 
# Derinliklere göre doğruluk oranlarını hesapla
depths = range(1, 21)
accuracies = []

for depth in depths:
    tree = decision_tree_train(X_train, y_train, max_depth=depth)
    accuracy_value = accuracy_score(y_test, [predict(tree, instance) for instance in X_test])
    accuracies.append(accuracy_value)

plt.plot(depths, accuracies, marker='o')
plt.xlabel('Tree Depth')
plt.ylabel('Accuracy')
plt.title('Depth vs. Accuracy')
plt.show()


#confusion matrix visualize
from sklearn.metrics import confusion_matrix
import seaborn as sns

test_predictions = [predict(tree, instance) for instance in X_test]

cm = confusion_matrix(y_test, test_predictions)

labels = ["bad", "good"]
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()




class RandomForest:
    def __init__(self, n_trees, max_depth=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []

    def train(self, data, target):
        num_rows, num_cols = data.shape

        for _ in range(self.n_trees):
            # Random subspace method ile rastgele öznitelikler seç
            random_indices = np.random.choice(num_cols, size=int(np.sqrt(num_cols)), replace=False)
            subset_data = data[:, random_indices]

            
            tree = decision_tree_train(subset_data, target, max_depth=self.max_depth)
            self.trees.append(tree)

    def predict(self, data):
        if not self.trees:
            raise ValueError("Model not trained. Call train method first.")

        # Her bir decision tree'den tahminleri al
        predictions = np.array([predict(tree, instance) for instance in data]).T

        
        ensemble_predictions = np.array([np.unique(prediction)[np.argmax(np.unique(prediction, return_counts=True)[1])] for prediction in predictions])
        return ensemble_predictions


random_forest = RandomForest(n_trees=100, max_depth=6)
random_forest.train(X_train, y_train)


ensemble_test_predictions = random_forest.predict(X_test)


conf_matrix = confusion_matrix(y_test, ensemble_test_predictions, labels=["good", "bad"])
ensemble_test_accuracy = accuracy_score(y_test, ensemble_test_predictions)
ensemble_test_recall = recall_score(y_test, ensemble_test_predictions, average="weighted", zero_division=1)
ensemble_test_precision = precision_score(y_test, ensemble_test_predictions, average="weighted", zero_division=1)
ensemble_test_f1_score = f1_score(y_test, ensemble_test_predictions, average="weighted")
ensemble_test_confusion_matrix = confusion_matrix(y_test, ensemble_test_predictions, labels=["good", "bad"])


print(np.unique(y_test))
print(ensemble_test_predictions)

print("\nRandom Forest Test results:")
print("Accuracy: {:.3f}".format(ensemble_test_accuracy))
print("TPrate (Recall): {:.3f}".format(ensemble_test_recall))
print("TNrate: {:.3f}".format(ensemble_test_confusion_matrix[1, 1] / (ensemble_test_confusion_matrix[1, 0] + ensemble_test_confusion_matrix[1, 1])))
print("Precision: {:.3f}".format(ensemble_test_precision))
print("F-Score: {:.3f}".format(ensemble_test_f1_score))
print("Total number of TP: {}".format(conf_matrix[0, 0]))
print("Total number of TN: {}".format(conf_matrix[1, 1]))
