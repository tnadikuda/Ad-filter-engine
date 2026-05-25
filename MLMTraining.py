import pandas as pd
from sklearn import metrics
import warnings
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
import pickle

warnings.filterwarnings('ignore')

df = pd.read_csv("data/projectDataSet.csv")

X = df.drop(["url", "class", "label", "hostname"], axis=1)
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

ML_Model = []
accuracy = []
f1_score = []
recall = []
precision = []

def storeResults(model, a, b, c, d):
    ML_Model.append(model)
    accuracy.append(round(a, 3))
    f1_score.append(round(b, 3))
    recall.append(round(c, 3))
    precision.append(round(d, 3))

# Logistic Regression
log = LogisticRegression()
log.fit(X_train, y_train)
y_test_log = log.predict(X_test)
acc_test_log = metrics.accuracy_score(y_test, y_test_log)
f1_score_test_log = metrics.f1_score(y_test, y_test_log)
recall_score_train_log = metrics.recall_score(y_train, log.predict(X_train))
precision_score_train_log = metrics.precision_score(y_train, log.predict(X_train))
print("Logistic Regression : Accuracy on test Data: {:.3f}".format(acc_test_log))
storeResults('Logistic Regression', acc_test_log, f1_score_test_log, recall_score_train_log, precision_score_train_log)

# K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
y_test_knn = knn.predict(X_test)
acc_test_knn = metrics.accuracy_score(y_test, y_test_knn)
f1_score_test_knn = metrics.f1_score(y_test, y_test_knn)
recall_score_train_knn = metrics.recall_score(y_train, knn.predict(X_train))
precision_score_train_knn = metrics.precision_score(y_train, knn.predict(X_train))
print("K-Nearest Neighbors : Accuracy on test Data: {:.3f}".format(acc_test_knn))
storeResults('K-Nearest Neighbors', acc_test_knn, f1_score_test_knn, recall_score_train_knn, precision_score_train_knn)

# Support Vector Machine
param_grid = {'gamma': [0.1], 'kernel': ['rbf', 'linear']}
svc = GridSearchCV(SVC(), param_grid)
svc.fit(X_train, y_train)
y_test_svc = svc.predict(X_test)
acc_test_svc = metrics.accuracy_score(y_test, y_test_svc)
f1_score_test_svc = metrics.f1_score(y_test, y_test_svc)
recall_score_train_svc = metrics.recall_score(y_train, svc.predict(X_train))
precision_score_train_svc = metrics.precision_score(y_train, svc.predict(X_train))
print("Support Vector Machine : Accuracy on test Data: {:.3f}".format(acc_test_svc))
storeResults('Support Vector Machine', acc_test_svc, f1_score_test_svc, recall_score_train_svc, precision_score_train_svc)

# Decision Tree
tree = DecisionTreeClassifier(max_depth=30)
tree.fit(X_train, y_train)
y_test_tree = tree.predict(X_test)
acc_test_tree = metrics.accuracy_score(y_test, y_test_tree)
f1_score_test_tree = metrics.f1_score(y_test, y_test_tree)
recall_score_train_tree = metrics.recall_score(y_train, tree.predict(X_train))
precision_score_train_tree = metrics.precision_score(y_train, tree.predict(X_train))
print("Decision Tree : Accuracy on test Data: {:.3f}".format(acc_test_tree))
storeResults('Decision Tree', acc_test_tree, f1_score_test_tree, recall_score_train_tree, precision_score_train_tree)

# Random Forest
forest = RandomForestClassifier(n_estimators=10)
forest.fit(X_train, y_train)
y_test_forest = forest.predict(X_test)
acc_test_forest = metrics.accuracy_score(y_test, y_test_forest)
f1_score_test_forest = metrics.f1_score(y_test, y_test_forest)
recall_score_train_forest = metrics.recall_score(y_train, forest.predict(X_train))
precision_score_train_forest = metrics.precision_score(y_train, forest.predict(X_train))
print("Random Forest : Accuracy on test Data: {:.3f}".format(acc_test_forest))
storeResults('Random Forest', acc_test_forest, f1_score_test_forest, recall_score_train_forest, precision_score_train_forest)

# Gradient Boosting
gbc = GradientBoostingClassifier(max_depth=4, learning_rate=0.7)
gbc.fit(X_train, y_train)
y_test_gbc = gbc.predict(X_test)
acc_test_gbc = metrics.accuracy_score(y_test, y_test_gbc)
f1_score_test_gbc = metrics.f1_score(y_test, y_test_gbc)
recall_score_train_gbc = metrics.recall_score(y_train, gbc.predict(X_train))
precision_score_train_gbc = metrics.precision_score(y_train, gbc.predict(X_train))
print("Gradient Boosting : Accuracy on test Data: {:.3f}".format(acc_test_gbc))
storeResults('Gradient Boosting Classifier', acc_test_gbc, f1_score_test_gbc, recall_score_train_gbc, precision_score_train_gbc)

# XGBoost
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
y_test_xgb = xgb.predict(X_test)
acc_test_xgb = metrics.accuracy_score(y_test, y_test_xgb)
f1_score_test_xgb = metrics.f1_score(y_test, y_test_xgb)
recall_score_train_xgb = metrics.recall_score(y_train, xgb.predict(X_train))
precision_score_train_xgb = metrics.precision_score(y_train, xgb.predict(X_train))
print("XGBoost : Accuracy on test Data: {:.3f}".format(acc_test_xgb))
storeResults('XGBoost Classifier', acc_test_xgb, f1_score_test_xgb, recall_score_train_xgb, precision_score_train_xgb)

# Results
result = pd.DataFrame({
    'ML Model': ML_Model,
    'Accuracy': accuracy,
    'f1_score': f1_score,
    'Recall': recall,
    'Precision': precision,
})
sorted_result = result.sort_values(by=['Accuracy', 'f1_score'], ascending=False).reset_index(drop=True)
print(sorted_result)

# Save best model (Random Forest)
pickle.dump(forest, open('models/forest_model.pkl', 'wb'))
