import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import category_encoders as ce
from sklearn import metrics
from sklearn.metrics import confusion_matrix

int_attrs = ['age', 'r_decile_score', 'priors_count', 'v_decile_score', 'is_recid']

datas = pd.read_csv('../data/compas.csv')[int_attrs]

datas.columns = int_attrs

# print(car['class'].value_counts())

datas.loc[datas.is_recid == 1, 'is_recid'] = -1
datas.loc[datas.is_recid == 0, 'is_recid'] = 1

X = datas.drop(['is_recid'], axis=1)
y = datas['is_recid']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2, stratify=y)

encoder = ce.OrdinalEncoder(cols=['age', 'r_decile_score', 'priors_count', 'v_decile_score'])

X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)

clf_standard = DecisionTreeClassifier(random_state=42)

clf_standard.fit(X_train, y_train)
y_preds_standard = clf_standard.predict(X_test)

print(f"Acc train: {clf_standard.score(X_train, y_train)}")
print(f"Acc test: {metrics.accuracy_score(y_test, y_preds_standard)}")

print("################################")

clf_entropy = DecisionTreeClassifier(random_state=42, criterion="entropy", max_depth=8)

clf_entropy.fit(X_train, y_train)
y_preds_entropy = clf_entropy.predict(X_test)

print(f"Acc train entropy : {clf_entropy.score(X_train, y_train)}")
print(f"Acc test entropy: {metrics.accuracy_score(y_test, y_preds_entropy)}")

print("################################")

cm = confusion_matrix(y_test, y_preds_entropy)

print(cm)

