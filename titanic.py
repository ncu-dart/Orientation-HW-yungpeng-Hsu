import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
submission = pd.read_csv("gender_submission.csv")

train['Sex_Code'] = train['Sex'].map({'female' : 1, 'male': 0}).astype('int')
test['Sex_Code'] = test['Sex'].map({'female' : 1, 'male': 0}).astype('int')

Base = ['Sex_Code','Pclass']
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

#KNN
#測試k是多少可以得到最高的準確度
k_range = range(1,100)
k_scores = []
for k_number in k_range:
    knn = KNeighborsClassifier(n_neighbors=k_number)
    scores = cross_val_score(knn,train[Base],train['Survived'],cv=5,scoring='accuracy')
    k_scores.append(scores.mean())
print('max score:',max(k_scores))
print('knn best k:',k_scores.index(max(k_scores)) + 1)
plt.plot(k_range,k_scores)
plt.xlabel('Value of K')
plt.ylabel('Cross Validated Accuracy')
plt.show()  

knn = KNeighborsClassifier(n_neighbors=6)
#cross validation
scores = cross_val_score(knn,train[Base],train['Survived'],cv=5,scoring='accuracy')
print('knn score: ', scores)

knn.fit(train[Base],train['Survived'])
submit = knn.predict(test[Base])
submit = pd.DataFrame({'PassengerId': submission['PassengerId'], 'Survived':submit})
submit.to_csv("submit_knn.csv", index = False)
#kaggle 0.77511

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

dctree = DecisionTreeClassifier()
dctree.fit(train[Base],train['Survived'])
submit = dctree.predict(test[Base])
'''
#dot_data = tree.export_graphviz(dctree, out_file=None, feature_names=Base)
dotfile = open("C:/Users/徐永棚/ML_hw/dtree2.dot", 'w')
tree.export_graphviz(dctree, out_file = dotfile, feature_names = Base)
dotfile.close()
'''
dctree = DecisionTreeClassifier(max_depth=3)
scores = cross_val_score(dctree,train[Base],train['Survived'],cv=5,scoring='accuracy')
print('decision tree score: ', scores)

dctree = DecisionTreeClassifier(max_depth=3)
dctree.fit(train[Base],train['Survived'])
submit = dctree.predict(test[Base])
submit = pd.DataFrame({'PassengerId': submission['PassengerId'], 'Survived':submit})
submit.to_csv("submit_dctree.csv", index = False)
#kaggle 0.77511

from sklearn.naive_bayes import CategoricalNB
NB = CategoricalNB()
scores = cross_val_score(NB,train[Base],train['Survived'],cv=5,scoring='accuracy')
print('naive bayes score: ', scores)

NB.fit(train[Base],train['Survived'])
submit = NB.predict(test[Base])
submit = pd.DataFrame({'PassengerId': submission['PassengerId'], 'Survived':submit})
submit.to_csv("submit_naive.csv", index = False)
#kaggle 0.76555