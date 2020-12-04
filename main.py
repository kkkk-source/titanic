import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df_test  = pd.read_csv("test.csv")
df_train = pd.read_csv("train.csv")

def log_inf():
    print(df_test.shape)
    print(df_train.shape)
    print(pd.isnull(df_test).sum())
    print(pd.isnull(df_train).sum())
    print(df_test.describe())
    print(df_train.describe())

df_train['Sex'].replace(['female','male'], [0,1], inplace=True)
df_test['Sex'].replace(['female','male'], [0,1], inplace=True)
df_train['Embarked'].replace(['Q','S', 'C'], [0,1,2], inplace=True)
df_test['Embarked'].replace(['Q','S', 'C'], [0,1,2], inplace=True)

mean = round((df_train["Age"].mean() + df_test["Age"].mean()) / 2)

df_train['Age'] = df_train['Age'].replace(np.nan, mean)
df_test['Age'] = df_test['Age'].replace(np.nan, mean)

bins = [0, 8, 15, 18, 25, 40, 60, 100]
names = ['1', '2', '3', '4', '5', '6', '7']

df_train['Age'] = pd.cut(df_train['Age'], bins, labels=names)
df_test['Age'] = pd.cut(df_test['Age'], bins, labels=names)
df_train.drop(['Cabin'], axis=1, inplace=True)
df_test.drop(['Cabin'], axis=1, inplace=True)
df_train = df_train.drop(['PassengerId','Name','Ticket'], axis=1)
df_test = df_test.drop(['Name','Ticket'], axis=1)
df_train.dropna(axis=0, how='any', inplace=True)
df_test.dropna(axis=0, how='any', inplace=True)

x = np.array(df_train.drop(['Survived'], 1))
y = np.array(df_train['Survived'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
ids = df_test['PassengerId']

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train, y_train)

print(f'confianza = {knn.score(x_train, y_train)}')
prediccion_knn = knn.predict(df_test.drop('PassengerId', axis=1))
out_knn = pd.DataFrame({ 'PassengerId' : ids, 'Survived': prediccion_knn })
print(out_knn.head(100))
