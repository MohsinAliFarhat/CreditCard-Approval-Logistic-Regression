import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

df = pd.read_csv('E:\Datasets\credit-card-approval\cc_approvals.data.txt')

#initial insight
print(df.head())

#Stats summary
print(df.describe())

info = df.info()
print(info)

#check for missing values
print(df.tail(17))

#handling missing values
df = df.replace('?',np.NaN)

print(df.tail(17))

df.fillna(df.mean(),inplace=True)

# Count not availables
print(df.isnull().sum())

for col in df.columns:
    if(df[col].dtypes=="object"):
        df = df.fillna(df[col].value_counts().index[0])
        
print(df.isnull().sum())

#conveting non-numeric to numeric
le = LabelEncoder()

for col in df.columns:
    if df[col].dtypes == 'object':
        df[col] = le.fit_transform(df[col])
        
print(df.head())
df = df.drop([df.columns[10],df.columns[13]], axis=1)
print(df.head())
#return numpy representation of data
df = df.values
print(df)
X,y = df[:,0:13] , df[:,13]
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(rescaledX,
                                y,
                                test_size=0.33,
                                random_state=42)

logreg = LogisticRegression()

# Fit logreg to the train set
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

# Get the accuracy score of logreg model and print it
print("Accuracy of logistic regression classifier: ", logreg.score(X_test, y_test))

# Print the confusion matrix of the logreg model
print('Confusion Matrix')
print(confusion_matrix(y_test, y_pred))