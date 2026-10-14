
import pandas as pd
df=pd.read_csv('Brain.csv')
df
df.isnull().sum()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

df.isnull().sum()

x=df.drop('stroke',axis=1)
y=df['stroke']

catcols=x.select_dtypes(include='object').columns

catcols

numcols=x.select_dtypes(exclude='object').columns

numcols
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), catcols)
    ],
    remainder='passthrough'                                                                     
)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=RandomForestClassifier(n_estimators=100,random_state=42)
from sklearn.pipeline import Pipeline
pipeline=Pipeline(steps=[('preprocessor',preprocessor),('model',model)])

pipeline.fit(x_train,y_train)
y_pred=pipeline.predict(x_test)
y_pred

acc=accuracy_score(y_test,y_pred)
acc

cm=confusion_matrix(y_test,y_pred)
cm
cr=classification_report(y_test,y_pred)
print(cr)


import joblib

joblib.dump(pipeline,"brainstroke_model.pkl")



