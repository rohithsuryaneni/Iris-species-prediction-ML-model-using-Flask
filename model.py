import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

#load the CSV file
df=pd.read_csv("iris.csv")
print(df.sample(5))
print(df.shape)

# select independent and dependent "variables
X=df[["Sepal_Length","Sepal_Length","Petal_Length","Petal_Width"]]
y=df["Class"]

#split the dataset into train and test
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=50)


#instantiate the model
classifier=RandomForestClassifier()

#fit the model
classifier.fit(X_train,y_train)

print(classifier.score(X_test,y_test))
print(classifier.predict([[6.1,2.8,4.0,1.3 ]]))

# make pickle file of our model
pickle.dump(classifier,open("model.pkl","wb"))


