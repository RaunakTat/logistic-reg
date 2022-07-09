import pandas as pd
dataframe = pd.read_csv('heart.csv')
dataframe.info()

#dividing data into input(x) and outpur(y)
x = dataframe.iloc[:,2:13].values
y = dataframe.iloc[:,13].values

#spliting both input and output data into two sets for training and testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 0)


#scaling the input data due to huge difference in values in different columns
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

#Applying the classifier alogorithm
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

#fitting the model
model.fit(x_train,y_train)

#predictor variable
y_pred = model.predict(x_test)
print(y_pred)  #predicted output values

#checking the accuracy of the model
from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred,y_test)*100)
