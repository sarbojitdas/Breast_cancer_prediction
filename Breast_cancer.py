import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



df=pd.read_csv('data.csv')
print(df.head(7))
 
#count the number of rowes and collumns

print(df.shape)

# count the number of empty (NAN,NAN,Na)value

print(df.isna().sum())

#drop the columns with all missing values
print(df.dropna(axis=1))
#count the number of column and row
print(df.shape)

#get a count of the number of malignant(M) or benign
print(df['diagnosis'].value_counts())

#visualizie the count

#sns.countplot(df['diagnosis'],label='count')
###plt.show()

#look at the data types to see which column needs to beincluded
print(df.dtypes)

#encode the catagorial data values
from sklearn.preprocessing import LabelEncoder
labelencoder_y=LabelEncoder()
df.iloc[:,1]=labelencoder_y.fit_transform(df.iloc[:,1].values)
print(labelencoder_y.fit_transform(df.iloc[:,1].values))
print(df.iloc[:,1])

#Create a pair plot
#sns.pairplot(df.iloc[:,1:6])

#get the correlation of the columns
print(df.iloc[:,:].corr())

#visualize this correlation
sns.heatmap(df.iloc[:,1:12].corr())
#plt.show()

#split the dataset into independent(x) and dependent dataset

x=df.iloc[:,2:31].values
y=df.iloc[:,1].values

#split the dataset into 75% training and 25% testing

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.75,random_state=0)
#scale the date(Feature scaling)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

x_train=sc.fit_transform(x_train)
x_test= sc.fit_transform(x_test)
#print(x_train)

#create function for the model

def models(x_train,y_train):
    #Logistic Regression
    from sklearn.linear_model import LogisticRegression
    log=LogisticRegression(random_state=0)
    log.fit(x_train,y_train)

    #Decision Tree

    from sklearn.tree import DecisionTreeClassifier

    tree=DecisionTreeClassifier(criterion='entropy', random_state=0)

    tree.fit(x_train,y_train)

    #Random Forest

    from sklearn.ensemble import RandomForestClassifier
    forest=RandomForestClassifier(n_estimators=10,criterion='entropy', random_state=0)
    forest.fit(x_train,y_train)

    #kmean

    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    classifier.fit(x_train, y_train)

    #print model accuracy

    print('[0]Logistic Regression Training Accuracy:',log.score(x_train,y_train))

    print('[1]Desicion Tree Classifier Training Accuracy:',tree.score(x_train,y_train))

    print('[2]Random Forest Training Accuracy:',forest.score(x_train,y_train))

    print('[3] Classifier Training Accuracy:',classifier.score(x_train,y_train))


    

    return log,tree,forest,classifier

model=models(x_train,y_train)

#test model accuracy on test data on confusion matrix
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test, model[0].predict(x_test))
for i in range(len(model)):
        TP=cm[0][0]
        TN=cm[1][1]
        FP=cm[0][1]
        FN=cm[1][0]

        print(cm)

        print('Testing Accuracy:',(TP+TN)/(TP+TN+FN+FP))

#another way to get metrics of the model

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score
for i in range(len(model)):
    print("Model",i)
    print(classification_report(y_test, model[0].predict(x_test)))

    print(accuracy_score(y_test, model[0].predict(x_test)))

    print()




    

