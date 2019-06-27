#!/usr/bin/env python
# coding: utf-8

# # TITANIC SURVIVAL PREDICTION
In this blog,I will be doing the Titanic survival prediction through the famous Titanic dataset(Kaggle).
This project aims at predicting whether a given passenger in titanic survives or not
# In[4]:


# Importing necessary libraires

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib import style
get_ipython().run_line_magic('matplotlib', 'inline')



# In[5]:


#Algorithms
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


# In[6]:


#loading data

data=pd.read_csv('../day3/data/titanic1.csv')


# # Data Exploration/Analysis

# In[7]:


data.keys()


# In[8]:


data.info()

Here we can observe that 2 features are of float and 5 features are of int type and 5 features are strings.

Hence,for computation using machine learning models,all values should be converted into int.

Also,the following is the description about the columns:
    
    PassengerId    : Id of passenger
    Survived       : Passenger survived or not
    Pclass         : Ticket class
    Name           : Name of Passenger
    Sex            : Sex
    Age            : Age 
    SibSp          : number of siblings/spouses aboard the Titanic
    Parch          : number of parents/children aboard the Titanic
    Ticket         : Ticket number
    Fare           : Passenger Fare
    Cabin          : Cabin number
    Embarked       : Port of Embarkation
# In[9]:


#printing the top 5 rows of data
data.head()


# In[10]:


data.describe()

We can see that the features have widely different ranges,so we need to convert into roughly the same scale. 

We can also spot some more features, that contain missing values (NaN = not a number), that we need to deal with.
# In[11]:


num_nan=data.isnull().sum()
num_nan

As we can see,there are 177 Nan values in Age which should be dealt carefully in analysis.

Whereas,the column Cabin can be removed as it has high number of Nan values.

The column embarked is having 2 missing values and it can be easily filled.
# In[12]:


#printing the total number of nan values along with the percentage.

num_nan=data.isnull().sum().sort_values(ascending=False)
percentage1=data.isnull().sum()/data.isnull().count()*100
percentage2=(round(percentage1,1)).sort_values(ascending=False)
missing_data=pd.concat([num_nan,percentage2],axis=1,keys=['Total','%'])
missing_data


# # Data Visualisation

# In[13]:


survived='Survived'
not_survived='not Survived'

fig, axes=plt.subplots(nrows=1,ncols=2, figsize=(10,4))
women=data[data['Sex']=='female']
men=data[data['Sex']=='male']

ax=sns.distplot(women[women['Survived']==1].Age.dropna(),bins=18,label='survived',ax=axes[0],kde=False)
ax=sns.distplot(women[women['Survived']==0].Age.dropna(),bins=40,label='not_survived',ax=axes[0],kde=False)

ax.legend()
ax.set_title('Female')

ax=sns.distplot(men[men['Survived']==1].Age.dropna(),bins=18,label=survived,ax=axes[1],kde=False)
ax=sns.distplot(men[men['Survived']==0].Age.dropna(),bins=40,label=not_survived,ax=axes[1],kde=False)

ax.legend()
ax.set_title('Men')




# In[14]:


sns.barplot(x='Pclass',y='Survived',data=data)

Clearly it shows that Pclass is contributing to a persons chance of survival.
Now,another way of plotting can be done as follows:
# In[15]:


grid=sns.FacetGrid(data,col='Survived',row='Pclass',size=3.2,aspect=1.8)
grid.map(plt.hist,'Age',alpha=0.5,bins=20)
grid.add_legend()


# # Data preprocessing

# In[16]:


data=data.drop('PassengerId',axis=1)


# In[17]:


#filling nan values in 'Age' with the mean value of that coulmn,that is average age.


# In[18]:


m=data.shape[0]
for dataset in range(m):
    mean=data['Age'].mean()
    std=data['Age'].std()
    isnull=data['Age'].isnull().sum()
    
    rand_age=np.random.randint(mean-std,mean+std,size=isnull)
    age_slice = data["Age"].copy()

    age_slice[np.isnan(age_slice)] = rand_age
    data["Age"] = age_slice
    data["Age"] = data["Age"].astype(int)
data["Age"].isnull().sum()
    


# In[19]:


data['Age']


# In[20]:


data['Embarked'].describe()


# In[21]:


common_value='S'
for i in range(data.shape[0]):
    data['Embarked']=data['Embarked'].fillna(common_value)


# In[22]:


data['Embarked']


# In[23]:


data['Embarked'].isnull().sum()


# In[24]:


embark_mapping={
    'S':0,
    'C':1,
    'Q':2
}
data.Embarked=data.Embarked.map(embark_mapping)


# In[25]:


data['Embarked'].head()


# In[ ]:





# In[ ]:





# In[26]:


sex_mapping={'male':0,
            'female':1}
data.Sex=data.Sex.map(sex_mapping)


# In[27]:


data['Sex'].head()


# In[28]:


data['Fare']=data['Fare'].astype('int')


# In[29]:


data['Fare'].head()


# In[30]:


data.head()


# In[31]:


data=data.drop(['Ticket','Cabin','Name'],axis=1)


# In[32]:


data.head(10)


# In[33]:


data.shape


# In[34]:


split = int(0.8*data.shape[0])


# In[35]:


split


# In[36]:


x_train=data.iloc[:split,1:]
x_test=data.iloc[split:,1:]

y_train=data.iloc[:split,0]
y_test=data.iloc[split:,0]

print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)

using different machine learning classifiers to predict
# In[45]:


# using logistic regression from sklearn

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)
lr.score(x_test,y_test)*100


# In[46]:


#stochastic gradient descent

sgd = linear_model.SGDClassifier(max_iter=5000, tol=None)
sgd.fit(x_train, y_train)

acc_sgd = round(sgd.score(x_test, y_test) * 100, 2)
acc_sgd


# In[48]:


#using random forest classifier

random_forest = RandomForestClassifier(n_estimators=1000)
random_forest.fit(x_train, y_train)
acc_random_forest = round(random_forest.score(x_test, y_test) * 100, 2)
acc_random_forest


# In[49]:


#using decision tree classifier
regressor = DecisionTreeClassifier(random_state=0)
regressor.fit(x_train,y_train)
regressor.score(x_test,y_test)*100


# In[50]:


#using KNeighborsClassifier
knc=KNeighborsClassifier()
knc.fit(x_train,y_train)
knc.score(x_test,y_test)*100


# In[ ]:




