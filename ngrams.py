#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
import pandas as pd
import numpy
import numpy as np
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import string
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
# from nltk.stem import WordNetLemmatizer
# from nltk.stem.wordnet import WordNetLemmatizer
# from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from abc import ABC, abstractmethod

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# In[2]:


dftrain = pd.read_csv(r'C:\Users\Lenovo\OneDrive\Documents\year 3\semester 2\ccs\train.csv')
dftrain.head()


# In[3]:


dftrain["job_info"] = dftrain["job description"].str.cat(dftrain["company information"],sep="-")
print(dftrain['job_info'])


# In[4]:


dftrain = dftrain.drop(['job description','company information'],axis=1)


# In[5]:


dftrain = dftrain.drop(['salary offered for the job'],axis=1)


# In[6]:


dftrain.isnull().sum()


# In[7]:


dftrain['office location'].fillna(dftrain['office location'].mode()[0], inplace=True)
dftrain['department'].fillna(dftrain['department'].mode()[0], inplace=True)
dftrain['benefits'].fillna(dftrain['benefits'].mode()[0], inplace=True)
dftrain['job requirements'].fillna(dftrain['job requirements'].mode()[0], inplace=True)
dftrain['job_info'].fillna(dftrain['job_info'].mode()[0], inplace=True)
dftrain['function'].fillna(dftrain['function'].mode()[0], inplace=True)
dftrain['industry'].fillna(dftrain['industry'].mode()[0], inplace=True)
dftrain['experience required'].fillna(dftrain['experience required'].mode()[0], inplace=True)
dftrain['education required'].fillna(dftrain['education required'].mode()[0], inplace=True)
dftrain['employment_type'].fillna(dftrain['employment_type'].mode()[0], inplace=True)


# In[8]:


dftrain.isnull().sum()


# In[9]:


string.punctuation


# In[10]:


dftrain['job_info'] = dftrain['job_info'].str.replace(r"[^a-zA-Z\d]+"," ")
dftrain['job title'] = dftrain['job title'].str.replace(r"[^a-zA-Z\d]+"," ")
dftrain['office location'] = dftrain['office location'].str.replace(r"[^a-zA-Z\d]+"," ")
dftrain['job requirements'] = dftrain['job requirements'].str.replace(r"[^a-zA-Z\d]+"," ")
dftrain['benefits'] = dftrain['benefits'].str.replace(r"[^a-zA-Z\d]+"," ")


# In[11]:


print(dftrain['job_info'])


# In[12]:


catg = ['industry','fake?','function','education required','experience required','employment_type','department']
for i in catg:
    le = preprocessing.LabelEncoder()
    dftrain[i]= le.fit_transform(dftrain[i])


# In[13]:


dftrain.head()


# In[14]:


stopword = nltk.corpus.stopwords.words('english')


# In[15]:


dftrain['job_info'] = dftrain['job_info'].str.lower()
dftrain['job title'] = dftrain['job title'].str.lower()
dftrain['office location'] = dftrain['office location'].str.lower()
dftrain['job requirements'] = dftrain['job requirements'].str.lower()
dftrain['benefits'] = dftrain['benefits'].str.lower()
for i in stopword :
    dftrain['job_info'] = dftrain['job_info'].replace(to_replace=r'\b%s\b'%i, value="",regex=True)
    dftrain['job title'] = dftrain['job title'].replace(to_replace=r'\b%s\b'%i, value="",regex=True)
    dftrain['office location'] = dftrain['office location'].replace(to_replace=r'\b%s\b'%i, value="",regex=True)
    dftrain['job requirements'] = dftrain['job requirements'].replace(to_replace=r'\b%s\b'%i, value="",regex=True)
    dftrain['benefits'] = dftrain['benefits'].replace(to_replace=r'\b%s\b'%i, value="",regex=True)


# In[16]:


dftrain['job_info'].head()


# In[17]:


w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()


# In[18]:


for i in range(len(dftrain['job title'])):
    words = nltk.word_tokenize(dftrain['job title'][i])
    words =[lemmatizer.lemmatize(word,pos='v')for word in words]
    dftrain['job title'][i]=' '.join(words)


for i in range(len(dftrain['job_info'])):
    words = nltk.word_tokenize(dftrain['job_info'][i])
    words =[lemmatizer.lemmatize(word,pos='v')for word in words]
    dftrain['job_info'][i]=' '.join(words)

for i in range(len(dftrain['job requirements'])):
    words = nltk.word_tokenize(dftrain['job requirements'][i])
    words =[lemmatizer.lemmatize(word,pos='v')for word in words]
    dftrain['job requirements'][i]=' '.join(words)

for i in range(len(dftrain['benefits'])):
    words = nltk.word_tokenize(dftrain['benefits'][i])
    words =[lemmatizer.lemmatize(word,pos='v')for word in words]
    dftrain['benefits'][i]=' '.join(words)    


# In[19]:


dftrain['job title'].head()


# In[20]:


# tokenize
# dftrain['job_info'] = dftrain['job_info'].apply(nltk.word_tokenize)
# dftrain['job title'] = dftrain['job title'].apply(nltk.word_tokenize)
# dftrain['office location'] = dftrain['office location'].apply(nltk.word_tokenize)
# dftrain['benefits'] = dftrain['benefits'].apply(nltk.word_tokenize)
# dftrain['job requirements'] = dftrain['job requirements'].apply(nltk.word_tokenize)

# print(dftrain['job_info'])


# In[21]:


# dftext = dftrain.loc[:,['job_info','job title','office location','benefits','job requirements']]
# dftext.head()


# In[22]:


# dftrain['text'] = dftrain[['job_info', 'benefits']].agg('-'.join, axis=1)
# print(dftrain['text'])
dftrain['combined'] = dftrain['job title'].astype(str) + '_' + dftrain['job_info'] + '_' + dftrain['benefits']+dftrain['job requirements']+dftrain['office location']
dftrain['combined'].head()

print(dftrain['combined'])

# In[23]:


dftrain = dftrain.drop(['job title','job_info','benefits','job requirements','office location'],axis=1)


# In[24]:


# ngrams

from nltk import bigrams

L = [x for x in dftrain['combined'] for x in bigrams(x.split())]
# print (L)


# In[25]:


from collections import Counter
c = Counter(L)
print (c)


# In[26]:


counts = np.array(list(c.values()))


# In[27]:


# #tfidf

# tfidf_vectorizer = TfidfVectorizer() 

# tfidfvec1 = tfidf_vectorizer.fit_transform(dftrain['combined'].values)


# In[28]:


# print(tfidfvec1)


# In[29]:


# print(np.shape(tfidfvec1))


# In[30]:


dft = pd.DataFrame(counts)

# dft = [float(x) for x in n]
dfall = pd.concat([dft, dftrain], axis=1)
dfall.head()


# In[31]:


dfall = dfall.drop('combined',axis=1)


# In[32]:


dfall.head()


# In[33]:


# dfall['fake?']
dfall.info()


# In[41]:


dfall[np.isnan(dfall)] = 0

dfall.info()


# In[42]:


# establish input and output
X = dfall.drop(['fake?'], axis=1)
y = dfall['fake?']
 # split data into training and testing set
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# In[43]:


print(np.shape(x_train))
print(np.shape(x_test))


# In[44]:


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)


# In[ ]:


print('Accuracy score:', round(knn.score(x_train, y_train) * 100, 2))


# In[ ]:


LR = LogisticRegression(solver = "liblinear")
DTC = DecisionTreeClassifier()
# xgb = xgb.XGBClassifier(learining_rate = 0.1, n_estimators = 10,colsample_bytree = 0.3,max_depth = 5 ,alpha =10)
# lgb = LGBMClassifier()
nvb = GaussianNB()
ada=AdaBoostClassifier()


# In[ ]:


DTC.fit(x_train,y_train)
output2 = DTC.predict(x_test)
print("DecisionTreeClassifier:")
print('Accuracy score:', round(DTC.score(x_train, y_train) * 100, 2))


# In[ ]:


# RFC.fit(x_train,y_train)
# output3 = RFC.predict(x_test)
# print("RandomForestClassifier:")
# print('Accuracy score:', round(RFC.score(x_train, y_train) * 100, 2))


# In[ ]:





from sklearn.metrics import confusion_matrix
output=knn.predict(x_test)
cf = confusion_matrix(y_test, output)

print(cf)

# In[ ]:




