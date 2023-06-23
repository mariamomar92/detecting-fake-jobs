#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
import pandas as pd
import numpy
import numpy as np
from sphinx.roles import RFC

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
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from abc import ABC, abstractmethod
from sklearn.metrics import r2_score
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
from sklearn.metrics import confusion_matrix
# from sklearn.metrics._classification import confusion_matrix,accuracy_score
import warnings
warnings.filterwarnings('ignore')

##TRAINDATA
dftrain = pd.read_csv(r'C:\Users\Lenovo\OneDrive\Documents\year 3\semester 2\ccs\train.csv')
dftrain.head()

#preprocessing
#drop and merge
dftrain["job_info"] = dftrain["job description"].str.cat(dftrain["company information"],sep="-")
# print(dftrain['job_info'])
dftrain = dftrain.drop(['job description','company information'],axis=1)
dftrain = dftrain.drop(['salary offered for the job'], axis=1)
# dftrain.isnull().sum()


#fill null
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
# dftrain.isnull().sum()


#cleaning str col. using reg exp
dftrain['job_info'] = dftrain['job_info'].str.replace(r"[^a-zA-Z\d]+"," ")
dftrain['job title'] = dftrain['job title'].str.replace(r"[^a-zA-Z\d]+"," ")
dftrain['office location'] = dftrain['office location'].str.replace(r"[^a-zA-Z\d]+"," ")
dftrain['job requirements'] = dftrain['job requirements'].str.replace(r"[^a-zA-Z\d]+"," ")
dftrain['benefits'] = dftrain['benefits'].str.replace(r"[^a-zA-Z\d]+"," ")
# print(dftrain['job_info'])


#label encoding cat.cols.
catg = ['industry','fake?','function','education required','experience required','employment_type','department']
for i in catg:
    le = preprocessing.LabelEncoder()
    dftrain[i]= le.fit_transform(dftrain[i])
# dftrain.head()


##stopwords
stopword = nltk.corpus.stopwords.words('english')
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
# dftrain['job_info'].head()


##lemm
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()
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
# dftrain['job title'].head()


dftrain['combined'] = dftrain['job title'].astype(str) + ' ' + dftrain['job_info'] + ' ' + dftrain['benefits'] + ' ' + dftrain['job requirements'] + ' ' + dftrain['office location']
dftrain['combined'].head()
dftrain = dftrain.drop(['job title','job_info','benefits','job requirements','office location'],axis=1)


# ngrams
# from nltk import bigrams
# LTrain = [x for x in dftrain['combined'] for x in bigrams(x.split())]
# # print (L)
# from collections import Counter
# cTrain = Counter(LTrain)
# #print(cTrain)
# countsTrain = np.array(list(cTrain.values()))


#tfidf
tfidf_vectorizer = TfidfVectorizer()
tfidfvec1Train = tfidf_vectorizer.fit_transform(dftrain['combined'].values)
# print(tfidfvec1Train)
# print(np.shape(tfidfvec1Train))

dftTrain = pd.DataFrame(tfidfvec1Train.toarray())
dfallTrain = pd.concat([dftTrain, dftrain], axis=1)
# dfall.head()
dfallTrain = dfallTrain.drop('combined', axis=1)
# dfall.head()
# dfall.info()
# dfallTrain[np.isnan(dfallTrain)] = 0
# dfall.info()
#########################

#########################
#TESTDATA
dftest = pd.read_csv(r'C:\Users\Lenovo\OneDrive\Documents\year 3\semester 2\ccs\test.csv')
#dftest.head()

#preprocessing
#drop and merge
dftest["job_info"] = dftest["job description"].str.cat(dftest["company information"], sep="-")
dftest = dftest.drop(['job description', 'company information'], axis=1)
dftest = dftest.drop(['salary offered for the job'], axis=1)
# print(dftest.isnull().sum())

#fill null
dftest['office location'].fillna(dftest['office location'].mode()[0], inplace=True)
dftest['department'].fillna(dftest['department'].mode()[0], inplace=True)
dftest['benefits'].fillna(dftest['benefits'].mode()[0], inplace=True)
dftest['job requirements'].fillna(dftest['job requirements'].mode()[0], inplace=True)
dftest['job_info'].fillna(dftest['job_info'].mode()[0], inplace=True)
dftest['function'].fillna(dftest['function'].mode()[0], inplace=True)
dftest['industry'].fillna(dftest['industry'].mode()[0], inplace=True)
dftest['experience required'].fillna(dftest['experience required'].mode()[0], inplace=True)
dftest['education required'].fillna(dftest['education required'].mode()[0], inplace=True)
dftest['employment_type'].fillna(dftest['employment_type'].mode()[0], inplace=True)
# print(dftest.isnull().sum())



#cleaning string columns using reg exp
dftest['job_info'] = dftest['job_info'].str.replace(r"[^a-zA-Z\d]+", " ")
dftest['job title'] = dftest['job title'].str.replace(r"[^a-zA-Z\d]+", " ")
dftest['office location'] = dftest['office location'].str.replace(r"[^a-zA-Z\d]+", " ")
dftest['job requirements'] = dftest['job requirements'].str.replace(r"[^a-zA-Z\d]+", " ")
dftest['benefits'] = dftest['benefits'].str.replace(r"[^a-zA-Z\d]+", " ")
# print(dftest['job_info'])



#label encoding cat. col.
catg = ['industry','fake?','function','education required','experience required','employment_type','department']
for i in catg:
    le = preprocessing.LabelEncoder()
    dftest[i]=le.fit_transform(dftest[i])
# print(dftest[catg])


##stopwords
stopword = nltk.corpus.stopwords.words('english')

dftest['job_info'] = dftest['job_info'].str.lower()
dftest['job title'] = dftest['job title'].str.lower()
dftest['office location'] = dftest['office location'].str.lower()
dftest['job requirements'] = dftest['job requirements'].str.lower()
dftest['benefits'] = dftest['benefits'].str.lower()
for i in stopword:
    dftest['job_info'] = dftest['job_info'].replace(to_replace=r'\b%s\b' % i, value="", regex=True)
    dftest['job title'] = dftest['job title'].replace(to_replace=r'\b%s\b' % i, value="", regex=True)
    dftest['office location'] = dftest['office location'].replace(to_replace=r'\b%s\b' % i, value="", regex=True)
    dftest['job requirements'] = dftest['job requirements'].replace(to_replace=r'\b%s\b' % i, value="", regex=True)
    dftest['benefits'] = dftest['benefits'].replace(to_replace=r'\b%s\b' % i, value="", regex=True)
# print(dftest['job_info'])

#lemm
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()
for i in range(len(dftest['job title'])):
    words = nltk.word_tokenize(dftest['job title'][i])
    words =[lemmatizer.lemmatize(word,pos='v')for word in words]
    dftest['job title'][i]= ' '.join(words)


for i in range(len(dftest['job_info'])):
    words = nltk.word_tokenize(dftest['job_info'][i])
    words =[lemmatizer.lemmatize(word,pos='v')for word in words]
    dftest['job_info'][i]= ' '.join(words)

for i in range(len(dftest['job requirements'])):
    words = nltk.word_tokenize(dftest['job requirements'][i])
    words =[lemmatizer.lemmatize(word, pos='v')for word in words]
    dftest['job requirements'][i]=' '.join(words)

for i in range(len(dftest['benefits'])):
    words = nltk.word_tokenize(dftest['benefits'][i])
    words =[lemmatizer.lemmatize(word,pos='v')for word in words]
    dftest['benefits'][i]=' '.join(words)
# print(dftest['job_info'])


dftest['combined'] = dftest['job title'].astype(str) + ' ' + dftest['job_info'] + ' ' + dftest['benefits'] + ' ' + dftest['job requirements'] + ' ' + dftest['office location']
# print(dftest['combined'].head())
dftest = dftest.drop(['job title', 'job_info', 'benefits', 'job requirements', 'office location'], axis=1)

##ngrams
# from nltk import bigrams
# LTest = [x for x in dftest['combined'] for x in bigrams(x.split())]
# #print (L)
# from collections import Counter
# cTest = Counter(LTest)
# #print(cTest) ##
# countsTest = np.array(list(cTest.values()))

#tfidf
tfidfvec1Test = tfidf_vectorizer.transform(dftest['combined'].values)
# print(tfidfvec1Test)
# print(np.shape(tfidfvec1Test))

dftTest = pd.DataFrame(tfidfvec1Test.toarray())
dfallTest = pd.concat([dftTest, dftest], axis=1)
# dfallTest.head()
dfallTest = dfallTest.drop('combined', axis=1)
# dfallTest.head()
# dfallTest.info()
# dfallTest[np.isnan(dfallTest)] = 0
# dfallTest.info()
###############

# establish input and output
##train
xTrain = dfallTrain.drop(['fake?'], axis=1)
yTrain = dfallTrain['fake?']

##test
xTest = dfallTest.drop(['fake?'], axis=1)
yTest = dfallTest['fake?']

#split data into training and validating sets
x_train, x_valid, y_train, y_valid = train_test_split(xTrain, yTrain, test_size=0.25, random_state=42)


##Scaling
scaler = StandardScaler()
x_train = pd.DataFrame(scaler.fit_transform(x_train))
x_valid = pd.DataFrame(scaler.transform(x_valid))
xTest = pd.DataFrame(scaler.transform(xTest))
# yTest = pd.DataFrame(scaler.fit_transform(yTest))

# print(np.shape(x_train))
# print(np.shape(x_valid))
# print(np.shape(xTest))

##models
knn = KNeighborsClassifier(n_neighbors=5)
LR = LogisticRegression(solver="liblinear")
DTC = DecisionTreeClassifier(max_depth=3)
RFC = RandomForestClassifier(max_depth=20,
    min_samples_leaf=4,
    min_samples_split=10,
    max_features='sqrt',
    n_jobs=-1,
    n_estimators=600)
nvb = GaussianNB()
ada = AdaBoostClassifier()

###validation
##model 1
knn.fit(x_train, y_train)
# output=knn.predict(x_valid)
# print("KNN:")
# print('Accuracy score:', round(knn.score(x_train, y_train) * 100, 2))


##model 2
DTC.fit(x_train, y_train)
# output2 = DTC.predict(x_valid)
# print("DecisionTreeClassifier:")
# print('Accuracy score:', round(DTC.score(x_train, y_train) * 100, 2))

##model 3
RFC.fit(x_train, y_train)
# output3 = RFC.predict(x_valid)
# print("RandomForestClassifier:")
# print('Accuracy score:', round(RFC.score(x_train, y_train) * 100, 2))

##confsuion matrix
# cf_knn = confusion_matrix(x_valid, output)
# cf_DTC = confusion_matrix(x_valid, output2)
# cf_RFC = confusion_matrix(x_valid, output3)
# print("KNN: ", cf_knn)
# print("DecisionTreeClassifier: ", cf_knn)
# print("RandomForestClassifier: ", cf_knn)


###test
##model 1
# knn.fit(xTest, yTest)
output=knn.predict(xTest)
print("KNN:")
print(' score: ', round(knn.score(xTest, yTest) * 100, 2))
print('R2 score: ', round(r2_score(yTest, output) * 100, 2))
print('Accuracy score: ', round(accuracy_score(yTest, output) * 100, 2))


##model 2
# DTC.fit(xTest, yTest)
output2 = DTC.predict(xTest)
print("DecisionTreeClassifier:")
print('score: ', round(DTC.score(xTest, yTest) * 100, 2))
print('R2 score: ', round(r2_score(yTest, output2) * 100, 2))
print('Accuracy score: ', round(accuracy_score(yTest, output2) * 100, 2))

##model 3
# RFC.fit(xTest, yTest)
output3 = RFC.predict(xTest)
print("RandomForestClassifier:")
print('score: ', round(RFC.score(xTest, yTest) * 100, 2))
print('r2 score: ', round(r2_score(yTest, output3) * 100, 2))
print('Accuracy score: ', round(accuracy_score(yTest, output3) * 100, 2))

##confsuion matrix
cf_knn = confusion_matrix(yTest, output)
cf_DTC = confusion_matrix(yTest, output2)
cf_RFC = confusion_matrix(yTest, output3)
print("KNN: ", cf_knn)
print("DecisionTreeClassifier: ", cf_DTC)
print("RandomForestClassifier: ", cf_RFC)
print("Precision Score of rfc : ",precision_score(yTest, output3, pos_label='positive',average='macro'))
print("Recall Score of rfc: ",recall_score(yTest, output3,pos_label='positive',average='micro'))
print('f1 score of rfc: %.2f',f1_score(yTest,output3,pos_label='positive',average='weighted'))

print("Precision Score of knn: ",precision_score(yTest, output, pos_label='positive',average='macro'))
print("Recall Score of knn: ",recall_score(yTest, output,pos_label='positive',average='micro'))
print('f1 score of knn: %.2f',f1_score(yTest,output,pos_label='positive',average='weighted'))


print("Precision Score of dtc: ",precision_score(yTest, output2, pos_label='positive',average='macro'))
print("Recall Score of dtc: ",recall_score(yTest, output2,pos_label='positive',average='micro'))
print('f1 score of dtc: %.2f',f1_score(yTest,output2,pos_label='positive',average='weighted'))
