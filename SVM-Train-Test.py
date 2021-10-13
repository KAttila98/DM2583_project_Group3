#!/usr/bin/env python
# coding: utf-8

# In[3]:


from sklearn.metrics import accuracy_score
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
import pandas as pd
import pickle5 as pickle
from thundersvm import SVC


# In[4]:


train_pickle = "../Movies_and_TV_5_balanced_tokenized.pkl"
evaluate_pickle = "../Sports_prep_tokenized.pkl"


# In[5]:


def train(df):
    df["reviewText"] = df["reviewText"].apply(lambda x: " ".join(x))
    print("Done joining the text")

    train_x, test_x, train_y, test_y = model_selection.train_test_split(df['reviewText'],df['label'],test_size=0.2)
    print("Done splitting")
    
    encoder = LabelEncoder()
    train_y = encoder.fit_transform(train_y)
    test_y = encoder.fit_transform(test_y)
    print("Done encoding")
    
    tfidf = TfidfVectorizer(max_features=4000)
    tfidf.fit(df['reviewText'])
    train_x_tfidf = tfidf.transform(train_x)
    test_x_tfidf = tfidf.transform(test_x)
    print("Done vectorizing")
    
    svm = SVC(C=100)
    svm.fit(train_x_tfidf,train_y)
    print("Done modeling")
    
    accuracy = svm.score(test_x, test_y)
    print("SVM Accuracy Score -> ", accuracy * 100, "%")

    return svm


# In[6]:


df_train = pd.read_pickle(train_pickle)


# In[ ]:


svm = train(df_train)


# In[ ]:


import pickle
with open("../sports-model.pickle", "wb") as handle:
    pickle.dump(svm, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[ ]:




