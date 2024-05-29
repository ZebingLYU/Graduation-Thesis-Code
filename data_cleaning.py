#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException


# In[2]:


df = pd.read_csv('cell_phone_datasets/reviews.csv')


# In[5]:


# Function to detect language
def is_english(text):
    try:
        # Return True if the detected language is English
        return detect(text) == 'en'
    except LangDetectException:
        # If detection fails (e.g., due to short text), consider it as not English
        return False


# In[6]:


# Apply the language detection function and keep rows where the body is English or NaN
df['is_english'] = df['body'].apply(lambda x: is_english(x) if pd.notnull(x) else True)


# In[7]:


# Filter the DataFrame to keep only English rows
df_english = df[df['is_english']].drop(columns=['is_english'])


# In[9]:


df_english['date'] = pd.to_datetime(df_english['date'])


# In[11]:


df_english['helpfulVotes'] = df_english['helpfulVotes'].fillna(0)


# In[14]:


df_english.to_csv('english_reviews.csv', index=False)

