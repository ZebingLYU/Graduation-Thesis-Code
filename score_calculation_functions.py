#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity


# In[2]:


word_vectors = KeyedVectors.load_word2vec_format('cc.en.300.vec', binary=False)


# In[3]:


lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


# In[4]:


sia = SentimentIntensityAnalyzer()


# In[5]:


def completeness_score(row, critical_columns):
    total_columns = len(critical_columns)
    filled_columns = row[critical_columns].notnull().sum()
    completeness_percentage = filled_columns / total_columns
    score = 10 * completeness_percentage
    return score


# In[6]:


def assign_completeness_score(df, critical_columns):
    df['Completeness_Score'] = df.apply(completeness_score, axis=1, args=(critical_columns,))
    return df


# In[7]:


def timeliness_score(review_date, latest_date, earliest_date):
    if pd.isna(review_date):
        return 0
    total_days = (latest_date - earliest_date).days
    days_difference = (latest_date - review_date).days
    score = 10 * (1 - days_difference / total_days) if total_days > 0 else 0
    return score


# In[8]:


def assign_timeliness_score(df):
    latest_date = df['date'].max()
    earliest_date = df['date'].min()
    df['Timeliness_Score'] = df['date'].apply(timeliness_score, args=(latest_date, earliest_date))
    return df


# In[9]:


def validity_score(is_verified):
    """
    Calculate the validity score based on the 'verified' value.
    - 15 points if the purchase is verified (verified == True)
    - 0 points if the purchase is not verified or data is missing (is_verified != True or NaN)
    """
    validitity_score = 15 if is_verified == True else 0
    return validitity_score


# In[10]:


def assign_validity_score(df):
    df['Validity_Score'] = df['verified'].apply(validity_score)
    return df


# In[11]:


def assign_uniqueness_score(df, key_columns):
    is_duplicate = df.duplicated(subset=key_columns, keep=False)
    df['Uniqueness_Score'] = 10 * (~is_duplicate).astype(int)
    return df


# In[12]:


def normalize_rating(rating, scale_max):
    """ Normalize ratings to a [-1, 1] scale. Assuming ratings are from 1 to scale_max. """
    return 2 * ((rating - 1) / (scale_max - 1)) - 1


# In[13]:


def score_consistency(text, rating, scale_max=5):
    """ Calculate the consistency between VADER sentiment and normalized user ratings. """
    # Check for null in either text or rating
    if pd.isnull(text) or pd.isnull(rating):
        return 0
    
    # Get the sentiment score
    sentiment_score = sia.polarity_scores(text)['compound']  # Compound score is sufficient for most cases
    
    # Normalize the user rating
    normalized_rating = normalize_rating(rating, scale_max)
    
    # Calculate consistency as the inverse of the absolute difference, scaled to a score out of 20
    consistency = (1 - abs(normalized_rating - sentiment_score)) * 25
    consistency_score = max(0, consistency)  # Ensure consistency does not go below 0
    
    return (consistency_score, sentiment_score) 


# In[14]:


def assign_consistency_score(df):
    """
    Assign consistency scores and sentiment scores to a DataFrame.
    """
    results_df = df.apply(lambda row: score_consistency(row['body'], row['rating']), axis=1, result_type='expand')
    df['Consistency_Score'], df['Sentiment_Score'] = results_df[0], results_df[1]
    return df


# In[15]:


def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return tokens


# In[16]:


def get_similarity(spec, token, word_vectors):
    try:
        return cosine_similarity([word_vectors[spec]], [word_vectors[token]])[0][0]
    except KeyError:
        return 0


# In[17]:


def find_similar_terms(specifications, review_tokens, word_vectors, threshold=0.7):
    similar_terms = {}
    for spec in specifications:
        similar_terms[spec] = []
        for token in review_tokens:
            similarity = get_similarity(spec, token, word_vectors)
            if similarity > threshold:
                similar_terms[spec].append((token, similarity))
                print(f"Match found: Spec='{spec}', Token='{token}', Similarity={similarity}")
    # Filter out empty lists
    similar_terms = {k: v for k, v in similar_terms.items() if v}
    return similar_terms


# In[18]:


def find_similar_terms_in_reviews(df, specifications, word_vectors, threshold=0.7):
    df['similar_terms'] = df['tokenized_body'].apply(find_similar_terms, args=(specifications, word_vectors, threshold))
    return df


# In[19]:


def extract_terms(similar_terms):
    if not similar_terms:
        return []
    terms = set()  # Use a set to automatically remove duplicates
    for key, value in similar_terms.items():
        terms.add(key)
    return list(terms)


# In[20]:


def score_accuracy(row, mean_num_terms, scale_factor=15, max_score=30):
    num_terms = len(row['terms'])
    score = min((num_terms / mean_num_terms) * scale_factor, max_score)
    return score


# In[21]:


def assign_accuracy_score(df):
    df['terms'] = df['similar_terms'].apply(extract_terms)
    mean_num_terms = df['terms'].apply(len).mean()
    df['Accuracy_Score'] = df.apply(score_accuracy, axis=1, args=(mean_num_terms,))
    return df 


# In[22]:


def calculate_total_quality_score(df, score_columns):
    """
    Calculate the total data quality score by summing up individual dimension scores.
    """
    df['Total_Quality_Score'] = df[score_columns].sum(axis=1)
    return df

