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
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity


# In[2]:


word_vectors = KeyedVectors.load_word2vec_format('cc.en.300.vec', binary=False)


# In[3]:


lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


# In[5]:


# Function to calculate the completeness score for a review
def completeness_score(row, critical_columns):
    total_columns = len(critical_columns)  # Total number of critical columns
    filled_columns = row[critical_columns].notnull().sum()  # Count of non-null values in critical columns
    completeness_percentage = filled_columns / total_columns  # Proportion of filled columns
    score = 10 * completeness_percentage  # Scale the score to a maximum of 10
    return score


# In[6]:


# Function to assign completeness scores to all reviews in the DataFrame
def assign_completeness_score(df, critical_columns):
    # Apply the completeness_score function to each row in the DataFrame
    df['Completeness_Score'] = df.apply(completeness_score, axis=1, args=(critical_columns,))
    return df


# In[7]:


# Function to calculate the timeliness score for a review based on its date
def timeliness_score(review_date, latest_date, earliest_date):
    if pd.isna(review_date):
        return 0  # Return a score of 0 if the review date is missing
    total_days = (latest_date - earliest_date).days  # Total days in the review period
    days_difference = (latest_date - review_date).days  # Days between the latest review and the current review
    # Scale the score to a maximum of 10 based on the recency of the review
    score = 10 * (1 - days_difference / total_days) if total_days > 0 else 0
    return score


# In[8]:


# Function to assign timeliness scores to all reviews in the DataFrame
def assign_timeliness_score(df):
    latest_date = df['date'].max()  # Find the latest review date
    earliest_date = df['date'].min()  # Find the earliest review date
    # Apply the timeliness_score function to each review date in the DataFrame
    df['Timeliness_Score'] = df['date'].apply(timeliness_score, args=(latest_date, earliest_date))
    return df


# In[9]:


# Function to calculate the validity score based on whether the purchase is verified
def validity_score(is_verified):
    """
    Calculate the validity score based on the 'verified' value.
    - 10 points if the purchase is verified (verified == True)
    - 0 points if the purchase is not verified or data is missing (is_verified != True or NaN)
    """
    validity_score = 10 if is_verified == True else 0
    return validity_score


# In[10]:


# Function to assign validity scores to all reviews in the DataFrame
def assign_validity_score(df):
    # Apply the validity_score function to the 'verified' column in the DataFrame
    df['Validity_Score'] = df['verified'].apply(validity_score)
    return df


# In[11]:


# Function to assign uniqueness scores based on whether the review is a duplicate
def assign_uniqueness_score(df, key_columns):
    # Identify duplicate rows based on key columns
    is_duplicate = df.duplicated(subset=key_columns, keep=False)
    # Assign a score of 10 for unique reviews and 0 for duplicates
    df['Uniqueness_Score'] = 10 * (~is_duplicate).astype(int)
    return df


# In[15]:


# Function to preprocess text by cleaning and tokenizing
def preprocess_text(text):
    if pd.isna(text):
        return ""  # Return an empty string if the text is missing
    text = text.lower()  # Convert text to lowercase
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)  # Tokenize the text
    # Lemmatize tokens and remove stopwords
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return tokens


# In[16]:


# Function to calculate cosine similarity between specification terms and review tokens using word vectors
def get_similarity(spec, token, word_vectors):
    try:
        # Calculate and return the cosine similarity between the vectors of the specification term and the review token
        return cosine_similarity([word_vectors[spec]], [word_vectors[token]])[0][0]
    except KeyError:
        # Return 0 if either the spec or token is not found in the word_vectors
        return 0


# In[17]:


# Function to find similar terms in a review based on a list of specifications
def find_similar_terms(specifications, review_tokens, word_vectors, threshold=0.7):
    similar_terms = {}  # Dictionary to hold the specification terms and their similar tokens
    for spec in specifications:
        similar_terms[spec] = []  # Initialize a list for each specification term
        for token in review_tokens:
            similarity = get_similarity(spec, token, word_vectors)  # Get similarity score
            if similarity > threshold:  # Check if similarity exceeds the threshold
                similar_terms[spec].append((token, similarity))  # Append the token and similarity score
                print(f"Match found: Spec='{spec}', Token='{token}', Similarity={similarity}")
    # Remove specification terms with no matching tokens
    similar_terms = {k: v for k, v in similar_terms.items() if v}
    return similar_terms


# In[18]:


# Function to apply find_similar_terms across a DataFrame of reviews
def find_similar_terms_in_reviews(df, specifications, word_vectors, threshold=0.7):
    # Apply the find_similar_terms function to each tokenized review and store the results in a new column
    df['similar_terms'] = df['tokenized_body'].apply(find_similar_terms, args=(specifications, word_vectors, threshold))
    return df


# In[19]:


# Function to extract unique specification terms from the similar terms found in a review
def extract_terms(similar_terms):
    if not similar_terms:
        return []
    terms = set()  # Use a set to automatically remove duplicates
    for key, value in similar_terms.items():
        terms.add(key)  # Add each specification term to the set
    return list(terms)


# In[20]:


# Function to calculate the accuracy score for a review based on the number of relevant terms it contains
def score_accuracy(row, mean_num_terms, scale_factor=15, max_score=30):
    num_terms = len(row['terms'])  # Count the number of unique relevant terms in the review
    # Scale the score based on the average number of terms and the scale factor, with a maximum limit
    score = min((num_terms / mean_num_terms) * scale_factor, max_score)
    return score


# In[21]:


# Function to assign accuracy scores to all reviews in the DataFrame
def assign_accuracy_score(df):
    df['terms'] = df['similar_terms'].apply(extract_terms)  # Extract relevant terms from similar terms
    mean_num_terms = df['terms'].apply(len).mean()  # Calculate the mean number of relevant terms across all reviews
    # Apply the score_accuracy function to each review to calculate its accuracy score
    df['Accuracy_Score'] = df.apply(score_accuracy, axis=1, args=(mean_num_terms,))
    return df

