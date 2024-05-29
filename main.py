#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import json
from score_calculation_functions import preprocess_text, word_vectors, find_similar_terms_in_reviews, assign_accuracy_score, assign_timeliness_score, assign_completeness_score, assign_uniqueness_score, assign_validity_score, assign_consistency_score, calculate_total_quality_score


# In[2]:


df = pd.read_csv('english_reviews.csv', parse_dates=['date'])


# In[25]:


df


# In[4]:


df['date'].info()


# In[5]:


df['tokenized_body'] = df['body'].apply(preprocess_text)


# In[6]:


with open('cell_phone_specifications.json', 'r') as f:
    specifications = json.load(f)


# In[7]:


print(specifications)


# In[8]:


df = find_similar_terms_in_reviews(df, specifications, word_vectors)


# In[9]:


df = assign_accuracy_score(df)


# In[10]:


df = assign_validity_score(df)


# In[11]:


critical_columns = ['name', 'rating', 'date', 'verified', 'title', 'body']


# In[12]:


df = assign_completeness_score(df, critical_columns)


# In[13]:


df = assign_timeliness_score(df)


# In[14]:


df = assign_uniqueness_score(df, critical_columns)


# In[15]:


df = assign_consistency_score(df)


# In[16]:


df


# In[17]:


score_columns = ['Completeness_Score', 'Validity_Score', 'Timeliness_Score', 'Uniqueness_Score', 'Consistency_Score', 'Accuracy_Score']


# In[18]:


df = calculate_total_quality_score(df, score_columns)


# In[19]:


df


# In[20]:


import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Initialize the StandardScaler
scaler = StandardScaler()

# Standardize the 'Total_Score' and 'helpfulVotes' columns
df[['Total_Score_Standardized', 'helpfulVotes_Standardized']] = scaler.fit_transform(df[['Total_Quality_Score', 'helpfulVotes']])


# In[21]:


# Calculate the Pearson correlation coefficient
correlation_pearson = df['helpfulVotes'].corr(df['Total_Quality_Score'])
print("Pearson correlation coefficient:", correlation_pearson)

# Calculate the Spearman correlation coefficient
correlation_spearman = df['helpfulVotes'].corr(df['Total_Quality_Score'], method='spearman')
print("Spearman correlation coefficient:", correlation_spearman)

# Calculate the Kendall correlation coefficient
correlation_kendall = df['helpfulVotes'].corr(df['Total_Quality_Score'], method='kendall')
print("Kendall correlation coefficient:", correlation_kendall)


# In[22]:


# Scatter plot with standardized data
plt.figure(figsize=(10, 6))
plt.scatter(df['Total_Score_Standardized'], df['helpfulVotes_Standardized'], alpha=0.5)
plt.title('Scatter Plot of Standardized Total Score vs. Standardized Helpfulness Votes')
plt.xlabel('Standardized Total Score')
plt.ylabel('Standardized Helpfulness Votes')
plt.show()


# In[23]:


df


# In[24]:


df.to_csv('df_result.csv', index=False)

