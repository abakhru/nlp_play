#!/usr/bin/env python

"""
Here we are using ntlk and gensim libraries to classify twitter text
pip install gensim matplotlib wordcloud numpy pandas nltk xgboost
"""

import gensim
import nltk
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from wordcloud import WordCloud
from xgboost.sklearn import XGBClassifier

from nlp_libs import clean_data, tweets, label

nltk.download('corpora')  # this downloads the nltk packages
tqdm.pandas(desc="progress-bar")


def word_vector(tokens, size=200):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += model_w2v[word].reshape((1, size))
            count += 1.
        except KeyError:  # handling the case where the token is not in vocabulary
            continue
    if count != 0:
        vec /= count
    return vec


# ************  Exploratory data analysis *********************

# Generating word frequency
word_list = []
for tweet in tweets:
    for word in tweet.split():
        word_list.append(word)

word_freq = pd.Series(word_list).value_counts()
print("Top 20 words:")
print(word_freq[:20])

# Generate word cloud for visualizing the data
wc = WordCloud(width=400, height=330, max_words=100,
               background_color='white').generate_from_frequencies(word_freq)

plt.figure(figsize=(12, 10))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()

# from the word-cloud we see that there are a lot of unnecessary words like @user and &amp.
# Also there are stop-words like "the", "in", "to", "of" which do not hold any value
# We need to clean-up the data using the function clean_data defined above
tweets = tweets.apply(lambda x: clean_data(x))

label.hist()

# lets split the tweets into regular and racist tweets
reg_tweets = tweets[label == 0]
racist_tweets = tweets[label == 1]
# lets generate word clouds for regular and racist tweets
wc1 = WordCloud(width=800, height=500,
                random_state=21,
                max_font_size=110).generate("".join(tweet for tweet in reg_tweets))
plt.figure(figsize=(10, 7))
plt.imshow(wc1, interpolation="bilinear")
plt.axis('off')
plt.show()
wc2 = WordCloud(width=800, height=500,
                random_state=21,
                max_font_size=110).generate("".join(tweet for tweet in racist_tweets))
plt.figure(figsize=(10, 7))
plt.imshow(wc2, interpolation="bilinear")
plt.axis('off')
plt.show()

# ******************* Feature engineering ************************

# Now we have to tokenize the tweets, ie. split each tweet into a list of words
tweets = tweets.apply(lambda x: word_tokenize(x))

# Now we normalize the words using lemmatization
lemm = WordNetLemmatizer()
tweets = tweets.apply(lambda tweet: [lemm.lemmatize(word) for word in tweet])

# plot the count of no. of words in a tweet
len_tw = pd.Series([len(tweet) for tweet in tweets])
len_tw.hist(bins=20)
# we see that the max no. of words in a tweet is 17, and most of them are around 8

# Now we have to vectorize the words into embeddings.

# We will use Word2Vec
model_w2v = gensim.models.Word2Vec(
    tweets,
    size=200,  # desired no. of features/independent variables
    window=5,  # context window size
    min_count=2,
    sg=1,  # 1 for skip-gram model
    hs=0,
    negative=10,  # for negative sampling
    workers=2,  # no.of cores
    seed=34)

model_w2v.train(tweets, total_examples=len(tweets), epochs=20)

# since each word is represented in 200 dimensions by word2vec,
# it will result in very large data for each tweet
# to avoid this, we convert each word in the tweet to a vector and then take its average.
# This way each tweet will have only 200 dimensions

# Preparing word2vec feature set, which will be a nx200 matrix
wordvec_arrays = np.zeros((len(tweets), 200))
for i in range(len(tweets)):
    wordvec_arrays[i, :] = word_vector(tweets[i], 200)

# convert the matrix to a features dataframe
wordvec_df = pd.DataFrame(wordvec_arrays)
print(wordvec_df.shape)

# *******************  Build and test the XGBoost model with word2vec ****************************

wordvec_df_train, wordvec_df_test, label_train, label_test = train_test_split(wordvec_df, label,
                                                                              test_size=0.3,
                                                                              random_state=100,
                                                                              stratify=label)

# Since the data is highly skewed
SCALE_FACTOR = label.value_counts()[0] / label.value_counts()[1]

xgb1 = XGBClassifier(max_depth=6, n_estimators=1000, scale_pos_weight=SCALE_FACTOR)

print("Training w2v....")
xgb1.fit(wordvec_df_train, label_train)
print("Score=", xgb1.score(wordvec_df_test, label_test))
pred_labels = xgb1.predict(wordvec_df_test)
print("F1 score=", f1_score(label_test, pred_labels))
print("ROC AUC score = ", roc_auc_score(label_test, pred_labels))

# we get an f1 score of 0.68
