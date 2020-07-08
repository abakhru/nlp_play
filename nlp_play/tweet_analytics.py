#!env python

""" pip install vaex xlrd twint wordcloud nltk spacy pandas"""

import re
import pandas as pd
import vaex
import numpy as np

from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import spacy
import nltk

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt


class TweetAnalytics:

    def __init__(self):
        self.df = pd.read_csv('trump.xls')[['date', 'time', 'tweet']]
        self.stemmer = PorterStemmer()
        # nlp = spacy.load('en_core_web_md')
        # self.stop_words = spacy.lang.en.stop_words.STOP_WORDS

        self.clean_tweets()

    @staticmethod
    def get_hashtags(text):
        hashtags = re.findall(r'\#\w+', text.lower())
        return hashtags

    @staticmethod
    def get_mentions(text):
        mentions = re.findall(r'\@\w+', text.lower())
        return mentions

    def clean_tweets(self):
        self.df['cleaned_tweets'] = self.df['tweet'].apply(lambda x: self.process_text(x))
        self.df['tweet'] = self.df['tweet'].apply(lambda x: self.remove_content(x))

    @staticmethod
    def remove_content(text):
        text = re.sub(r"http\S+", "", text)  # remove urls
        text = re.sub(r'\S+\.com\S+', '', text)  # remove urls
        text = re.sub(r'\@\w+', '', text)  # remove mentions
        text = re.sub(r'\#\w+', '', text)  # remove hashtags
        return text

    def process_text(self, text, stem=False):  # clean text
        text = remove_content(text)
        text = re.sub('[^A-Za-z]', ' ', text.lower())  # remove non-alphabets
        tokenized_text = word_tokenize(text)  # tokenize
        clean_text = [
            word for word in tokenized_text
            if word not in self.stop_words
            ]
        if stem:
            clean_text = [self.stemmer.stem(word) for word in clean_text]
        return ' '.join(clean_text)

    def word_cloud(self):
        temp = ' '.join(self.df['cleaned_tweets'].tolist())
        wordcloud = WordCloud(width=800, height=500,
                              background_color='white',
                              min_font_size=10).generate(temp)
        plt.figure(figsize=(8, 8), facecolor=None)
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.show()


def try_vaex():
    """Huge file => faster reads than pandas"""
    file_path = 'data/big_file.csv'
    # dv = vaex.from_csv(file_path, convert=True, chunk_size=5_000_000)
    # print(type(dv))
    dv = vaex.open('data/big_file.csv.hdf5')
    dv.plot1d(dv.col2, figsize=(14, 7))


def create_fle():
    n_rows = 100000
    n_cols = 1000
    df = pd.DataFrame(np.random.randint(0, 100, size=(n_rows, n_cols)),
                      columns=['col%d' % i for i in range(n_cols)])
    df.head()
    df.info(memory_usage='deep') # memory used by df
    file_path = 'data/big_file.csv'
    df.to_csv(file_path, index=False)


if __name__ == '__main__':
    # p = TweetAnalytics()
    # p.word_cloud()
    try_vaex()
