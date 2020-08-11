import re
import pandas as pd
from gensim.parsing import remove_stopwords


def clean_data(text):
    text = re.sub('@[\w]*', '', text)  # remove @user
    text = re.sub('&amp;', '', text)  # remove &amp;
    text = re.sub('[?!.;:,,#@-]', '', text)  # remove special characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # remove Unicode characters
    text = text.replace("[^A-Za-z#]", "")  # Replace everything except alphabets and hash
    text = text.lower()  # make everything lowercase for uniformity
    # removing short words which are of length 3 or lower(eg. hmm, oh) since they dont add any value
    text = " ".join(w for w in text.split() if len(w) > 3)
    # removing stop-words eg. 'we', 'our', 'ours', 'ourselves', 'just', 'don', "don't", 'should'
    text = remove_stopwords(text)
    return text

# *************************  read training data ********************************
# df = pd.read_csv('data//train_tweets.csv')
df = pd.read_csv('data/parsed_classified_Musk_tweets.csv')
print(df.head())

df.drop('id', axis=1, inplace=True)
df.drop_duplicates()
print(df.isna().sum())

tweets = df['text']
label = df['sentiment_labels']
