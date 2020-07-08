#!/usr/bin/env python

"""
pip install twitter-nlp-toolkit spacy
python -m spacy download en_core_web_sm

- https://github.com/eschibli/twitter-toolbox
- https://towardsdatascience.com/simple-twitter-analytics-with-twitter-nlp-toolkit-7d7d79bf2535
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from twitter_nlp_toolkit.tweet_json_parser import tweet_json_parser
from twitter_nlp_toolkit.tweet_sentiment_classifier import tweet_sentiment_classifier
from twitter_nlp_toolkit.twitter_REST_downloader import twitter_REST_downloader
from twitter_nlp_toolkit.twitter_listener import twitter_listener

from nlp_play import LOGGER


class TwitterNLPToolKitExperiments:
    """Sentiment analysis using NLP and tweeter-toolkit"""

    def __init__(self):
        self.data_dir = Path(__file__).parent.parent.resolve().joinpath('data')
        self.credentials = json.load(self.data_dir.parent.joinpath('twitter_keys.json').open())
        # each member of the list is a separate search condition
        self.target_words = ['Musk', 'Tesla']
        self.json_output_file_name = f'{self.target_words[0]}_tweets.json'
        self.csv_parsed_path = (self.data_dir / f'parsed_'
                                                f'{self.json_output_file_name}').with_suffix('.csv')

    def stream_realtime_tweets(self):
        """Streams real-time tweets for target words"""
        out = self.data_dir / self.json_output_file_name
        stream1 = twitter_listener.TwitterStreamListener(**self.credentials)
        stream1.collect_from_stream(max_tweets=1000,
                                    output_json_name=f"{out}",
                                    console_interval=2,
                                    target_words=self.target_words)

    def json_to_csv_parse(self):
        LOGGER.info("convert json to csv")
        parser = tweet_json_parser.json_parser()
        parser.stream_json_file(json_file_name=f"{self.data_dir}"
                                               f"/{self.json_output_file_name}",
                                output_file_name=f"{self.csv_parsed_path}",
                                verbose=True)

    def bulk_download(self):
        csv_file_path = (self.data_dir / self.json_output_file_name).with_suffix('.csv')
        downloader = twitter_REST_downloader.bulk_downloader(**self.credentials)
        downloader.get_tweets_csv_for_this_user("@elonmusk", f"{csv_file_path}")

    def sentiment_analysis(self):
        LOGGER.info("sentiment analysis")
        classifier = tweet_sentiment_classifier.SentimentAnalyzer(model_path=f'{self.data_dir}')
        classifier.load_large_ensemble()
        # other option is load_small_ensemble()
        tweets = pd.read_csv(self.csv_parsed_path)
        sentiment = classifier.predict_proba(tweets['text'])
        sentiment_labels = classifier.predict(tweets['text'])
        tweets['sentiment'] = sentiment
        tweets['sentiment_labels'] = sentiment_labels
        tweets.to_csv(self.csv_parsed_path.parent.joinpath(f'parsed_classified_'
                                                           f'{self.target_words[0]}_tweets.csv'))

    def plot_sentiment(self):
        df = pd.read_csv(self.csv_parsed_path.parent.joinpath(f'parsed_classified_'
                                                              f'{self.target_words[0]}_tweets.csv'))
        LOGGER.debug(f'Columns: {df.columns}')
        df['sentiment'].plot(kind='hist', x='created_at')
        plt.show()


if __name__ == '__main__':
    p = TwitterNLPToolKitExperiments()
    p.stream_realtime_tweets()
    p.json_to_csv_parse()
    p.bulk_download()
    p.sentiment_analysis()
    p.plot_sentiment()
