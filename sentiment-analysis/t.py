#!/usr/bin/env python


from twitterscraper import query_tweets

if __name__ == '__main__':
    list_of_tweets = query_tweets("Trump OR Clinton", 10, poolsize=1)

    # Or save the retrieved tweets to file:
    file = open('output.txt', 'w')
    for tweet in list_of_tweets:
        print(tweet.text)
        file.write(str(tweet.text.encode('utf-8')))
    file.close()
