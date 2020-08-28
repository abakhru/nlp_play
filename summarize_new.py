#!/usr/bin/env python

"""
pip install newspaper3k
- https://github.com/codelucas/newspaper
"""
import newspaper

from nlp_play import LOGGER


class NewsSummary:

    def __init__(self, url):
        self.url = url
        self.summary = None

    def summarize(self):
        article = newspaper.Article(self.url)
        article.download()
        article.parse()
        LOGGER.info(f'Authors: {article.authors}')
        article.nlp()
        self.summary = article.summary
        LOGGER.info({self.summary})


if __name__ == '__main__':
    p = NewsSummary(url='https://www.cnn.com/2020/08/25/asia/india-university-virtual-reality-graduation-scli-intl/index.html')
    p.summarize()
