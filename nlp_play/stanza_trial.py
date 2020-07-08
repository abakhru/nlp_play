#!/usr/bin/env python

"""https://github.com/stanfordnlp/stanza"""
import os

import stanza
from nlp_play import LOGGER

LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')


def english_exp():
    # This downloads the English models for the neural pipeline
    stanza.download('en', logging_level=LOG_LEVEL)
    # This sets up a default neural pipeline in English
    nlp = stanza.Pipeline('en', logging_level=LOG_LEVEL)
    doc = nlp("Barack Obama was born in Hawaii.  He was elected president in 2008.")
    doc.sentences[0].print_dependencies()


def print_document_details(doc):
    LOGGER.info('the text, lemma and POS tag of each word in '
                'each sentence of an annotated document')
    for sentence in doc.sentences:
        for word in sentence.words:
            print(word.text, word.lemma, word.pos)
    LOGGER.info('all named entities and dependencies in a document:')
    for sentence in doc.sentences:
        print(sentence.ents)
        print(sentence.dependencies)


def hindi_trial():
    stanza.download('hi', logging_level=LOG_LEVEL)
    nlp = stanza.Pipeline('hi', logging_level=LOG_LEVEL)
    hindi_text = ('भोपाल. मुख्यमंत्री ने की मरीजों से बातचीत')
    doc = nlp(hindi_text)
    doc.sentences[0].print_dependencies()
    print_document_details(doc)


if __name__ == '__main__':
    hindi_trial()
    # english_exp()
