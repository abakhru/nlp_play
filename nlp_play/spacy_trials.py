#!/usr/bin/env python

# Install: pip install spacy && python -m spacy download en
import spacy

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load('en')

# Process whole documents
text = open('customer_feedback_627.txt').read()
doc = nlp(text)

# Find named entities, phrases and concepts
for entity in doc.ents:
    print(entity.text, entity.label_)

# Determine semantic similarities
doc1 = nlp('the fries were gross')
doc2 = nlp('worst fries ever')
doc1.similarity(doc2)

# Hook in your own deep learning models
nlp.add_pipe(load_my_model(), before='parser')
