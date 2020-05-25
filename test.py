#!/usr/bin/env python
# coding: utf8
"""Example of training an additional entity type

This script shows how to add a new entity type to an existing pretrained NER
model. To keep the example short and simple, only four sentences are provided
as examples. In practice, you'll need many more — a few hundred would be a
good start. You will also likely need to mix in examples of other entity
types, which might be obtained by running the entity recognizer over unlabelled
sentences, and adding their annotations to the training set.

The actual training is performed by looping over the examples, and calling
`nlp.entity.update()`. The `update()` method steps through the words of the
input. At each word, it makes a prediction. It then consults the annotations
provided on the GoldParse instance, to see whether it was right. If it was
wrong, it adjusts its weights so that the correct action will score higher
next time.

After training your model, you can save it to a directory. We recommend
wrapping models as Python packages, for ease of deployment.

For more details, see the documentation:
* Training: https://spacy.io/usage/training
* NER: https://spacy.io/usage/linguistic-features#named-entities

Compatible with: spaCy v2.1.0+
Last tested with: v2.1.0
"""
from __future__ import unicode_literals, print_function

# import plac
import random
from pathlib import Path
import spacy
import config
from spacy.util import minibatch, compounding,decaying
from spacy.vocab import Vocab
from spacy.tokens import Doc
import thinc_gpu_ops

spacy.require_gpu()


# new entity label
LABEL = "flavour"

# training data
# Note: If you're using an existing model, make sure to mix in examples of
# other entity types that spaCy correctly recognized before. Otherwise, your
# model might learn the new type, but "forget" what it previously knew.

TRAIN_DATA = config.train
# print (TRAIN_DATA)

def train_spacy(data, iterations):
    TRAIN_DATA = data
    nlp = spacy.blank('en')  # create blank Language class
    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)

    # add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    ner.add_label("Flavour")
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training(use_device=0)

        #BATCH SIZE
        batch_size = compounding(1, config.general['batch_size'], 1.001)
        # batch up the examples using spaCy's minibatch
        dropout = decaying(0.4, 0.2, 0.005)
        for itn in range(iterations):
            print("Statring iteration " + str(itn))
            drop = next(dropout)
            print("drop", drop)

            random.shuffle(TRAIN_DATA)
            losses = {}
            batches = minibatch(TRAIN_DATA, size=batch_size)
            for batch in batches:
                # print("bathc",batch)
                text, annotations = zip(*batch)

                nlp.update(
                    text,  # batch of texts
                    annotations,  # batch of annotations
                    drop=drop,  # dropout - make it harder to memorise data
                    sgd=optimizer,  # callable to update weights
                    losses=losses)
            print("losses",losses)

        # for itn in range(iterations):
        #     print("Statring iteration " + str(itn))
        #     random.shuffle(TRAIN_DATA)
        #     losses = {}
        #     drop = next(dropout)
        #     print("drop",drop)
        #     for text, annotations in TRAIN_DATA:
        #         nlp.update(
        #             [text],  # batch of texts
        #             [annotations],  # batch of annotations
        #             drop= drop,  # dropout - make it harder to memorise data
        #             sgd=optimizer,  # callable to update weights
        #             losses=losses)
        #     print(losses)
    with nlp.use_params(optimizer.averages):
        modelfile = input("Enter your Model Name: ")
        nlp.to_disk(modelfile)
    return nlp

"""codeee to steal"""
nlp = train_spacy(TRAIN_DATA, 100)

# nlp = spacy.load('model')

# Save our trained Model
# modelfile = input("Enter your Model Name: ")
# with nlp.use_params(optimizer.averages):
#     nlp.to_disk("/model")

# Test your text
test_text = input("Enter your testing text: ")
doc = nlp(test_text)
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)
test_text = input("Enter your testing text: ")
doc = nlp(test_text)
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)
test_text = input("Enter your testing text: ")
doc = nlp(test_text)
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)
