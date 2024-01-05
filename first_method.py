import csv
from nltk.tag import pos_tag
from nltk.corpus import wordnet as wn, stopwords
from nltk.stem import WordNetLemmatizer
from icecream import ic
from loader import get_instances
from nltk.wsd import lesk
from nltk.tokenize import word_tokenize
from utils import *


STOPWORDS = set(stopwords.words('english'))
dev_instances, test_instances, dev_key, test_key = get_instances()


def first_method(test=True):
    # initiate counter vars
    counter_baseline = 0
    counter_lesk = 0

    # check for dev or test set
    if test:
        total = len(test_instances)
        true_keys = test_key
        instances = test_instances
    else:
        total = len(dev_instances)
        true_keys = dev_key
        instances = dev_instances

    lemmatizer = WordNetLemmatizer()
    for id, wsd in instances.items():
        word = lemmatizer.lemmatize(wsd.lemma)

        # get sense keys
        keys = true_keys[id]
        sense_key = [wn.synset_from_sense_key(key) for key in keys]
        # ic(sense_key)

        # baseline method
        sense1 = get_sense_baseline(word)
        if sense1 in sense_key:
            counter_baseline += 1

        # lesk algorithm
        context = word_tokenize(wsd.context)
        pos_tags = pos_tag(context)
        for token, tag in pos_tags:
            if token == word:
                pos = convert_pos_to_wordnet(tag)
                break
        # ic(wsd.context)

        sense2 = lesk(context, word, pos)
        if not sense2:
            sense2 = lesk(context, word)

        # ic(word, pos, sense2)
        if sense2 in sense_key:
            counter_lesk += 1

    baseline_accuracy = round(counter_baseline/total, 2)
    lesk_accuracy = round(counter_lesk/total, 2)
    print(f'Baseline Method: {baseline_accuracy}')
    print(f'Lesk Method: {lesk_accuracy}')

    with open('report/first_method.csv', 'w', newline='') as file:
        csv_writer = csv.writer(file)
        header = ['Method', 'Accuracy']
        csv_writer.writerow(header)
        csv_writer.writerow(['Baseline Method', baseline_accuracy])
        csv_writer.writerow(['Lesk Method', lesk_accuracy])


if __name__ == '__main__':
    first_method()
