import csv
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import wordnet as wn, stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from utils import *
from loader import *
from nltk import word_tokenize, WordNetLemmatizer

STOPWORDS = set(stopwords.words('english'))

dev_instances, test_instances, dev_key, test_key = get_instances()


def lemmatize_sentence(sentence):
    "Lemmatize sentences"
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(sentence)
    lemmatized_words = [lemmatizer.lemmatize(word.lower()) for word in words]
    return lemmatized_words


def highest_overlap(sentence1, sentence2):
    "Count highest overlap between 2 sentences"
    lemmatized1 = lemmatize_sentence(sentence1)
    lemmatized2 = lemmatize_sentence(sentence2)

    overlap_count = len(set(lemmatized1) & set(lemmatized2))
    return overlap_count


def find_best_sense(wsd):
    "Find best sense following the hybrid model described in report."
    word = wsd.lemma
    senses = wn.synsets(word)
    # weights = [1, 0.7, 0.6, 0.5, 0.4]
    weights = [1, 0.6, 0.5, 0.4, 0.3]

    max_overlap = 0
    best_sense = senses[0]
    for index, sense in enumerate(senses):
        definition = sense.definition()
        context = wsd.context

        overlap_count = highest_overlap(definition, context)
        # apply a penlaty to less frequent senses
        if index < 5:
            overlap_count *= weights[index]
        else:
            overlap_count *= 0.3

        if overlap_count > max_overlap:
            # ic('UPDATING SENSE')
            max_overlap = overlap_count
            best_sense = sense

    return best_sense


def third_method(test=True):
    "Main function call for the third method."
    # initiate counter vars
    count = 0

    # check for dev or test set
    if test:
        total = len(test_instances)
        true_keys = test_key
        instances = test_instances
    else:
        total = len(dev_instances)
        true_keys = dev_key
        instances = dev_instances

    for id, wsd in instances.items():
        best_sense = find_best_sense(wsd)

        true_senses = true_keys[id]
        true_senses = [wn.synset_from_sense_key(
            sense) for sense in true_senses]

        if best_sense in true_senses:
            count += 1

    accuracy = round(count/total, 2)
    print(f'Testing accuracy: {accuracy}')

    with open('report/third_method.csv', 'w', newline='') as file:
        csv_writer = csv.writer(file)
        header = ['Method', 'Accuracy']
        csv_writer.writerow(header)
        csv_writer.writerow(['Baseline + Lesk', accuracy])


if __name__ == '__main__':
    third_method()
