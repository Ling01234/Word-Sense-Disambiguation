from icecream import ic
from nltk.corpus import wordnet as wn
import os
from loader import get_instances
from datasets import load_dataset
from tqdm import trange

dev_instances, test_instances, dev_key, test_key = get_instances()


def write_dev():
    unique_dev = set()
    with open('dev_context.txt', 'w') as file:
        print('Writing in dev_context...')
        for wsd in dev_instances.values():
            context = wsd.context
            if context not in unique_dev:
                file.write(context + '\n')
                unique_dev.add(context)


def write_test():
    unique_test = set()
    with open('test_context.txt', 'w') as file:
        print('Writing in test_context...')
        for wsd in test_instances.values():
            context = wsd.context
            if context not in unique_test:
                file.write(context + '\n')
                unique_test.add(context)


# write_dev()
# write_test()


def get_dev_sentences():
    sentences = []

    with open('dev_context.txt', 'r') as file:
        for line in file:
            sentences.append(line.strip())
    return sentences


def get_test_sentences():
    sentences = []

    with open('test_context.txt', 'r') as file:
        for line in file:
            sentences.append(line.strip())
    return sentences


def convert_pos_to_wordnet(pos_tag):
    """
    Convert NLTK POS tags to WordNet POS tags.

    :param pos_tag: NLTK POS tag
    :return: Corresponding WordNet POS tag
    """
    if pos_tag.startswith('N'):
        return wn.NOUN
    elif pos_tag.startswith('V'):
        return wn.VERB
    elif pos_tag.startswith('R'):
        return wn.ADV
    elif pos_tag.startswith('J'):
        return wn.ADJ
    else:
        # If the POS tag is not recognized, default to NOUN
        return wn.NOUN


def find_most_common_words():
    """
    Find most common words between the dev set and testing set.
    """
    words = {}
    for id, wsd in test_instances.items():
        word = wsd.lemma

        if word not in words:
            words[word] = 1
        else:
            words[word] += 1

    sorted_words = sorted(words.items(), key=lambda x: x[1], reverse=True)
    return sorted_words[:5]  # take top 5


# ic(find_most_common_words())


def filter_by_word(word, instance='dev'):
    if instance == 'dev':
        instances = dev_instances
        key_instances = dev_key
    elif instance == 'test':
        instances = test_instances
        key_instances = test_key

    sentences = []
    if instance == 'dev' or instance == 'test':
        for id, wsd in instances.items():
            lemma = wsd.lemma

            if lemma == word:
                sense_keys = key_instances[id]
                # breakpoint()
                senses = [wn.synset_from_sense_key(
                    sense_key) for sense_key in sense_keys]
                context = wsd.context
                sentences.append((context, senses))

    else:  # specific text file
        with open(instance, 'r') as file:
            for line in file:
                if word in line:
                    sentences.append(line.strip())

    return sentences


# filter_by_word('year', 'test')


def get_sense_baseline(word):
    return wn.synsets(word)[0]


# build training data from ontonotesv5
dataset = load_dataset('conll2012_ontonotesv5', 'english_v12')


def build_training_data(sentences):
    """
    Gather training data from the ontonotes dataset and
    write the sentences into a txt file.

    Args:
        sentences (int): number of sentences to gather
    """
    with open('train.txt', 'w') as file, open('sample_train.txt', 'w') as sample_file:
        training_data = dataset['train']['sentences']
        for i in trange(sentences):  # extract training sentences
            sentence = ' '.join(training_data[i][0]['words']) + '\n'
            if i <= 40:
                sample_file.write(sentence)  # write in sample

            # write in training file
            file.write(sentence)


# build_training_data(800)
