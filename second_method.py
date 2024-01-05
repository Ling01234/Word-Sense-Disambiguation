import csv
from sklearn.feature_extraction.text import CountVectorizer
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from utils import *


game_seed = {wn.synset('plot.n.1'): ['they concocted a plot to discredit the governor', 'I saw through his little game from the start'],
             wn.synset('game.n.02'):  ['the game lasted two hours', 'a good game of golf', 'what a great game'],
             }

year_seed = {
    wn.synset('year.n.01'): ['she is 4 years old',
                             'in the year 1920',
                             'during the year',
                             "The Earth completes one orbit around the sun in approximately 365 days, defining a solar year.",
                             "Their friendship grew stronger with each passing year, marked by shared experiences and memories.",
                             "The company celebrated its tenth anniversary this year, reflecting on a decade of achievements and progress."],
    wn.synset('year.n.02'): ["Leap years have 366 days.",
                             'a school year',
                             "Lunar new year marks new beginnings."]
}

player_seed = {
    wn.synset('player.n.01'): ["It's time now to look at professional soccer and rugby and how it can be bad for a player's health."],
    wn.synset('player.n.05'): ['He was a major player in setting up the corporation.']
}

team_seed = {
    wn.synset('team.n.01'): ["It's been four years since the U.S. women's soccer team made history winning the World Cup before more than 90,000 fans at the Rose Bowl.",
                             "Bi Li, one of the coach patriarchs of the National Women's Football Team, revealed the recent '3 step melody' of the national women's football team to journalists a few days ago in the Wuhu arena of the National Women's Football League Competition."],
    wn.synset('team.v.01'): ['We teamed up for this new project',
                             'The students decided to team up for the group project, combining their skills to tackle the challenging assignment together.']
}

case_seed = {
    wn.synset('lawsuit.n.01'): ['The family brought suit against the landlord',
                                'Judges hearing the case against two men accused in the bombing of Pan Am Flight 103 are expected to announce this week when they will reach a verdict.'],
    wn.synset('case.n.01'): ['It was a case of bad judgment',
                             'And the growing number of SARS cases in China.']
}

SEED_SET = {
    'game': game_seed,
    'year': year_seed,
    'player': player_seed,
    'team': team_seed,
    'case': case_seed
}
ITERS = 10


class Model:
    def __init__(self, word, start_seed, training_iters=10):
        self.word = word
        self.classifier = None

        # sentences from dev set
        try:
            self.data = filter_by_word(self.word, instance='train.txt')
        except:
            self.data = filter_by_word(self.word, instance='sample_train.txt')
        self.test_set = filter_by_word(self.word, instance='test')
        self.training_iters = training_iters

        # list of tuples (sentence, sense)
        self.seed = self.create_dataset(start_seed)

    def create_dataset(self, start_seed):
        # Create initial dataset given the start seed
        data = []
        for sense, sentences in start_seed.items():
            for sentence in sentences:
                data.append((sentence, sense))

        return data

    def preprocess_text(self, text):
        "Preprocess text in format to pass to Naive Bayes"
        features = {}
        words = word_tokenize(text.lower())
        for word in words:
            if word.isalpha() and word not in stopwords.words('english'):
                features[word] = True
        return features

    def build_classifier(self):
        # Preprocess the text data
        featureset = [(self.preprocess_text(sentence), sense)
                      for sentence, sense in self.seed]

        # Train classifier
        self.classifier = NaiveBayesClassifier.train(featureset)

    def train(self):
        "Train using yarowsky's algorithm"
        self.build_classifier()
        for i in range(1, self.training_iters+1):
            # print(f'...iteration {i} for {self.word}...')

            if self.data == []:
                break

            # classify unlabelled data
            classified_data = []
            for sentence in self.data:
                transformed_sentence = self.preprocess_text(sentence)
                classified_label = self.classifier.classify(
                    transformed_sentence)
                classified_data.append(
                    (sentence, classified_label, transformed_sentence))

            # check for high confidence data
            labelled_data = []
            for sentence, label, transformed_sentence in classified_data:
                prob = float(self.classifier.prob_classify(
                    transformed_sentence).prob(label))
                # ic(prob)
                if prob > 0.85:  # add high confidence data
                    # print(f'HIGH CONFIDENCE')
                    labelled_data.append((sentence, label))
                    self.data.remove(sentence)

            self.seed.extend(labelled_data)  # add this to current seed
            self.build_classifier()  # train classifier on updated data

    def predict(self, new_sentence):
        "Prediction function"
        transformed_sentence = self.preprocess_text(new_sentence)
        predicted_synset = self.classifier.classify(transformed_sentence)
        # ic(predicted_synset)
        return predicted_synset

    def test(self):
        count = 0
        for sentence, senses in self.test_set:
            prediction = self.predict(sentence)
            if prediction in senses:
                count += 1

        accuracy = round(count/len(self.test_set), 2)
        return accuracy


def find_weights():
    "Find weights for each word"
    d = {}
    common_words = find_most_common_words()
    total = 0
    for _, count in common_words:
        total += count

    for word, count in common_words:
        d[word] = count/total

    return d


def second_method(iterations):
    "Main function call for bootstrap algo"
    with open('report/boostrap.csv', 'w', newline='') as file:
        # create header
        csv_writer = csv.writer(file)
        header = ['Word', 'Accuracy']
        csv_writer.writerow(header)

        # find weighted average
        weights = find_weights()
        average = 0

        # loop through seed set
        for word, seed in SEED_SET.items():
            model = Model(word, seed, iterations)
            model.train()
            accuracy = model.test()
            print(f'Accuracy for {word}: {accuracy:.2f}')
            csv_writer.writerow([word, accuracy])

            average += weights[word] * accuracy

        print(f'Weighted Average: {average:.2f}')
        average = round(average, 2)
        csv_writer.writerow(['Weighted Average', average])


if __name__ == '__main__':
    second_method(ITERS)
