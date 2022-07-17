import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from sklearn.datasets import load_files
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn import tree
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn import neighbors
from sklearn import ensemble
from sklearn.feature_selection import SelectPercentile, chi2
import numpy as np
import matplotlib.pyplot as plt
import time

french_stemmer = nltk.stem.SnowballStemmer('english')
class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([french_stemmer.stem(w) for w in analyzer(doc)])

#vectorizer_s = StemmedCountVectorizer(min_df=3, analyzer="word", stop_words='english')

def classify(dataset, classifier, stop_words):
    print("Classifier: ", classifier.__class__.__name__)
    start = time.time()
    pipeline = Pipeline([
        ('vect', StemmedCountVectorizer(analyzer="word", ngram_range=(2,2),stop_words=frozenset(stop_words), max_features=150)),
        #('vect', CountVectorizer(analyzer='word', stop_words=frozenset(stop_words))),
        #('vect', CountVectorizer()),
        #('chi2', SelectPercentile(chi2, percentile=50)),
        ('tfidf', TfidfTransformer()),
        ('clf', classifier),
    ])

    # prediction in cross-validation
    prediction = cross_val_predict(pipeline, dataset.data, dataset.target, cv=5)
    end = time.time()
    print("Execution time: ", end-start, " seconds")
    return prediction

def create_stop_words():
    # building stop-words array
    f = open('./stop_words/stop_words.txt', 'r')
    stop_words = []

    for line in f:
        stop_words.append(line)

    return stop_words

def print_results(dataset, prediction):
    # metrics extractions (precision    recall  f1-score   support)
    print(metrics.classification_report(dataset.target, prediction, target_names=dataset.target_names))
    print("Confusion matrix:\n", metrics.confusion_matrix(dataset.target, prediction))

categories = ['clickbait', 'non_clickbait']
dataset = load_files("./data", description=None, categories=categories, encoding='cp437', load_content=True, shuffle=True, random_state=42)
#stop_W = create_stop_words()


text_file = open("./stop_words/stop_words.txt", "r")
stop_words = text_file.read().split('\n')

# Training the first classifier
prediction = classify(dataset, MultinomialNB(), stop_words)
print_results(dataset, prediction)

# Training the second classifier
prediction = classify(dataset, svm.LinearSVC(), stop_words)
print_results(dataset, prediction)

# Training the third classifier
prediction = classify(dataset, tree.DecisionTreeClassifier(), stop_words)
print_results(dataset, prediction)

# Training the forth classifier
prediction = classify(dataset, neighbors.KNeighborsClassifier(1), stop_words)
print_results(dataset, prediction)

# Training the fifth classifier
prediction = classify(dataset, neighbors.KNeighborsClassifier(3), stop_words)
print_results(dataset, prediction)

# Training the sixth classifier
prediction = classify(dataset, neighbors.KNeighborsClassifier(5), stop_words)
print_results(dataset, prediction)

# Training the seventh classifier
prediction = classify(dataset, ensemble.RandomForestClassifier(), stop_words)
print_results(dataset, prediction)

# Training the eighth classifier
prediction = classify(dataset, ensemble.AdaBoostClassifier(), stop_words)
print_results(dataset, prediction)





