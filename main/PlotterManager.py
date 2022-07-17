from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import svm
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
from sklearn.model_selection import train_test_split

# ***** functions *****
french_stemmer = nltk.stem.SnowballStemmer('english')
class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([french_stemmer.stem(w) for w in analyzer(doc)])

def get_n_gram(corpus, ngram_range, max_features=None):
  vectorizer = CountVectorizer(ngram_range=ngram_range,max_features=max_features).fit(corpus)
  bag_of_words = vectorizer.transform(corpus)
  sum_words = bag_of_words.sum(axis=0)
  words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
  words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
  return words_freq

categories = ['clickbait', 'non_clickbait']
dataset=load_files("./data", description=None, categories=categories, encoding='cp437', load_content=True, shuffle=True, random_state=42)

# splitting Training and Test set
training_data, testing_data, training_labels, testing_labels = train_test_split(dataset.data, dataset.target, test_size=0.2)

#classifier = svm.LinearSVC()
classifier = tree.DecisionTreeClassifier()

print("Classifier: ", classifier.__class__.__name__)
pipeline = Pipeline([
    ('vect', StemmedCountVectorizer(analyzer="word" , max_features=150)),
    ('chi2', SelectPercentile(chi2, percentile=50)),
    ('tfidf', TfidfTransformer()),
    ('clf', classifier),
])
# training
pipeline.fit(training_data, training_labels)
plot_confusion_matrix(pipeline, testing_data, testing_labels)
plt.show()

""" CODE for the N-Gram vectorization
ngram_fetch = get_n_gram(training_data, (1,1), 150)
ngram_df = pd.DataFrame(ngram_fetch, columns=['Keyword', 'Frequency'])

ngram_df.info()
ngram_df.describe()
ngram_df.loc[:]
plt.scatter(ngram_df['Keyword'], ngram_df['Frequency'])
plt.show()

ngram_vectorizer = CountVectorizer(ngram_range=(1,1), max_features=40)
ngram_training_count = ngram_vectorizer.fit_transform(training_data)
print("List of extracted tokens")
print(ngram_vectorizer.get_feature_names_out())
print(ngram_vectorizer.vocabulary_)

newdict={k:v for k, v in sorted(ngram_vectorizer.vocabulary_.items(),reverse=True,key=lambda key:key[1])}

print(newdict)
"""