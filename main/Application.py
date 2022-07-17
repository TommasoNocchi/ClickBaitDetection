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
from MongoDBManager import *
from CLI import *

# ***** functions *****
french_stemmer = nltk.stem.SnowballStemmer('english')
class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([french_stemmer.stem(w) for w in analyzer(doc)])

# ******************************************************************************** initialisation ********************************************************************************

categories = ['clickbait', 'non_clickbait']
dataset = load_files("./data", description=None, categories=categories, encoding='cp437', load_content=True, shuffle=True, random_state=42)
text_file = open("./stop_words/stop_words.txt", "r")
stop_words = text_file.read().split('\n')

training_data, testing_data, training_labels, testing_labels = train_test_split(dataset.data, dataset.target, test_size=0.2)
classifier = svm.LinearSVC()

# ******************************************************************************** Traning step **********************************************************************************
print("Classifier: ", classifier.__class__.__name__)
pipeline = Pipeline([
    ('vect', StemmedCountVectorizer(analyzer="word",stop_words=frozenset(stop_words), max_features=150)),
    #('vect', CountVectorizer(analyzer='word', stop_words=frozenset(stop_words))),
    #('vect', CountVectorizer()),
    #('chi2', SelectPercentile(chi2, percentile=50)),
    ('tfidf', TfidfTransformer()),
    ('clf', classifier),
])
# training
start = time.time()
pipeline.fit(training_data, training_labels)
end = time.time()
print("Tranining Execution time: ", end-start, " seconds")

# ******************************************************************************** start the applciation ****************************************************************************
cli = CLI()
mongoDBManager = MongoDBManager()
start = ""
logged = False
while(start != "end"):
    start = cli.startWelcomeMenu()
    user = {
        '_id': None
    }

    #Log-In
    if(start == "S"):
        print("=======> Sign In:")
        logIn = []

        logIn = cli.log()
        user = mongoDBManager.retriveUser(logIn[0])
        password = user.get("password")
        if(password == logIn[1]):
            logged = True

    #Registration
    elif(start == "R"):
        print("=======> Registration:")
        registration = []
        chekUserNotExist = False;

        while(chekUserNotExist != True):
            registration = cli.log()
            user = mongoDBManager.retriveUser(registration[0])
            if(user == None): #because the username is the _id field in MongoDB and it must be unique
                mongoDBManager.insertUser(registration[0], registration[1])
                chekUserNotExist = True
                user = {
                    '_id' : registration[0],
                    'password' : registration[1]
                }

        logged = True

    if(logged):
        print("***Welcome ",user.get("_id"),"***")

    choice = "M"
    while(logged):
        #questionString = input("Do you want to insert a new title for checking it? (Y/N) ")
        if (choice == "Y"):
            inputString = input("Entre the title to check: ")
            doc_new = []
            doc_new.append(inputString)

            start = time.time()

            # Adopting the text classifier:
            predicted = pipeline.predict(doc_new)  # prediction

            # Result of the prediction on new texts
            for doc, category in zip(doc_new, predicted):
                print('%r => %s' % (doc, dataset.target_names[category]))
                mongoDBManager.insertNewPrediction(user.get("_id"), inputString, dataset.target_names[category] )

            end = time.time()
            print("Execution time: ", end - start, " seconds")

        elif(choice == "L"):
            list = mongoDBManager.retriveTitlesList(user.get("_id"))
            if(list != None):
                for title in list:
                    print("*")
                    print(title.get("title"))
                    print(title.get("prediction"))
                    print("*")
            else:
                print("List not present")

        elif(choice == "N"):
            logged = False
            chekUserNotExist = False
            break

        elif (choice == "M"):
            cli.mainMenu()
        choice = input("Insert a command or press M to see the menu\n> ")


"""
questionString = ""
while(questionString != "N"):
    questionString = input("Do you want to insert a new title for checking it? (Y/N) ")
    if(questionString == "Y"):
        inputString = input("Entre the title to check: ")
        doc_new = []
        doc_new.append(inputString)

        start = time.time()

        # Adopting the text classifier:
        predicted = pipeline.predict(doc_new)  # prediction

        # Result of the prediction on new texts
        for doc, category in zip(doc_new, predicted):
            print('%r => %s' % (doc, dataset.target_names[category]))

        end = time.time()
        print("Execution time: ", end - start, " seconds")
"""

"""
doc_new.append(inputString)
new_counts = counter_vect.transform(doc_new)
new_tfidf = tfidf_transformer.transform(new_counts)

print('Values of features extracted from the first testing document')
print(new_tfidf[0])
print('Values of features extracted from the second testing document')
print(new_tfidf[1])

start = time.time()
predicted = classifier.predict(new_tfidf)  # prediction
end = time.time()
print("Execution time: ", end - start, " seconds")

for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, dataset.target_names[category]))
"""