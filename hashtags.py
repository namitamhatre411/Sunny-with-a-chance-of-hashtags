import pandas as p
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
import scipy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn import svm
import nltk
from nltk.tokenize import wordpunct_tokenize
np.set_printoptions(suppress =True, precision=3)

train = p.read_csv('train.csv')
test = p.read_csv('test.csv')

class SnowballTokenizer(object):
     def __init__(self):
         self.wnl = nltk.stem.SnowballStemmer("english")
     def __call__(self, doc):
         return [self.wnl.stem(t) for t in wordpunct_tokenize(doc)]

classify = Pipeline([('VEC',TfidfVectorizer(max_features=85000, strip_accents='unicode',  
     analyzer='word',token_pattern=r'\w{3,}',sublinear_tf=1,
     ngram_range=(1, 1),stop_words = 'english',tokenizer = SnowballTokenizer())),
     ('clf',svm.LinearSVC())])

y = np.array(train.ix[:,4:])
 
classify = classify.fit(train['tweet'],y)


