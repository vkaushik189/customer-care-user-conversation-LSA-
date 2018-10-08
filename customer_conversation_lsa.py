# -*- coding: utf-8 -*-
"""
@author: kaushik
"""

"""
steps for LSA
1. TFIDF vectorizer or any other techniques such as word2vec or countvectorizer
2. LSA

or

1. Count Vectorier
2. Doc-term matrix
3. LSA 
"""
import pandas as pd
import numpy as np
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn 
import glob
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from pylab import *


#importing all files into a dataframe
path = "C:\\git\\customer-conversation----LDA\\files"
allfiles = glob.glob(path + "/*.tsv")
frame = pd.DataFrame()
list_ = []
for file in allfiles:
    dframe = pd.read_csv(file, sep='\t', header=None, index_col=None, names=['time','person1','person2','conversation'], error_bad_lines = False)
    list_.append(dframe)
frame = pd.concat(list_)

data = pd.DataFrame(frame.iloc[:,3]).reset_index(drop=True)

#we know every 4 lines is one conversatio, so lets combine wvery 4 lines
x=0
y=0
newdf = pd.DataFrame(np.nan, index = range(0,int((data.shape[0]/4))), columns = ['conversation'])
for i in range(0, int((data.shape[0]/4))):
    newdf.conversation[x] = data.conversation[y] + " " +data.conversation[y+1] + " " + data.conversation[y+2] + " " + data.conversation[y+3]
    x += 1
    y += 4
    

#the data is in proper format, let's start preprocessing it
#remove punctuations
def clean(text):
    text = text.lower()
    text = re.sub("\;|\=|\%|\^|\_|\*|\'|\"|\?|\.|\,|\:|\<|\>|\*|\@|\#|\&|\[|\]"," ",text)
    text = re.sub("www"," ",text)
    text = re.sub("com"," ",text)
    text = re.sub("thanks"," ", text)
    text = re.sub("  ", " ", text)
    text = ' '.join(w for w in text.split() if len(w)>1)
    text = text.strip()
    return text
newdf['conversation'] = newdf['conversation'].apply(lambda x: clean(x))

#remove stopwords and lemmatizing the text
stopw = set(stopwords.words('english'))
lemma = WordNetLemmatizer()
def preprocess(doc):
    doc = ' '.join([w for w in doc.split() if w not in stopw])
    doc = ' '.join(lemma.lemmatize(w) for w in doc.split())
    return doc

# to get in dataframe format
newdf['conversation'] = newdf['conversation'].apply(lambda x:preprocess(x))
#to get it in a list format
newdf_list = [preprocess(doc) for doc in newdf['conversation']]
    


#calculating tf-idf vectoriser
vectorizer = TfidfVectorizer(max_df=0.8, max_features=4000,
                             min_df=2, stop_words='english',
                             use_idf=True)

# Build the tfidf vectorizer from the training data ("fit"), and apply it 
x_train_tfidf = vectorizer.fit_transform(newdf_list)
print("  Actual number of tfidf features: %d" % x_train_tfidf.get_shape()[1])

# Get the words that correspond to each of the features.
feat_names = vectorizer.get_feature_names()





lsa_model = TruncatedSVD(n_components=5)
lsa_topic_matrix = lsa_model.fit_transform(x_train_tfidf)


for compNum in range(0, 10):

    comp = lsa_model.components_[compNum]
    
    # Sort the weights in the first component, and get the indeces
    indeces = np.argsort(comp).tolist()
    
    # Reverse the indeces, so we have the largest weights first.
    indeces.reverse()
    
    # Grab the top 10 terms which have the highest weight in this component.        
    terms = [feat_names[weightIndex] for weightIndex in indeces[0:10]]    
    weights = [comp[weightIndex] for weightIndex in indeces[0:10]]    
   
    # Display these terms and their weights as a horizontal bar graph.    
    # The horizontal bar graph displays the first item on the bottom; reverse
    # the order of the terms so the biggest one is on top.
    terms.reverse()
    weights.reverse()
    positions = arange(10) + .5    # the bar centers on the y axis
    
    figure(compNum)
    barh(positions, weights, align='center')
    yticks(positions, terms)
    xlabel('Weight')
    title('Strongest terms for component %d' % (compNum))
    grid(True)
    show()
