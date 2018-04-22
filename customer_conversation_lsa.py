# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 02:04:48 2018

@author: kaushik
"""

"""
steps for LSA
1. TFIDF vectorizer
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

newdf['conversation'] = newdf['conversation'].apply(lambda x:preprocess(x))

    
