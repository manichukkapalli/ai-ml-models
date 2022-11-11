'''
    This function load the load from the saved dir,
    cleaned the feched data using nlp technique and regex function
    
'''

#importing the dependecies

import pandas as pd
import numpy as np
import spacy
import re
import contractions
import unicodedata
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
vader=SentimentIntensityAnalyzer()
from spacy.lang.en.stop_words import STOP_WORDS
nlp = spacy.load('en_core_web_sm')

def data_cleaning(text):

    ## to remove url flag
    text= re.sub(r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?',' ', text)
    
    ## to remove the special chars or punctuation
    text= re.sub(r"[^0-9a-zA-Z:,]+",' ',text)
    
    ## to remove mutiple white spaces
    text=re.sub(r'f^[\+\sa-zA-Z]+',' ',text)
    
    ## removing hash characters
    text = re.sub('#',' ',text)  
    
    ## removoing retweet characters
    text=re.sub("(RT @[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text)

    ### applying the lemmatization
    text   = ' '.join([x.lemma_ for x in nlp(text)])

    return text


def contra_expan(text):

    if type(text)!= str:
        return text
    else:
        expanded_words=[]
        for word in text.split():
            expanded_words.append(contractions.fix(word))
            expanded_text=' '.join(expanded_words)   
        return expanded_text

def remove_accented(text):
   
    text = unicodedata.normalize('NFD', text)\
           .encode('ascii', 'ignore')\
           .decode("utf-8")
    return str(text)


def getanalysis(score):
    if(score<= - 0.05):
        return 'Negative'
    elif score >= 0.05:
        return 'Positive'
    else:
        return 'Neutral'


