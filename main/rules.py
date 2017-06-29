#This file contains the rules behind the STEP project
#Written by Matthew Wallace and Subhan Poudel

import nltk
import sys
from nltk.tokenize import sent_tokenize
#from nltk.tokenize.punkt import PunktSentenceTokenizer

text_file = str(sys.argv[1])


'''
We need the following functions:
1. Tokenizer for whole text
2. Break text up by sentence
3. Break text up by sentence type 
4. Function that looks for a colon at the end of a sentence? 
5. 


'''

def textToString(textFile):
    #Reads in a file and assigns it to a string
    #to make it easier to deal with NLTK functions
    whole_text = ""
   # file = open(textFile, 'r')
    with open(textFile) as z:
        for line in z:
            cleansed_line = line.replace('\n',' ')
            whole_text+=cleansed_line
    return whole_text

def setMaker(text):
    #Reads in text string and puts each sentence into an element in a list. Returns the list.
    sent_list = sent_tokenize(text)
    return sent_list

def syntax_classifier(listText):
    #Reads through a list, add a classification to each element and add the element and classification to a tuple. Returns a list of tuples.
    for item in listText:
        
'''
def setRules(textFile):
    with open(textFile) as fp:
        for line in fp:
            
'''
text_test  = textToString(text_file)
print setMaker(textToString(text_file))
