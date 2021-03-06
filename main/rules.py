#This file contains the rules behind the STEP project
#Written by Matthew Wallace and Subhan Poudel

import nltk
import sys
from nltk.tokenize import sent_tokenize
from nltk.corpus import treebank

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
    #Reads through a list, add a classification to each element and add the element and classification to a tuple.
    #Returns a list of tuples.
    #For specific sentence types, without regard to the content of the given sentence.
    tokens_list = [] # this is a list of tokens with their pos types, each is a tuple
    for item in listText:
        #Split each sentence into tokens, and tag them
        #Then, each tagged item should be checked against our grammars, and label the sentence as the type it
        #corresponds to
        tokens = nltk.word_tokenize(item)
        pos_types = nltk.pos_tag(tokens)
        tokens_list += pos_types 
        t = treebank.parsed_sents(

        

'''
 Current Approach:Parse the given sentence and check if the given sentence is declarative.
        If the sentence is not declarative and there is no subject included,
        then the sentence should be set as imperative

'''


       #Possible Approach:Find a corpus with imperative command verbs in it,
        #train on that corpus with nltk default methods, and use that training to help
        #the program correctly recognize imperative verbs at the beginning of
        #sentences so that we can correctly categorize sentences 
        
        

        
'''

def setRules(textFile):
    with open(textFile) as fp:
        for line in fp:
            
'''
text_test  = textToString(text_file)
print setMaker(textToString(text_file))
