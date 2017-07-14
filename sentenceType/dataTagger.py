#outputs a file with sentences and their tag

import sys
import nltk
from nltk.tokenize import sent_tokenize

inFile = str(sys.argv[1])
outFile = str(sys.argv[2])

def textToString(textFile):
    #Reads in a file and assigns it to a string
    #to make it easier to deal with NLTK functions
    whole_text = ""
    with open(textFile) as z:
        for line in z:
            cleansed_line = line.replace('\n',' ')
            whole_text+=cleansed_line
    return whole_text

def setMaker(text):
    #Reads in text string and puts each sentence into an element in a list. Returns the list.
    sent_list = sent_tokenize(text)
    return sent_list

textItem = textToString(inFile)
sentList = setMaker(textItem)

with open(outFile, 'w+') as op:
    for item in sentList:
        print item
        senType = str(raw_input("Classify: "))
        op.write(item + ' ~' + senType + '\n')
