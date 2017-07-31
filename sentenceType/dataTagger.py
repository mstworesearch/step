#Created by Subhan Poudel and Matthew Wallace for the step project
#This program takes in the name of a txt file as the first command line arguement and
#outputs a txt file with the name given of the second txt file.
#This program will ask you to classify sentences as declaritive(-1) or imperative(1). 
#The output file has each sentence with their tag

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
    counter = 0
    for item in sentList:
        print item
        senType = str(raw_input("Classify: "))
	if senType == "exit":
            print counter
            break
        elif senType == "d":
            print "Line not included"
        else:    
            op.write(item + ' ~' + senType + '\n')
        counter += 1
        if counter % 10 == 0:
            print counter
        
