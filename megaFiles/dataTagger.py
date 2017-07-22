#outputs a file with sentences and their tag
#randoms the sentences

import sys
import nltk
import random
from nltk.tokenize import sent_tokenize

inFile = str(sys.argv[1])
outFile = str(sys.argv[2])
randSeed = int(sys.argv[3])

def textToString(textFile):
    #Reads in a file and assigns it to a string
    #to make it easier to deal with NLTK functions
    whole_text = ""
    with open(textFile) as z:
        count = 0
        for line in z:
            cleansed_line = line.replace('\n',' ')
            whole_text+=cleansed_line
            count += 1
        print "Total number of lines: "
        print count
    return whole_text

def setMaker(text):
    #Reads in text string and puts each sentence into an element in a list. Returns the list.
    sent_list = sent_tokenize(text)
    return sent_list

textItem = textToString(inFile)
sentList = setMaker(textItem)
random.seed(randSeed)
random.shuffle(sentList)

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
        
