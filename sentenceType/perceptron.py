#Written by Matt Wallace and Subhan Poudel on 7/16/17
#This is a neural network that decides whether a sentence is imperative or declarative through a perceptron type model.

import sys
import nltk
from nltk.tokenize import sent_tokenize
from numpy import exp, array, random, dot

inFile = str(sys.argv[1])

#Getting the data ready
def syntax_classifier(sentence):
    #First input dendrite
    #Reads through a sentece, add a classification to each element and add the element and classification to a tuple.
    #Returns a list of tuples.
    tokens = nltk.word_tokenize(sentence)
    return nltk.pos_tag(tokens)

def bool_verb(sentText):
    #Second input dendrite
    #Gets a list of tuples. Each tuple is style: (word, POS)
    #Returns bool for if there is a verb
    sentinel = 0
    for item in sentText:
        if 'V' in item[1]:
            sentinel = 1
    return sentinel

def bool_verbPos(senti):
    #Third input dendrite
    #Gets a list of tuples. Each tuple is style: (word, POS)
    #Returns bool if verb is at beginning and not gerund
    sent = 0
    if 'V' in senti[0][1] and senti[0][1] != 'VBG':
        sent = 1
    return sent

def bool_VBZ(sento):
    senti = 0
    for item in sento:
        if item[1] == "VBZ":
            senti = 1
    return senti

def input_builder(dataFile):
    #Reads in the data file.
    #outputs a list of lists that contains boolean values.
    listSent = []
    with open(dataFile, 'r') as op:
        for line in op:
            results = []
            i = line.split(' ~')
            tokenTag = syntax_classifier(i[0])

            #first input dendrite
            verBool = bool_verb(tokenTag)
            results.append(verBool)

            #second input dendrite
            posVerb = 0
            if verBool == 1:
                posVerb = bool_verbPos(tokenTag)
            results.append(posVerb)
            
            #third input dendrite
            verbVBZ = 0
            if verBool == 1:
                verbVBZ = bool_VBZ(tokenTag)
            results.append(verbVBZ)

            #add results to the overall list
            listSent.append(results)  
    return listSent

def output_builder(dataFile):
    #Reads in the data file
    #outputs a list of lists of answers. [[0, 1, 1, 0]]
    ansList = []
    with open(dataFile, 'r') as fp:
        anotherList = []
        for line in fp:
            i = line.split(' ~')
            i[1] = i[1].replace('\n', '')
            anotherList.append(int(i[1]))
    ansList.append(anotherList)
    return ansList

class Perceptron():
    def __init__(self):
        #Set the weights to zero
        #A single neuron model with n inputs and 1 output
        #The weights are assigned to a n by 1 matrix, which has
        #n is based on the number of dendrites
        self.weights = [0.0, 0.0, 0.0]

        #Set the bias to zero
        self.bias = [0.0, 0.0, 0.0]

        #Set the learning rate to one
        self.learn_rate = 1.0

        #Set the bipolar target
        self.target = 0.25

    def train(self, train_inputs, train_outputs, iterations):
        #stopping condition sentinel
        stop_cond = 0
        for iteration in xrange(iterations):
            while stop_cond == 0:
                #calculates the output
                output = self.think(train_inputs)

                #calculate the answer now
                answer = self.calc_ans(output)
                
                if answer != train_outputs:
                    adjustment = 
                    
                    


    def think_single(self, single_input):
        return 0

    def calc_ans_single(self,single_input):
        return 0

    def think(self, inputs):
        #Pass inputs through the network
        #This calculates sum of each input dendrite
        #times its weight
        return self.bias + dot(inputs, self.weights)
        
    def calc_ans(self, x):
        #Goes through the logic to generate the answer
        for i in x:
            if i > self.target:
                return 1
            elif i < (-1 * self.target):
                return -1
            else:
                return 0

        
