#Written by Matt Wallace and Subhan Poudel on 7/16/17
#This is a neural network that decides whether a sentence is imperative or declarative through a perceptron type model.
#The program will stop once the weights do not change on two iterations of the entire input set.
#The code is similar to neuralNet.py but uses a perceptron model neural net.

import sys
import nltk
from nltk.tokenize import sent_tokenize
from numpy import exp, array, random, dot



#Getting the data ready
def syntax_classifier(sentence):
    #Reads through a sentece, add a classification to each element and add the element and classification to a tuple.
    #Returns a list of tuples.
    tokens = nltk.word_tokenize(sentence)
    return nltk.pos_tag(tokens)

def bool_verb(sentText):
    #First input dendrite
    #Gets a list of tuples. Each tuple is style: (word, POS)
    #Returns bool for if there is a verb
    sentinel = 0
    for item in sentText:
        if 'V' in item[1]:
            sentinel = 1
    return sentinel

def bool_verbPos(senti):
    #Second input dendrite
    #Gets a list of tuples. Each tuple is style: (word, POS)
    #Returns bool if verb is at beginning and not gerund
    sent = 0
    if 'V' in senti[0][1]:
        sent = 1
    return sent

def bool_VBZ(sento):
    #Third input dendrite
    #Checks for the presence of a VBZ (verb singular third person) instance in the entire sentence
    senti = 0
    for item in sento:
        if item[1] == "VBZ":
            senti = 1
    return senti

def gerund_bool(sentence):
    #Fourth input dendrite
    #Checks for the presence of a gerund in a given sentence.
    for item in sentence:
        if item[1] == "VBG":
            return 1
        else:
            return 0

def gerund_first(sentence):
    #Fifth input dendrite
    #Checks for the presence of a gerund at the beginning of a sentence.
    if 'VBG' in sentence[0][1]:
        return 1
    else:
        return 0

def question_bool(sentence):
    c = len(sentence) -1 
    #Sixth input dendrite
    #Checks for the presence of a question mark in a given sentence
    if '?' in sentence[c][1] or '?' in sentence[c-1][1]:
        return 1
    else:
        return 0

def exclamation_bool(sentence):
    c = len(sentence) -1 
    #Seventh input dendrite
    #Checks for the presence of an exclamation point in a given sentence
    if '!' in sentence[c][1] or '!' in sentence[c-1][1]:
        return 1
    else:
        return 0

def colon_bool(sentence):
    #Eighth input dendrite
    #Checks for the presence of a colon instance in the entire sentence
    
    for item in sentence:
        if item[0] == ":":
            return 1
    return 0

def semi_colon_bool(sentence):
    #Ninth input dendrite
    #Checks for the presence of a semi-colon instance in the entire sentence
    
    for item in sentence:
        if item[0] == ";":
            return 1
    return 0

def proper_noun_bool(sentence):
    #Tenth input dendrite
    #Checks for the presence of a proper noun instance in the entire sentence
    
    for item in sentence:
        if item[1] == "NNP" or item[1] == "NNPS":
            return 1
    return 0
    
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

            #Fourth input dendrite
            verbGerund = 0
            if verBool == 1:
                verbGerund = gerund_bool(tokenTag)
            results.append(verbGerund)

            #Fifth input dendrite
            gerundFirst = 0
            if verbGerund == 1:
                gerundFirst = gerund_first(tokenTag)
            results.append(gerundFirst)

            #Sixth input dendrite
            questionMark = question_bool(tokenTag)
            results.append(questionMark)

            #Seventh input dendrite
            exclamationMark = exclamation_bool(tokenTag)
            results.append(exclamationMark)

            #Eighth input dendrite
            colonMark = colon_bool(tokenTag)
            results.append(colonMark)

            #Ninth input dendrite
            semiColonMark = semi_colon_bool(tokenTag)
            results.append(semiColonMark)

            #Tenth input dendrite
            properNoun = proper_noun_bool(tokenTag)
            results.append(properNoun)

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
        self.weights = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        #Set the bias to zero
        self.bias = 0.0
        
        #Set the learning rate to one
        self.learn_rate = 0.99

        #Set the bipolar target
        self.target = 0.5

    def train(self, train_inputs, train_outputs):
        #stopping condition sentinel
        stop_cond = 0
        counter = 0
        hello = [0,0,0,0,0,0,0,0,0,0]
        while stop_cond == 0:
            for x in xrange(10):
                hello[x] = self.weights[x]
            
            for i in xrange(len(train_inputs)):
                #calculates the output
                output = self.think_single(train_inputs[i])
                #calculate the answer now
                answer = self.calc_ans_single(output)
                
                if answer != train_outputs[0][i]:
                    number = len(self.weights)
                    for k in xrange(number):
                        self.weights[k] +=  self.learn_rate * train_outputs[0][i] * train_inputs[i][k]
                    self.bias += self.learn_rate * train_outputs[0][i]
               
            counter += 1
            print "This is hello: " + str(hello) 
            print self.weights
            if hello == self.weights:
                print "How many times the while loop ran: "
                print counter
                stop_cond = 1
            

    def think_single(self, single_input):
        summa = 0.0
        num = len(single_input)
        for i in xrange(num):
            summa += single_input[i] * self.weights[i]
        return self.bias + summa 

    
    def calc_ans_single(self,single_input):
        if single_input > self.target:
            return 1
        elif single_input < (-1 * self.target):
            return -1
        else:
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

    def check(self, test_inputs, test_outputs):
        correct = 0.0
        total = len(test_outputs[0])
        for i in xrange(len(test_inputs)):
            #calculates the output
            output = self.think_single(test_inputs[i])
            #calculate the answer now
            answer = self.calc_ans_single(output)
            print "Algo thinks the answer is: " + str(answer)
            print "The actual answer is: " + str(test_outputs[0][i])
            if answer == test_outputs[0][i]:
                correct += 1

        return 'Percentage correct: ' + str((correct/total)*100) + '%'
                
        
        

def main():
    perceptron = Perceptron()
    inFile = str(sys.argv[1])
    outFile = str(sys.argv[2])

    input_set = input_builder(inFile)
    output_set = output_builder(inFile)

    test_input_set = input_builder(outFile)
    test_output_set = output_builder(outFile)

    #print input_set
    
    print "Bias before training: "
    print perceptron.bias


    print "Perceptron weights before training: "
    print perceptron.weights

    
    perceptron.train(input_set,output_set)
    test_accuracy = perceptron.check(test_input_set,test_output_set)

    print "Bias after training: "
    print perceptron.bias
    print "Print the resulting weights after training: "
    print perceptron.weights

    print test_accuracy
main()
