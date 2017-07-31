#Written by Subhan Poudel and Matt Wallace on 7/14/17
#This is a neural network that decides whether a sentence is imperative or declarative.
#This file takes in the name of the tagged file created by dataTagger.py as the first command line arguement
#as training data. The second command line arguement takes in another file created by dataTagger.py for testing
#and returns the accuracy of the nerual net. The functions before the class "NeuralNet" create the inputs that 
#the neural net actually sees to create a file. Each file is translated to a list of n (currently 10) binary inputs
#that the nueral net uses to calculate one output. "1" is imperative, "0" if unsure and "-1" if declarative.
#The algorithm for the nerual net was found online but modified for this project.

import sys
import nltk
from nltk.tokenize import sent_tokenize
from numpy import exp, array, random, dot

inFile = str(sys.argv[1])

#seedIn = int(sys.argv[2])

#Getting the data ready
def syntax_classifier(sentence):
    #Reads through a sentence, adds a classification to each element and adds the element and classification to a tuple.
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
    #outputs a list of lists. Each list are the inputs for each sentence.
    listSent = []
    with open(dataFile, 'r') as op:
        for line in op:
            results = []
            i = line.split(' ~')
            tokenTag = syntax_classifier(i[0])

            #first input
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

class NeuralNetwork():
    def __init__(self):
        # Seed random number generator
        #random.seed(seedIn)
 
        #Each neural has 10 inputs so there are 10 overall weights that the NN has to keep track of
        self.synaptic_weights = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
 
    def __sigmoid(self,x):
        return 1 / (1 + exp(-x))
 
    def __sigmoid_derivative(self,x):
        return x * (1-x)
 
       
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
            #pass the training set through our neural net
        numOut = len(training_set_outputs[0])
        for iteration in xrange(number_of_training_iterations):
            for i in xrange(numOut):
                output = self.think(training_set_inputs[i])
                #print "The output is: " + str(output)
                #print "The actual answer is: " + str(training_set_outputs[0][i])
                #calculate the error between the desired output and the predicted output
                if training_set_outputs[0][i] == -1:
                    error = 0 - output
                else:
                    error = training_set_outputs[0][i] - output
                    
                #print "The error is: " + str(error)
 
                #multiply the error by the input and the sigmoid function in order to
                #adjust it by how far away it is from an ideal value
		#adjustment is a vector of inputs that are each dotted with the corrections (error*sigmoid)

                adjustment = self.adjustCalc(training_set_inputs[i], error, output)
                #print "The adjust is: " + str(adjustment)
                
                #adjust the weights
		#the below line adds the adjustment to each of the existing synaptic weights
                for n in xrange(10):
                    self.synaptic_weights[n] += adjustment[n]
 
    def think(self,inputs):            # Pass inputs through our neural network (our single neuron).
        summation = 0
        for num in xrange(10):
            summation += inputs[num] * self.synaptic_weights[num]
        return self.__sigmoid(summation)
 
    def adjustCalc(self, value, error, output):
        vector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for i in xrange(10):
            vector[i] = value[i] * (error * self.__sigmoid_derivative(output))
        return vector

    def calc_ans_single(self, inputs):
        if inputs <= 0.5:
            return -1
        else:
            return 1

    def check(self, test_inputs, test_outputs):
        correct = 0.0
        total = len(test_outputs[0])
        for i in xrange(len(test_inputs)):
            #calculates the output
            output = self.think(test_inputs[i])
            #calculate the answer now
            answer = self.calc_ans_single(output)
            print "The output is: " + str(output)
            print "Algo thinks the answer is: " + str(answer)
            print "The actual answer is: " + str(test_outputs[0][i])
            if answer == test_outputs[0][i]:
                correct += 1

        return 'Percentage correct: ' + str((correct/total)*100) + '%'


        
#Intialise a single neuron neural network.
neural_network = NeuralNetwork()
 
print "Non-Random starting synaptic weights: "
print neural_network.synaptic_weights

inputSet = input_builder(inFile)
#print inputSet
outputSet = output_builder(inFile)
#print outputSet

# The training set. We have 4 examples, each consisting of 3 input values
# and 1 output value.
#training_set_inputs = array(inputSet)
#print training_set_inputs
#training_set_outputs = array(outputSet).T
#print training_set_outputs
 
# Train the neural network using a training set.
neural_network.train(inputSet, outputSet, 10000)

print "New synaptic weights after training: "
print neural_network.synaptic_weights

testFile = str(sys.argv[2])

test_input_set = input_builder(testFile)
test_output_set = output_builder(testFile)

test_accuracy = neural_network.check(test_input_set, test_output_set)

print test_accuracy

