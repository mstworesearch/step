#Written by Subhan Poudel and Matt Wallace on 7/14/17
#This is a neural network that decides whether a sentence is imperative or declarative.

import sys
import nltk
from nltk.tokenize import sent_tokenize
from numpy import exp, array, random, dot

inFile = str(sys.argv[1])
seedIn = int(sys.argv[2])

#Getting the data ready
def syntax_classifier(sentence):
    #Reads through a sentece, add a classification to each element and add the element and classification to a tuple.
    #Returns a list of tuples.
    tokens = nltk.word_tokenize(sentence)
    return nltk.pos_tag(tokens)

def bool_verb(sentText):
    #Gets a list of tuples. Each tuple is style: (word, POS)
    #Returns bool for if there is a verb
    sentinel = 0
    for item in sentText:
        if 'V' in item[1]:
            sentinel = 1
    return sentinel

def bool_verbPos(senti):
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

            #first input
            verBool = bool_verb(tokenTag)
            results.append(verBool)

            #second input
            posVerb = 0
            if verBool == 1:
                posVerb = bool_verbPos(tokenTag)
            results.append(posVerb)
            
            #third input
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

class NeuralNetwork():
    def __init__(self):
        # Seed random number generator
        random.seed(seedIn)
 
        #A single neuron model with 3 inputs and 1 output
        #random weights are assigned to a 3 by 1 matrix, which
        #has values in the range -1 to 1 and a mean of 0
        self.synaptic_weights = 2 * random.random((3,1)) - 1
 
    def __sigmoid(self,x):
        return 1 / (1 + exp(-x))
 
    def __sigmoid_derivative(self,x):
        return x * (1-x)
 
       
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
            #pass the training set through our neural net
        for iteration in xrange(number_of_training_iterations):
            output = self.think(training_set_inputs)
 
                #calculate the error between the desired output and the predicted output
            error = training_set_outputs - output
 
                #multiply the error by the input and the sigmoid function in order to
                #adjust it by how far away it is from an ideal value
		#adjustment is a vector of inputs that are each dotted with the corrections (error*sigmoid)

            adjustment = dot(training_set_inputs.T, error*self.__sigmoid_derivative(output))
 
                #adjust the weights
		#the below line adds the adjustment to each of the existing synaptic weights
            self.synaptic_weights += adjustment
 
    def think(self,inputs):            # Pass inputs through our neural network (our single neuron).
        return self.__sigmoid(dot(inputs, self.synaptic_weights))
 
 
 

#Intialise a single neuron neural network.
neural_network = NeuralNetwork()
 
print "Random starting synaptic weights: "
print neural_network.synaptic_weights

inputSet = input_builder(inFile)
print inputSet
outputSet = output_builder(inFile)
print outputSet

# The training set. We have 4 examples, each consisting of 3 input values
# and 1 output value.
training_set_inputs = array(inputSet)
print training_set_inputs
training_set_outputs = array(outputSet).T
print training_set_outputs
 
# Train the neural network using a training set.
# Do it 10,000 times and make small adjustments each time.
neural_network.train(training_set_inputs, training_set_outputs, 10000)
 
print "New synaptic weights after training: "
print neural_network.synaptic_weights
 
# Test the neural network with a new situation.
print "Considering new situation,  [1, 0, 0] -> ?: "
print neural_network.think(array([1, 0, 0]))