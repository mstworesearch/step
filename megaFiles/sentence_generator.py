#Written by Matthew Wallace and Subhan Poudel on July 24, 2017

import sys
import random
imperative_num = int(sys.argv[1])
declarative_num = int(sys.argv[2])
imperative_file = str(sys.argv[3])
declarative_file = str(sys.argv[4])
final_file = str(sys.argv[5])
rand_seed = int(sys.argv[6])

with open(final_file, 'w+') as f:
    with open(imperative_file, 'r') as i:
        x = i.readlines()
        random.shuffle(x)
        for i in xrange(imperative_num):
            f.write(x[i])
        with open(declarative_file, 'r') as d:
            y = d.readlines()
            random.shuffle(y)
            for i in xrange(declarative_num):
                f.write(y[i])

       

