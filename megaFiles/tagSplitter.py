#Created by Subhan Poudel and Matt Wallace on 7/24/2017
#Given a tagged file, the program will create a file with all
#positive and all negative outputs
import sys

inFile = str(raw_input("What is the file? "))
outFile = "imperativeSplit.txt"
outFile1 = "declarativeSplit.txt"

with open(inFile) as z:
    with open(outFile, 'w+') as op:
        with open(outFile1, 'w+') as hg:
            for line in z:
                if '~1' in line:
                    op.write(line)
                else:
                    hg.write(line)




