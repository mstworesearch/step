#Created by Subhan Poudel and Matt Wallace on 7/24/2017
#Given a tagged file, the program will create a file with all
#positive and all negative outputs

inFile = str(raw_input("What is the file? "))
outFile = "imperSplit.txt"
outFile1 = "negaSplit.txt"

with open(inFile) as z:
    with open(outFile, 'w+') as op:
        for line in z:
            if '~1' in line:
                op.write(line)

    with open(outFile1, 'w+') as hg:
        for line in z:
            if '~-1' in z:
                hg.write(line)
