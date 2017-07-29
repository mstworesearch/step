import random
import sys

split_file = str(sys.argv[1])

lines = open(split_file).readlines()
random.shuffle(lines)
open(split_file, 'w').writelines(lines)

