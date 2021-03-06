#Created by Subhan Poudel and Matthew Wallace on 6/6/17 for the step project
#Last modified by Subhan Poudel on 6/7/17
#Sulav Acharaya, specifically, was not present

#File that extracts text from downloaded HTML files

from bs4 import BeautifulSoup
import sys

userData = str(sys.argv[1])

if ".html" in userData:
	newStr = userData.replace(".html", "Parsed.txt")
elif ".txt" in userData:
	newStr = userData.replace(".txt", "Parsed.txt")
else:
	newStr = userData + "Parsed.txt"

f = open(newStr, 'w+')

with open(userData) as fp:
    soup = BeautifulSoup(fp, "html.parser")

data = soup.find('title').getText()
f.write(data + '\n\n')

for a in soup.findAll('a', href = True):
	f.write("URL: " + a['href'])

for node in soup.findAll('p'):
   f.write(''.join(node.findAll(text=True)) + '\n')

fp.close()
f.close()
