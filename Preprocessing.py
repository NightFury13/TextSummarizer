#Author: Saksham Singhal
from Stemmer import Stemmer
import sys
import re,os
from collections import defaultdict

st = Stemmer('english')
pattern=re.compile(r'[\d+\.]*[\d]+|[^\w]+') #pattern to detect numbers (real/integer) non alphanumeric (no underscore)

#stopword dictionary from "stopwords.txt" file

stopWordDict = defaultdict(int)
stopWordFile = open("./stopwords.txt","r")
for line in stopWordFile:
	stopWordDict[line.strip()]=1

def extractDocumentCorpus(folder):
	os.chdir(folder)
	print folder
	document_to_senctence_corpus = {}
	for each_file in os.listdir('.'):
		print each_file
		fileptr = open(each_file,'r')
		fileText = fileptr.read()
		l = fileText.split()
		for i in xrange(len(l)):
			if (l[i].count('.')>1) or (not l[i].endswith('.')) :
				l[i] = l[i].replace('.','')
		fileText = " ".join(l)
		fileText = fileText.replace("_"," ")
		fileText = fileText.replace(".","_")
		fileText = re.sub(pattern,' ',fileText)
		fileText = re.sub(r'[\s]+',' ',fileText)
		fileText = fileText.replace('_',".")
	#	print fileText
		fileptr.close()
		if each_file not in document_to_senctence_corpus:
			print "yes"
			l= fileText.split(".")
			document_to_senctence_corpus[each_file] = l
#	print document_to_senctence_corpus.keys()
	os.chdir("..")
	return document_to_senctence_corpus


datasetFolder = 'DUC-2004/Cluster_of_Docs'
os.chdir(datasetFolder)
for cluster in os.listdir('.'):
	document_to_senctence_corpus = extractDocumentCorpus(cluster)
	print document_to_senctence_corpus
	print cluster
	break
	
		



