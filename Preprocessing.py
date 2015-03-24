#Author: Saksham Singhal
from Stemmer import Stemmer
import sys
import re,os
import math
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
	#		print "yes"
	#		l= fileText.split(".")
			document_to_senctence_corpus[each_file] = fileText
	os.chdir("..")
	return document_to_senctence_corpus


def generateInverseDocFrequency(corpus):
	total_docs = len(corpus.keys())
	idf_scores = defaultdict(float)
	term_doc_count = defaultdict(list)
	for each_doc in corpus:
		current_doc = corpus[each_doc]
		for word in current_doc.split():
			word = word.replace('.','')
			if word not in stopWordDict:
				word = st.stemWord(word)

				#Checking the douments it is belonging to

				if word not in term_doc_count:
					term_doc_count[word] = [each_doc]
				elif each_doc not in term_doc_count[word]:
						term_doc_count[word].append(each_doc)

	for term in term_doc_count:
		idf_scores[term] = math.log10(1+ ((1.0*total_docs)/len(term_doc_count[term])))

	return idf_scores	




datasetFolder = 'DUC-2004/Cluster_of_Docs'
os.chdir(datasetFolder)
for cluster in os.listdir('.'):
	document_to_senctence_corpus = extractDocumentCorpus(cluster)
	idf_scores = generateInverseDocFrequency(document_to_senctence_corpus)


	################################################################
	#														       #
	#			Here we need to start clustering the Docs          #
	# 															   #	
	################################################################


	break
	
		



