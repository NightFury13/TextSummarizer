#Author: Saksham Singhal
from Stemmer import Stemmer
import sys
import re,os
import math
from collections import defaultdict
from copy import deepcopy
import subprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import SpectralClustering

st = Stemmer('english')
pattern=re.compile(r'[\d+\.]*[\d]+|[^\w]+') #pattern to detect numbers (real/integer) non alphanumeric (no underscore)

Summary = []
lamda = 6
alpha  = 0.75
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
		fileText = fileText.replace('Ms.','Ms')
		fileText = fileText.replace('Mrs.','Mrs')
		fileText = fileText.replace('Mr.','Mr')
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

def generateClusterInputFile(corpus):
	ClusterInputFile = "../../Temp/SentencesToCluster.txt"
	ClusterInputFile_ptr = open(ClusterInputFile,'w')
	for each_doc in corpus:
		current_doc = corpus[each_doc]
		sentences = []
		sentences = current_doc.split('.')
		print each_doc
		print len(sentences)
	#	break
		for each_sentence in sentences:
			if len(each_sentence)>1:
				if each_sentence[0]==' ':
					each_sentence = each_sentence[1:]
				ClusterInputFile_ptr.write(each_sentence+'\n')

	ClusterInputFile_ptr.close()

def clusterSentencesandConsolidate():
	ClusterFile = open("../../Temp/SentencesToCluster.txt",'r')
	documents = ClusterFile.readlines()
	ClusterFile.close()
	line_count = len(documents)
	vectorizer = TfidfVectorizer(stop_words='english')
	X = vectorizer.fit_transform(documents)

	noOfClusters = line_count/10
	#####
	model = SpectralClustering(n_clusters=noOfClusters,eigen_solver='arpack',eigen_tol=0.01,assign_labels = 'discretize')
	y = model.fit_predict(X)

	clusterSentenceIndex = []
	for i in xrange(len(y)):
		temp = []
		temp.append(y[i])
		temp.append(documents[i])

		clusterSentenceIndex.append(temp)

	clusterSentenceIndex.sort()

	# Writing to the file
#	outputIndexFile = open('../../Temp/sentence-sluster-sorted-index.txt','w')
#	for i in xrange(len(clusterSentenceIndex)):
#		if int(clusterSentenceIndex[i][0]) >= 0:
#			line = clusterSentenceIndex[i][1] +'$'+clusterSentenceIndex[i][0]+'\n'
#			outputIndexFile.write(line)
#	outputIndexFile.close()		

## Consolidate into different clusterd
	
	cluster_to_sentence_dict = defaultdict(list)
	for each_line in clusterSentenceIndex:
		cluster,sentence=each_line
		if cluster in cluster_to_sentence_dict:
			cluster_to_sentence_dict[cluster].append(sentence)
		else:
			cluster_to_sentence_dict[cluster] = [sentence]

	return cluster_to_sentence_dict		


def removeStopWordsandStemming(sentence):
	processed_sentence = []
	for word in sentence:
		if word not in stopWordDict:
			processed_sentence.append(st.stemWord(word))

	return processed_sentence		
#

def cosine_similarity(sent1,sent2,idf_scores):
	cosine_sim_sum = 0.0
	set_sent1 = set(removeStopWordsandStemming(sent1.split()))
	set_sent2 = set(removeStopWordsandStemming(sent2.split()))
	for word in set_sent1:
		if (word in set_sent2) and (word in idf_scores):
			cosine_sim_sum += (sent1.count(word)*sent2.count(word)*idf_scores[word]*idf_scores[word])

	root_sum_sent1 = 0
	root_sum_sent2 = 0
	for word in set_sent1:
		if word in idf_scores:
			root_sum_sent1 += ((sent1.count(word)*idf_scores[word])**2)

	for word in set_sent2:
		if word in idf_scores:
			root_sum_sent2 += ((sent2.count(word)*idf_scores[word])**2)

	root_sum_sent1 = math.sqrt(root_sum_sent1)
	root_sum_sent2 = math.sqrt(root_sum_sent2)
	if (root_sum_sent2==0) or (root_sum_sent1==0):
		return 0
	else:	
		return ((cosine_sim_sum)/(root_sum_sent1*root_sum_sent2))

def calculateSimilarityWithSummary(sentence,summary,idf_scores):
	Summary_similarity = 0
	for i in summary:
		if len(i)>1:
			Summary_similarity += cosine_similarity(sentence,i,idf_scores)

	return Summary_similarity

def calculaterSimilarityWithCorpus(sentence,cluster_to_sentences_dict,idf_scores):
	corpus_similarity = 0
	for cluster in cluster_to_sentences_dict:
		current_cluster = cluster_to_sentences_dict[cluster]
		for each_sentence in current_cluster:
			if len(each_sentence)>1:
				corpus_similarity += cosine_similarity(each_sentence,sentence,idf_scores)

	return corpus_similarity		

def getTotalSenteces(folder_name):
	filename = "../../Temp/SentencesToCluster.txt"
	fp = open(filename,'r')
	text = fp.readlines()
	return len(text)

def getDiversity(total_sentences,summary,cluster_to_sentences_dict,idf_scores):
	diversity_measure = 0
	for cluster in cluster_to_sentences_dict:
		current_cluster = cluster_to_sentences_dict[cluster]
		intersection_set = set(summary).intersection(set(current_cluster))
		cluster_diversity = 0
		for sentence in intersection_set:
			cluster_diversity += (calculaterSimilarityWithCorpus(sentence,cluster_to_sentences_dict,idf_scores)*1.0)/total_sentences

		diversity_measure += math.sqrt(cluster_diversity)

	return diversity_measure		

def getCoverage(summary,total_sentences,cluster_to_sentences_dict,idf_scores,alpha):
	covereage_measure=0
	for cluster in cluster_to_sentences_dict:
		current_cluster = cluster_to_sentences_dict[cluster]
		for each_sentence in current_cluster:
			if len(each_sentence)>1:
				Summary_similarity = calculateSimilarityWithSummary(each_sentence,summary,idf_scores)
				corpus_similarity = calculaterSimilarityWithCorpus(each_sentence,cluster_to_sentences_dict,idf_scores)
				covereage_measure += min(Summary_similarity,((alpha*1.0*corpus_similarity)/total_sentences))

	return covereage_measure		

def writeToFile(Summary,folder_name):
	filename = "../../Temp/Summary_"+folder_name+".txt"
	outfile = open(filename,'w')
	for line in Summary:
		outfile.write(line+'\n')
	outfile.close()

def extractSummary(Summary,lamda,alpha,current_size,total_sentences,cluster_to_sentences_dict,idf_scores):
	current_sentence = ""
	current_score = 0
	max_sentence = ""
	max_score = 0
	check_flag = 0

	for cluster in cluster_to_sentences_dict:
		current_cluster = cluster_to_sentences_dict[cluster]
		for each_sentence in current_cluster:
			if (each_sentence not in Summary) and (current_size+len(each_sentence)<665):
				check_flag = 1
				current_summary = deepcopy(Summary)
				current_summary.append(each_sentence)

				coverage = getCoverage(current_summary,total_sentences,cluster_to_sentences_dict,idf_scores,alpha)

				diversity = getDiversity(total_sentences,current_summary,cluster_to_sentences_dict,idf_scores)

				current_score = coverage + (lamda*diversity)
				current_sentence = each_sentence
				check_cluster = cluster

				#print current_score
				#print current_sentence
				if current_score>max_score:
					max_score = current_score
					max_sentence = current_sentence

	Summary.append(max_sentence)
	current_size+=len(max_sentence)
	return Summary,current_size,check_flag		

def runDocumentSummarization(folder_name):

	# for each cluster segmentint corpus
	corpus = extractDocumentCorpus(folder_name)

	# extracting idf scores for this cluster of docs
	idf_scores = generateInverseDocFrequency(corpus)

	# Create the clustering Input File
	generateClusterInputFile(corpus)

	# Creating individual Dictionaries for each cluster
	cluster_to_sentences_dict = clusterSentencesandConsolidate()

	Summary = []
	current_length = 0 
	current_size = 0
	lamda = 6
	alpha = 15
	total_sentences = getTotalSenteces(folder_name)

	while 1:
		Summary,current_size,flag = extractSummary(Summary,lamda,alpha,current_size,total_sentences,cluster_to_sentences_dict,idf_scores)
		if flag == 0:
			break

	writeToFile(Summary,folder_name)

	cmd = "java -cp ../../C_Rouge/C_ROUGE.jar executiverouge.C_ROUGE ../../Temp/Summary_"+folder_name+".txt " + "../Test_Summaries/"+folder_name+"/ 1 B R"
	rouge_r = subprocess.check_output(cmd,shell=True)
#	rouge_r = rouge_r.replace('\n','')
#	Rouge_R_avg.append(float(rouge_r))
	cmd = "java -cp ../../C_Rouge/C_ROUGE.jar executiverouge.C_ROUGE ../../Temp/Summary_"+folder_name+".txt " + "../Test_Summaries/"+folder_name+"/ 1 B F"
	rouge_f = subprocess.check_output(cmd,shell=True)
#	rouge_f = rouge_f.replace('\n','')
#	Rouge_F_avg.append(float(rouge_f))
#	main_output_file = open('../../Final_Output.txt','a')
#	main_output_file.write(str(folder_name)+"\t"+str(rouge_r)+"\t"+str(rouge_f)+"\n")
#	main_output_file.close()

datasetFolder = 'DUC-2004/Cluster_of_Docs'
os.chdir(datasetFolder)

folder_list = os.listdir('.')

for folder in folder_list:
	runDocumentSummarization(folder)
	print folder
	break