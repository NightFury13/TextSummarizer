#Author: Saksham Singhal
from Stemmer import Stemmer
import sys
import re,os
import math
from collections import defaultdict
from copy import deepcopy
import subprocess
from joblib import Parallel, delayed
import multiprocessing

st = Stemmer('english')
pattern=re.compile(r'[\d+\.]*[\d]+|[^\w]+') #pattern to detect numbers (real/integer) non alphanumeric (no underscore)

stopWordDict = defaultdict(int)
stopWordFile = open("./stopwords.txt","r")
for line in stopWordFile:
	stopWordDict[line.strip()]=1


alpha_lambda_outfile = open('Sweep_GridSearch.txt','w')
alpha_lambda_outfile.write('Lambda\tAlpha\tRouge-1 R(avg)\tRouge-1 F(avg)\n')
alpha_lambda_outfile.close()
datasetFolder = 'DUC-2004/Cluster_of_Docs'
os.chdir(datasetFolder)

def extractDocumentCorpus(folder):
	document_to_sentence_corpus = {}
	print folder
	for each_file in os.listdir(folder):
		filename = folder+"/"+each_file
		fileptr = open(filename,'r')
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
		fileptr.close()
		if each_file not in document_to_sentence_corpus:
			document_to_sentence_corpus[each_file] = fileText
	return document_to_sentence_corpus

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

def generateClusterInputFile(corpus,folder_name):
	ClusterInputFile = "../../Temp/SentencesToCluster_"+folder_name+".txt"
	ClusterInputFile_ptr = open(ClusterInputFile,'w')
	for each_doc in corpus:
		current_doc = corpus[each_doc]
		sentences = []
		sentences = current_doc.split('.')
	#	print each_doc
	#	print len(sentences)
		for each_sentence in sentences:
			each_sentence = re.sub(r'[\s]+',' ',each_sentence)
			if len(each_sentence)>1:
				if each_sentence[0]==' ':
					each_sentence = each_sentence[1:]
				ClusterInputFile_ptr.write(each_sentence+'\n')

	ClusterInputFile_ptr.close()

def convertFileToMatFormat(folder_name):
	system_cmd = "perl ../../doc2mat/doc2mat -mystoplist=../../stopwords.txt -nlskip=1 -skipnumeric ../../Temp/SentencesToCluster_"+folder_name+".txt ../../Temp/ClutoInput_"+folder_name+".mat"
	os.system(system_cmd)

def clusterSentences(folder_name):
	ClusterInputFile = "../../Temp/SentencesToCluster_"+folder_name+".txt"
	fileptr = open(ClusterInputFile,'r')
	line_count = len(fileptr.readlines())
	os.system("../../cluto/Linux/vcluster -clmethod=bagglo -sim=cos -niter=100 -seed=45 ../../Temp/ClutoInput_"+folder_name+".mat "+str(line_count/10))
	return line_count/10
	# Here -clmethod can be replaced with 'direct' for conventional k-Means
	# But in general 'rbr', works for efficiently
	# Limiting maximum number of iteration to 100 and setting similarity to 'cosine'
	# seed determines the start of randomness selection points
	# cstype chooses l2 as clustering criterion.

def mapSentencesToCluster(folder_name,NoOfClusters):
	sentenceInputFile = "../../Temp/SentencesToCluster_"+folder_name+".txt"
	sentenceFile = open(sentenceInputFile,'r')
	sentences = sentenceFile.readlines()
	sentenceFile.close()

	for idx in range(len(sentences)):
	    sentences[idx] = sentences[idx].split('\n')[0]

	# Creating cluster number index.

	clusterInputFile = "../../Temp/ClutoInput_"+folder_name+".mat.clustering."+str(NoOfClusters)
	clusterFile = open(clusterInputFile,'r')
	clusterIndex = clusterFile.readlines()
	clusterFile.close()

	for idx in range(len(clusterIndex)):
	    clusterIndex[idx] = clusterIndex[idx].split('\n')[0]

	# Merging the 2 together.

	clusterSentenceIndex = []
	for idx in range(len(clusterIndex)):
	    temp = []
	    temp.append(clusterIndex[idx])
	    temp.append(sentences[idx])

	    clusterSentenceIndex.append(temp)

	clusterSentenceIndex.sort()

	# Printing the sentences into the file.
	outputFile = "../../Temp/"+folder_name+".txt"
	outputIndexFile= open(outputFile,'w')
	for idx in range(len(clusterSentenceIndex)):
		if int(clusterSentenceIndex[idx][0]) >= 0:			# Handles Unneccesary empty sentences
			line = clusterSentenceIndex[idx][1]+'$'+clusterSentenceIndex[idx][0]+'\n'
			outputIndexFile.write(line)
	outputIndexFile.close()

def consolidateClusters(folder_name):
	filename = "../../Temp/"+folder_name+".txt"
	clusterSentencesFile = open(filename,'r')
	cluster_to_sentences_dict = defaultdict(list)
	for line in clusterSentencesFile.readlines():
		lin,cluster = line.split('$')
		cluster = cluster.replace('\n','')
		if cluster in cluster_to_sentences_dict:
			cluster_to_sentences_dict[cluster].append(lin)
		else:
			cluster_to_sentences_dict[cluster] = [lin]

	return cluster_to_sentences_dict	

def removeStopWordsandStemming(sentence):
	processed_sentence = []
	for word in sentence:
		if word not in stopWordDict:
			processed_sentence.append(st.stemWord(word))

	return processed_sentence		

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
	filename = "../../Temp/"+folder_name+".txt"
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

def writeToFile(Summary,folder_name):
	filename = "../../Temp/Summary_"+folder_name+".txt"
	outfile = open(filename,'w')
	for line in Summary:
		outfile.write(line+'\n')
	outfile.close()

def runDocumentSummarization(folder_name,lamda,alpha):
	global Rouge_R_avg
	global Rouge_F_avg

	# for each cluster segmentint corpus
	corpus = extractDocumentCorpus(folder_name)

	# extracting idf scores for this cluster of docs
	idf_scores = generateInverseDocFrequency(corpus)

	# Create the clustering Input File
	generateClusterInputFile(corpus,folder_name)

	# Convert to suitable format for Cluto
	convertFileToMatFormat(folder_name)

	# Create Cluster using CLUTO
	NoOfClusters = clusterSentences(folder_name)

	# Mapping Sentences to corresponding Clusters
	mapSentencesToCluster(folder_name,NoOfClusters)

	# Creating individual Dictionaries for each cluster
	cluster_to_sentences_dict = consolidateClusters(folder_name)

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

	# Write the Output Summary to file
	writeToFile(Summary,folder_name)

	cmd = "java -cp ../../C_Rouge/C_ROUGE.jar executiverouge.C_ROUGE ../../Temp/Summary_"+folder_name+".txt " + "../Test_Summaries/"+folder_name+"/ 1 B R"
	rouge_r = subprocess.check_output(cmd,shell=True)
	rouge_r = rouge_r.replace('\n','')
	Rouge_R_avg.append(float(rouge_r))
	cmd = "java -cp ../../C_Rouge/C_ROUGE.jar executiverouge.C_ROUGE ../../Temp/Summary_"+folder_name+".txt " + "../Test_Summaries/"+folder_name+"/ 1 B F"
	rouge_f = subprocess.check_output(cmd,shell=True)
	rouge_f = rouge_f.replace('\n','')
	Rouge_F_avg.append(float(rouge_f))
	main_output_file = open('../../Final_Output.txt','a')
	main_output_file.write(str(folder_name)+"\t"+str(rouge_r)+"\t"+str(rouge_f)+"\n")
	main_output_file.close()

numOfCores = multiprocessing.cpu_count()
folder_list = os.listdir('.')


for l in xrange(1,7):
	a=15
	while a<40:
		Rouge_R_avg = []
		Rouge_F_avg = []
		main_output_file = open('Final_Output.txt','w')
		main_output_file.write('ClusterID\tRouge-1 R\tRouge-1 F\n')
		main_output_file.close()
		Parallel(n_jobs = numOfCores)(delayed(runDocumentSummarization)(cluster,l,a) for cluster in folder_list)
		avg_RR = sum(Rouge_R_avg)/len(Rouge_R_avg)
		avg_RF = sum(Rouge_F_avg)/len(Rouge_F_avg)
		alpha_lambda_outfile = open('Sweep_GridSearch.txt','a')
		alpha_lambda_outfile.write(str(l)+"\t"+str(a)+"\t"+str(avg_RR)+"\t"+str(avg_RF)+"\n")
		alpha_lambda_outfile.close()
		os.system("rm -rf ../../Temp/*")
		a+=5