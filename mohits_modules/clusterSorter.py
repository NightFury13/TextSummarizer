#################################################
#                                               #
# Author : Mohit Jain                           #
# Email  : develop13mohit@gmail.com             #
#                                               #
#################################################

# Desc : Takes input two files, the sentence dump & clustering output, and creates a sorted index-hash for each of the sentences.
# Usage : $> clusterSorter.py sentenceDump.txt clusterOutput.mat 

# Libraries used.
import sys

# Tackling the sentences dump.

sentenceFile = open(sys.argv[1])
sentences = sentenceFile.readlines()
sentenceFile.close()

for idx in range(len(sentences)):
    sentences[idx] = sentences[idx].split('\n')[0]

# Creating cluster number index.

clusterFile = open(sys.argv[2])
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
outputIndexFile= open('sentence-cluster-sorted-index.txt','w')
for idx in range(len(clusterSentenceIndex)):
    line = clusterSentenceIndex[idx][1]+'$'+clusterSentenceIndex[idx][0]+'\n'
    outputIndexFile.write(line)
outputIndexFile.close()
