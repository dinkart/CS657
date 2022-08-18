CS657A : INFORMATION RETRIEVAL
ASSIGNMENT 2

NOTE: Codes are path sensitive. Please put all files in a respected folder as instructed below.

DEPENDENCIES USED:
gensim
pandas
pickle
math
numpy
matplotlib
sklearn
torch
transformers


FOLDERS AND FILES ATTACHED:
	1. Q1.py
	3. 	a. Q3a.ipynb
		b. Q3b.ipynb
		c. Q3c.ipynb
	6. README.txt
	8. Makefile

DESCRIPTION:
	1. Answer_1.py 
		a. This file let us find the accuracy of each model in each dimension. 
		b. It contains various functions whose working has already been mentioned in comment sections of code.
		c. After executing the Answer_1.py, it returns 40 .csv files with columns : Word1, Word2, similarity_score, groud_truth_similarity and label.
		d. The files will be saved in the root folder.
		e. Associated data are to be kept in root folder, named hi->{50, 100}(two different folders)->{cbow, fasttext, glove, sg}->{model files} 

	
	3. Answer_3.ipynb
		a. This file let us find all the top 100 unigrams, bigrams, trigrams and quadrigrams of characters and unigrams, bigrams and trigrams of words and syllables.
		b. The above task(b point) could be done by running the third cell in Answer_3.ipynb(commented out cell)
		c. Since there is no specific output format, the output has been saved in .txt files under root directory as "Top_Characters.txt", "Top_Syllable.txt", "Top_Words.txt".
		d. There are approximately 700 million hindi lines in 22GB txt file and It takes ages to process all lines. Hence, 200 million of lines have been processed and result has been saved on this basis.
		e. As the answer gets stable after processing 1 million of lines hence, 200 million of lines are sufficient in respected POV.
		f. There are total 3 txt files as output.
		g. It contains zipfian graphs which can be shown in code itself. The graph can be drawn for each data at the end of each file.
			zipfian distribution - rank of char/word/syllable is inversely proporational to frequency of char/word/syllable i.e. As the frequency of char/word/syllable is higher, the rank of char/word/syllable will be less.
			To plot the zipfian distribution, bar graph from matplotlib have been used. As we can see, the graph goes down eventually as the rank progresses.
		 