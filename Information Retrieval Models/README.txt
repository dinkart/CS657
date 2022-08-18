ASSIGNMENT 1(INFORMATION RETRIEVAL)

Dependencies used: python3, ntlk, json, numpy, num2words, math, sys, re, Path, csv, importlib.

NOTE : Please mail to dinkartewary8@gmail.com if you need the data
      NOTE – Please download and extract above zip file and put “english-corpora” folder in current working directory.

Folders and Files attached: 
	-> Q1.py
   	-> Q2a.py
	-> Q2b.py
 	-> Q2c.py
	-> Q4.py

   	-> query_file.txt
	-> ground_truth.txt

	-> Makefile
	-> README.txt

	-> Jupyter-notebook-files     //This folder contains code in .ipynb format as well, if needed.



Description:
I. Q1.py performs tokenization and stemming over all the given data, creates a nested dictionary and saves the output in json format.

Format of JSON File: "Word" : ["File-Name" : count_of_word_in_this_file, ..]

Commands to run: python3 Q_1.py
NOTE – Please excute Q_1.py to generate above json file if not dumped in current working directory.

II. Question-2 consists of 3 python files in which all 3 models have been implemented in respected python files. Each model takes a “query” from command line as an argument and prints relevant documents corresponding to the given Query. 

Commands to run each model: 
	a) python3 Q2a.py "query"
	b) python3 Q2b.py "query"
	c) python3 Q2c.py "query"



III. This Answer contains 2 manually written files with .txt format. File 1(query_file.txt) contains <QueryID, Query> with tab separation. File 2(ground_truth.txt) contains <QueryID, Iteration, DocID(top 10), Relevance> with comma separation. “ground_truth.txt” file has been created using manual documentation search, all 3 model result outputs to create the building block for my implemented models.

IV. Q4.py generates files with .txt format named as Boolean.txt, TFIDF.txt, and BM25.txt with top 5 document search. These files are the result of each query running in each model.

V. Answer_5 contains makefile which runs all 3 models in one click. It also contains README.txt which helps to understand this assignment.

Command to run Makefile:
make run ARGS=”query_file_name.txt”
