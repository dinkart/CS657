#pip install pickle5
#pip install googletrans==4.0.0-rc1

import streamlit as st
import numpy as np
import requests
from PIL import Image
import pickle5 as pickle
import numpy as np

try:
	#Load Hindi-Stopwords
	f = open("final_stopwords.txt", encoding = "UTF-8")
	stopWords = f.read().split("\n")	

	#Load All Data
	paras_h = open("paras_hindi.txt", encoding = "UTF-8").read().split("\n\n")
	paras_g = open("paras_gujrati.txt", encoding = "UTF-8").read().split("\n\n")
	paras_e = open("paras_english.txt", encoding = "UTF-8").read().split("\n\n")
	slokas = open("slokas.txt", encoding = "UTF-8").read().split("\n\n")
	with open('synonyms.pkl', 'rb') as f:
		syns = pickle.load(f)	

	#Punctuations
	punctuations=["।",";",",",":","!",'"',"?",":-","-","{","(","}",")","_","०","S","―","=","[","]","......",":-",".","॥",'”',"|","“","'"]	

	#Stemming Hindi Words
	suffixes = {
	    1: ["ो", "े", "ू", "ु", "ी", "ि", "ा"],
	    2: ["कर", "ाओ", "िए", "ाई", "ाए", "ने", "नी", "ना", "ते", "ीं", "ती", "ता", "ाँ", "ां", "ों", "ें"],
	    3: ["ाकर", "ाइए", "ाईं", "ाया", "ेगी", "ेगा", "ोगी", "ोगे", "ाने", "ाना", "ाते", "ाती", "ाता", "तीं", "ाओं", "ाएं", "ुओं", "ुएं", "ुआं"],
	    4: ["ाएगी", "ाएगा", "ाओगी", "ाओगे", "एंगी", "ेंगी", "एंगे", "ेंगे", "ूंगी", "ूंगा", "ातीं", "नाओं", "नाएं", "ताओं", "ताएं", "ियाँ", "ियों", "ियां"],
	    5: ["ाएंगी", "ाएंगे", "ाऊंगी", "ाऊंगा", "ाइयाँ", "ाइयों", "ाइयां"],
	}	

	def hi_stem(word):
	    for L in 5, 4, 3, 2, 1:
	        if len(word) > L + 1:
	            for suf in suffixes[L]:
	                if word.endswith(suf):
	                    return word[:-L]
	    return word	

	#Tokenize
	def token_stem(string, stopWords):
	    string = "".join([w if w not in punctuations else " " for w in string])  #To remove punctuations
	    tokens = string.split()
	    tokens = [hi_stem(word) for word in tokens if word not in stopWords]
	    tokens = [w for w in tokens if w not in stopWords]
	    return tokens	

	#Creating Posting list
	w_col = {}
	for idx, para in enumerate(paras_h):
	    words = token_stem(para, stopWords)
	    
	    for word in words:
	        if word in w_col.keys():
	            if idx in w_col[word].keys():
	                w_col[word][idx] += 1
	            else:
	                w_col[word][idx] = 1
	        else:
	            temp = {idx : 1}
	            w_col[word] = temp.copy()	

	with open('data.pkl', 'wb') as file:
	    pickle.dump(w_col, file, protocol=pickle.HIGHEST_PROTOCOL)	

	#BM25
	def BM25(query, w_coll, l = 5, b = 0.75, k = 2):
	    q_tokens = token_stem(query, stopWords)
	    lengths = {}
	    N = len(paras_h)
	    avg_len = 0
	    for idx, para in enumerate(paras_h):
	        lengths[idx] = len(para)             #cal no of words of each file
	        avg_len += lengths[idx]
	    avg_len /= N
	    #Calculate idf of each token
	    idf = {}
	    for word in np.unique(q_tokens):
	        if word in w_coll.keys():
	            df = len(w_coll[word].keys())
	        else:
	            df = 0
	        idf[word] = np.log((N - df + 0.5) / (df + 0.5))
	    score = {}
	    for idx, para in enumerate(paras_h):
	        s = 0
	        for word in np.unique(q_tokens):
	            tf = 0
	            if word in w_coll.keys() and idx in w_coll[word].keys():
	                tf = w_coll[word][idx]
	            s += idf[word] * (tf * (k + 1)) / (k*(1 - b + b*lengths[idx]/avg_len) + tf)
	        score[idx] = s
	    return sorted(score, key = score.get, reverse=True)[:l]	

	#Translator
	from googletrans import Translator	

	def get_translation(data, dest):
	    translator = Translator()
	    text = translator.translate(data, dest).text
	    return text	

	#Final Search Function
	def search(query, w_col, lang):
	    query = get_translation(query, "hi")
	    tokens = token_stem(query, stopWords)
	    for idx, token in enumerate(tokens):
	#         if token not in w_col:
	        if token in syns.keys():
	            tokens[idx] += " " + " ".join([w for w in syns[token]])
	    query = " ".join(tokens)
	    indxs = BM25(query, w_col)
	    print(indxs)
	    if lang == 'g':
	        return [f'{slokas[idx]}\n\n{paras_g[idx]}' for idx in indxs]
	    elif lang == 'e':
	        return [f'{slokas[idx]}\n\n{paras_e[idx]}' for idx in indxs]
	    else:
	        return [f'{slokas[idx]}\n\n{paras_h[idx]}' for idx in indxs]	

	#Query search
	#query = "women, courage, diet, modesty"
	#search(query, w_col, "e")	
	

	menu = ["Home","English","Hindi","Gujarati"]
	choice = st.sidebar.radio("Query Language Selection",menu)	

	col1, col2, col3 = st.columns(3)
	with col1:
		st.write(' ')
	with col2:
		st.image("logo1.png")
	with col3:
		st.write(' ')	
	

	if choice == "Home":	

		# Nav  Search Form
		with st.form(key='searchform'):
			search_term = st.text_input("")
			col1, col2, col3,col4,col5 = st.columns(5)
			with col3:
				submit_search = st.form_submit_button(label='Search')
		genre = st.radio(
     	"What's your preferred output languge preference of retrieved documents?",
     	('English', 'Hindi', 'Gujarati'))
		if genre=='English':
			out=search(search_term, w_col, "e")
			st.write('............................................................................................................................................................................')
			st.write('Document 1:')
			st.write(out[0])
			st.write('............................................................................................................................................................................')
			st.write('Document 2:')
			st.write(out[1])
			st.write('............................................................................................................................................................................')
			st.write('Document 3:')
			st.write(out[2])
			st.write('............................................................................................................................................................................')
			st.write('Document 4:')
			st.write(out[3])
			st.write('............................................................................................................................................................................')
			st.write('Document 5:')
			st.write(out[4])
			st.write('............................................................................................................................................................................')

		elif genre=='Hindi':
			out=search(search_term, w_col, "h")
			st.write('............................................................................................................................................................................')
			st.write('Document 1:')
			st.write(out[0])
			st.write('............................................................................................................................................................................')
			st.write('Document 2:')
			st.write(out[1])
			st.write('............................................................................................................................................................................')
			st.write('Document 3:')
			st.write(out[2])
			st.write('............................................................................................................................................................................')
			st.write('Document 4:')
			st.write(out[3])
			st.write('............................................................................................................................................................................')
			st.write('Document 5:')
			st.write(out[4])
			st.write('............................................................................................................................................................................')

		elif genre=='Gujarati':
			out=search(search_term, w_col, "g")
			st.write('............................................................................................................................................................................')
			st.write('Document 1:')
			st.write(out[0])
			st.write('............................................................................................................................................................................')
			st.write('Document 2:')
			st.write(out[1])
			st.write('............................................................................................................................................................................')
			st.write('Document 3:')
			st.write(out[2])
			st.write('............................................................................................................................................................................')
			st.write('Document 4:')
			st.write(out[3])
			st.write('............................................................................................................................................................................')
			st.write('Document 5:')
			st.write(out[4])
			st.write('............................................................................................................................................................................')

	elif choice == "English":
		st.subheader("Please enter your query in English")
		with st.form(key='searchform'):
			search_term = st.text_input("")
			col1, col2, col3,col4,col5 = st.columns(5)
			with col3:
				submit_search = st.form_submit_button(label='Search')	
		genre = st.radio(
     	"What's your preferred output languge preference of retrieved documents?",
     	('English', 'Hindi', 'Gujarati'))
		if genre=='English':
			out=search(search_term, w_col, "e")
			st.write('............................................................................................................................................................................')
			st.write('Document 1:')
			st.write(out[0])
			st.write('............................................................................................................................................................................')
			st.write('Document 2:')
			st.write(out[1])
			st.write('............................................................................................................................................................................')
			st.write('Document 3:')
			st.write(out[2])
			st.write('............................................................................................................................................................................')
			st.write('Document 4:')
			st.write(out[3])
			st.write('............................................................................................................................................................................')
			st.write('Document 5:')
			st.write(out[4])
			st.write('............................................................................................................................................................................')

		elif genre=='Hindi':
			out=search(search_term, w_col, "h")
			st.write('............................................................................................................................................................................')
			st.write('Document 1:')
			st.write(out[0])
			st.write('............................................................................................................................................................................')
			st.write('Document 2:')
			st.write(out[1])
			st.write('............................................................................................................................................................................')
			st.write('Document 3:')
			st.write(out[2])
			st.write('............................................................................................................................................................................')
			st.write('Document 4:')
			st.write(out[3])
			st.write('............................................................................................................................................................................')
			st.write('Document 5:')
			st.write(out[4])
			st.write('............................................................................................................................................................................')

		elif genre=='Gujarati':
			out=search(search_term, w_col, "g")
			st.write('............................................................................................................................................................................')
			st.write('Document 1:')
			st.write(out[0])
			st.write('............................................................................................................................................................................')
			st.write('Document 2:')
			st.write(out[1])
			st.write('............................................................................................................................................................................')
			st.write('Document 3:')
			st.write(out[2])
			st.write('............................................................................................................................................................................')
			st.write('Document 4:')
			st.write(out[3])
			st.write('............................................................................................................................................................................')
			st.write('Document 5:')
			st.write(out[4])
			st.write('............................................................................................................................................................................')		
	elif choice == "Hindi":
		st.subheader("Please enter your query in Hindi")
		with st.form(key='searchform'):
			search_term = st.text_input("")
			col1, col2, col3,col4,col5 = st.columns(5)
			with col3:
				submit_search = st.form_submit_button(label='Search')	
		genre = st.radio(
     	"What's your preferred output languge preference of retrieved documents?",
     	('English', 'Hindi', 'Gujarati'))
		if genre=='English':
			out=search(search_term, w_col, "e")
			st.write('............................................................................................................................................................................')
			st.write('Document 1:')
			st.write(out[0])
			st.write('............................................................................................................................................................................')
			st.write('Document 2:')
			st.write(out[1])
			st.write('............................................................................................................................................................................')
			st.write('Document 3:')
			st.write(out[2])
			st.write('............................................................................................................................................................................')
			st.write('Document 4:')
			st.write(out[3])
			st.write('............................................................................................................................................................................')
			st.write('Document 5:')
			st.write(out[4])
			st.write('............................................................................................................................................................................')

		elif genre=='Hindi':
			out=search(search_term, w_col, "h")
			st.write('............................................................................................................................................................................')
			st.write('Document 1:')
			st.write(out[0])
			st.write('............................................................................................................................................................................')
			st.write('Document 2:')
			st.write(out[1])
			st.write('............................................................................................................................................................................')
			st.write('Document 3:')
			st.write(out[2])
			st.write('............................................................................................................................................................................')
			st.write('Document 4:')
			st.write(out[3])
			st.write('............................................................................................................................................................................')
			st.write('Document 5:')
			st.write(out[4])
			st.write('............................................................................................................................................................................')

		elif genre=='Gujarati':
			out=search(search_term, w_col, "g")
			st.write('............................................................................................................................................................................')
			st.write('Document 1:')
			st.write(out[0])
			st.write('............................................................................................................................................................................')
			st.write('Document 2:')
			st.write(out[1])
			st.write('............................................................................................................................................................................')
			st.write('Document 3:')
			st.write(out[2])
			st.write('............................................................................................................................................................................')
			st.write('Document 4:')
			st.write(out[3])
			st.write('............................................................................................................................................................................')
			st.write('Document 5:')
			st.write(out[4])
			st.write('............................................................................................................................................................................')
	elif choice == "Gujarati":
		st.subheader("Please enter your query in Gujarati")
		with st.form(key='searchform'):
			search_term = st.text_input("")
			col1, col2, col3,col4,col5 = st.columns(5)
			with col3:
				submit_search = st.form_submit_button(label='Search')
		genre = st.radio(
     	"What's your preferred output languge preference of retrieved documents?",
     	('English', 'Hindi', 'Gujarati'))
		if genre=='English':
			out=search(search_term, w_col, "e")
			st.write('............................................................................................................................................................................')
			st.write('Document 1:')
			st.write(out[0])
			st.write('............................................................................................................................................................................')
			st.write('Document 2:')
			st.write(out[1])
			st.write('............................................................................................................................................................................')
			st.write('Document 3:')
			st.write(out[2])
			st.write('............................................................................................................................................................................')
			st.write('Document 4:')
			st.write(out[3])
			st.write('............................................................................................................................................................................')
			st.write('Document 5:')
			st.write(out[4])
			st.write('............................................................................................................................................................................')

		elif genre=='Hindi':
			out=search(search_term, w_col, "h")
			st.write('............................................................................................................................................................................')
			st.write('Document 1:')
			st.write(out[0])
			st.write('............................................................................................................................................................................')
			st.write('Document 2:')
			st.write(out[1])
			st.write('............................................................................................................................................................................')
			st.write('Document 3:')
			st.write(out[2])
			st.write('............................................................................................................................................................................')
			st.write('Document 4:')
			st.write(out[3])
			st.write('............................................................................................................................................................................')
			st.write('Document 5:')
			st.write(out[4])
			st.write('............................................................................................................................................................................')

		elif genre=='Gujarati':
			out=search(search_term, w_col, "g")
			st.write('............................................................................................................................................................................')
			st.write('Document 1:')
			st.write(out[0])
			st.write('............................................................................................................................................................................')
			st.write('Document 2:')
			st.write(out[1])
			st.write('............................................................................................................................................................................')
			st.write('Document 3:')
			st.write(out[2])
			st.write('............................................................................................................................................................................')
			st.write('Document 4:')
			st.write(out[3])
			st.write('............................................................................................................................................................................')
			st.write('Document 5:')
			st.write(out[4])
			st.write('............................................................................................................................................................................')

except:
	pass