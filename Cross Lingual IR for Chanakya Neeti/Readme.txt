Project Title :- Cross Lingual IR of Chanakya Neeti
Group No :- 11

We have created 6 folders namely Datasets,Jupyter Notebooks,Presentation,Project Report,Streamlit Files,Web Scraping.

The files that we have included are :- 3 text files are in Datasets folder namely "chanakya-neeti-hindi","chanakya-neeti-english" and 
					"chanakya-neeti-gujarati",
                                       2 ipynb notebooks are in Jupyter Notebooks folder, 
                                       main2.py is in Streamlit Files folder,
                                       report(pdf) is in Project Report folder and 
                                       presentation(ppt) is in Presentation folder,
                                       Web scrapping file in Web Scrapping folder.

The libraries that we have used to work on the project:-

1):- Numpy :- For performing arithmetic operations(Please install numpy by running the command "pip install numpy", if you do not have it installed)
2):- Pandas :- For Datasets handling(Please install pandas by running the command "pip install pandas", if you do not have it installed)
3):- Pyiwn :- For finding synonyms(Please install pyiwn by running the command "pip install pyiwn", if you do not have it installed)
4):- Streamlit :- We have used this library to make website for better visualization of our project.(Please install streamlit by running the 
                  command "pip install streamlit", if you do not have it installed)
5):- pickle5 :- For storing dictionary
6):- googletrans==4.0.0-rc1 :- Used this library for translating our query, please make sure to have the same version of googletrans in your system.

Make sure that you have latest version of python, so that all the above mentioned libraries/dependancies run smoothly.

Description about each file:-

Datasets :-

1):- chanakya-neeti-hindi.txt : This file contains the shloks,their meanings and their interpretations in hindi language.
2):- chanaky-neeti-english : This file contains meanings of the shloks in english language.
3):- chanakya-neeti-gujarati : This file contains meanings of the shloks in gujarati language.

Code Files:- 

1):- main2.py : This file contains the code that we have written inorder to design the frontend of our website 
2):- Final.ipynb :- This file contains the code of BM25 algorithm and TFIDF algorithm.
3):- Scrapper.ipynb :- This file contains the code of web scrapping that we have done to get lemmatization of words. 
4):- Synonyms.ipynb :- This file contains the code of finding synoyms using pyiwn library

We have used Streamlit to build a website and have also deployed our code. 
Here is the link to our website :- https://share.streamlit.io/divyansh009/ir/main/main2.py
If you wish to look at the project by running our code in a local host, then do the same by running the command "streamlit run main2.py" by making 
"g11-project\Streamlit Files" as the working directory.