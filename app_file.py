

import numpy as np
import pickle
import pandas as pd
import streamlit as st
from PIL import Image
from bs4 import BeautifulSoup
import re
import math
import nltk
from Summary.tf_idf import run_summarization_tf_idf



def decontracted(phrase):
	# specific
	phrase = re.sub(r"won't", "will not", phrase)
	phrase = re.sub(r"can\'t", "can not", phrase)

	# general
	phrase = re.sub(r"n\'t", " not", phrase)
	phrase = re.sub(r"\'re", " are", phrase)
	phrase = re.sub(r"\'s", " is", phrase)
	phrase = re.sub(r"\'d", " would", phrase)
	phrase = re.sub(r"\'ll", " will", phrase)
	phrase = re.sub(r"\'t", " not", phrase)
	phrase = re.sub(r"\'ve", " have", phrase)
	phrase = re.sub(r"\'m", " am", phrase)
	return phrase

stopwords= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
			"you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
			'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
			'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
			'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
			'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
			'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
			'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
			'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
			'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
			's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
			've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
			"hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
			"mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
			'won', "won't", 'wouldn', "wouldn't"])



def clean_text(sentence):
	sentence = re.sub(r"http\S+", "", sentence)
	sentence = BeautifulSoup(sentence, 'lxml').get_text()
	sentence = decontracted(sentence)
	sentence = re.sub("\S*\d\S*", "", sentence).strip()
	sentence = re.sub('[^A-Za-z]+', ' ', sentence)
	sentence = ' '.join(e.lower() for e in sentence.split() if e.lower() not in stopwords)
	return sentence.strip()

# Load the model and CountVectorizer object 
filename = 'model_new.pkl'
model = pickle.load(open(filename, 'rb'))
vect= pickle.load(open('vectorizer_new.pkl','rb'))



def clean(message):
	review_text= decontracted(message)
	return review_text

def predict(message):
	review_text= decontracted(message)
	review_text= clean_text(message)
	test_vect  = vect.transform(([review_text]))	
	prediction=model.predict(test_vect)
	return prediction

def main():
	st.title("Sentiment Analysis of Food Reviews")
	html_temp = """
	<div style="background-color: red;padding:5px">
	<h2 style="color:white;text-align:center;"> Machine Learning App  </h2>
	</div>
        """

	nltk.download("punkt")
	nltk.download("stopwords")

	
	st.markdown(html_temp, unsafe_allow_html=True)
	st.subheader("Enter your review below")
	message = st.text_area(label="Review")
	text = clean(message)

	if st.checkbox("Summarize"):
                summary = run_summarization_tf_idf(text)
                st.subheader("Summary using TF-IDF Algorithm")
                if summary:
                        st.success(summary)
                else:
                        st.error("This particular review cannot be summarized using TF-IDF")
	
	if st.checkbox("Predict"):
                result = predict(message)
                st.subheader("Prediction")
                if result:
                        st.success("Review with positive sentiment")
                else:
                        st.error("Review with a negative sentiment")

if __name__ =='__main__':
	main()
