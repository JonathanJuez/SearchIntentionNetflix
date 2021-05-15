# -*- coding: utf-8 -*-
"""
Created on Sat May  8 19:09:53 2021

@author: jonat
"""

import streamlit as st

import nltk, re
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords

st.title("Series or movie recommendations")
st.header("Search Intention")
st.text("Type the subject or series")


