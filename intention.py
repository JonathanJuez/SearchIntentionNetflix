# -*- coding: utf-8 -*-
"""
Created on Sat May  8 19:09:53 2021

@author: jonat
"""

import streamlit as st
import pandas as pd
import numpy as np


import nltk, re
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from ipywidgets import interact, interactive, fixed, interact_manual

st.title("Series or movie recommendations")
st.header("Search Intention")
st.text("Type the subject or series")

data = pd.read_csv('./netflix_titles.csv')
data.head()
#data.shape

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

#data["show_id"] = data["show_id"].astype(str)
#data.dtypes

####################################

def preprocess(data_input,column_name,language="english",tolower=True, html_tags=True, stemming = False,stop_words=False):
    
     
    stemmer_preprocces = SnowballStemmer(language=language)#revisar
    stop=set(stopwords.words(language))
    
    
    
    #remove special characters
    temp = data_input[column_name].copy()#.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')#verificar

    
    temp = temp.apply(lambda x: re.sub("[^A-Za-z0-9-\s]+","",str(x))) #elimina carcateres especiales y los convierte en string
    #^ negara todo lo que encuentre en el patron de busqueda de mayusculas y minusculas, tambien buscara los caracteres numericos
    #el + indica las veces en que se podra repetir es decir muchas veces
    
    #print("verificacion de eliminacion de caracteres",temp)
    
    #clean HTML tags
    if html_tags: temp = temp.apply( lambda x: re.sub(re.compile('<.*?>'), '', x) )
        # re sub re compile extraera las etiquetas <> las reemplazara por un espacio en blanco y se las asignara a x 
        #. significa cualquier cosa
        #* una o mas veces
        #? puede ser o no ser varias veces
    
    #tolower todo en minusculas
    if tolower: temp =temp.apply(lambda x: str(x).lower()) #tolower solo aplica cuando son strings
    #tomara todo como string  y lo convertira en minusculas, para que sea mas facil procesar y no tener confusiones 
    #con caracteres en mayusculas
    
    #print("verificacion de minusculas\n",temp)
    
    
    #stemming
    if stemming: temp = temp.apply( lambda x: ' '.join([stemmer_preprocces.stem(x_i) for x_i in x.split(' ')]) )
        #reducira las palabras a su origen reduciendo la dimension de la palabra
        #recorrera x i y los agrupara uno a uno con espacios
        #extraera solo los vocablos o prefijos 
        
    #print("verificacion de stemming\n",temp)
    
    #remove stopwords
    if stop_words: temp = temp.apply( lambda x: ' '.join([y for y in x.split(' ') if y not in stop]) )
        #''.join los devolvera como string separadados, (Y) recorrera y los agrupara uno a uno, if validara si se encuentra
        # en las palabras que no tienen significancia, si se encuentra dentro de las palabras de STOP las eliminara, las que no se encuentra
        # en STOP continuaran en Y, las cuales son las que se agrupan al final
    
    #print("verificacion de stopword\n",temp)
    
    #remove double space
    result = temp.apply(lambda x: re.sub("\s\s+" , " ", x))#remove espacios dobles y deja solo un espacio
    
    #print("verificacion de remove\n",temp)
        
    return result 

####################################

def cosine_similarity(vector1, vector2):
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    return np.dot(vector1, vector2) / (np.sqrt(np.sum(vector1*2)) * np.sqrt(np.sum(vector2*2)))

####################################

def mvp_recommender(user_input):#input data
    df_sample_title = pd.DataFrame([user_input])#convierte a un dataframe los datos entragados por el usuario

    preprocessed_sample = preprocess(df_sample_title,
                                       column_name=[0],\
                                       language='english',\
                                       html_tags=False,\
                                       stemming=True,\
                                       stop_words = True)

    bow_sample = name_bow_vectorizer.transform(preprocessed_sample).toarray()
    #devuelve un vector con una distancia como array para preprocessed sample
    
    #print("BOLSA DE PALABRAS:\n",bow_sample)

    temp_similarity = []#donde se almacenara j es [0 ]y coseno de similitud [1]
    
    #print("SIMILITUD:\n",temp_similarity)

    for j in range(0, name_bow_tfidf.shape[0]): 
        temp_similarity.append([j, cosine_similarity(bow_sample, name_bow_tfidf.toarray()[j]) [0]])
        #creara dos columnas una con los indices y otra con el coseno de similitud
        #[0] regresara solo un numero 

    #print(temp_similarity)
    
    similar_courses = pd.DataFrame(temp_similarity).sort_values(by=[1],ascending=False)#si es ascendente generara error
    #print("VALORES MAS ALTOS DE SIMILITUD\n",similar_courses) #ordenara los valores mas altos de similitud
    
    similar_coursesl = similar_courses.loc[similar_courses[1]>0].head(5)#regresara las 5 primeras
    #similitudes mas peque;as el [1] expecifica que se tomara como referencia el coseno de similitud y se establece la condicion
    #de si sera 0, si es 0 tiene mas similitud 
    #print("VALORES MAS ALTOS DE SIMILITUD ORDENADOS\n",similar_coursesl) 

    result=similar_coursesl.merge(data['title'],left_on=0,right_index=True, how='left')
    #print("MATRIZ DE RESULTADOS CONCATENADOS CON LOS TITULOS\n",result)#se tendra una matrix de similitudes entre el indice de similar courses p
    #en las que como condiciones del merge se tendran como indices el title
    
    
#se ha entrenado el algoritmo y despues se llama el algoritmo original para dar recomendaciones de los titulos
    
    
    return result    

##########################
    

preprocessed_summary= preprocess(data, column_name='description', language='english',html_tags=False,
                                   stemming=True, stop_words = True)
#se llamara la funcion preprocess y se le aplicaran las funciones definidas en ella


name_bow_vectorizer = TfidfVectorizer(norm="l2", analyzer='word', ngram_range=(1,2), max_features = 500)
#norm puede ser l1 o l2, ngramas de 1 a 2
#Genera una matris de entidades de TF ,idf 

name_bow_tfidf = name_bow_vectorizer.fit_transform(preprocessed_summary) #se aplica funcion a los datos procesados
# se extraen vectores 

###############################

#interact(mvp_recommender, user_input='war nations')

mvp_recommender(user_input= 'war')

###################

# Deploy streamlit

firstname = st.text_input("Type serie:")

if st.button("Accept"):
    result = mvp_recommender(user_input=firstname)
    st.write(result)
