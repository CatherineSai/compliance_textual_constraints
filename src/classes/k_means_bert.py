'''Mainly applied after these instructions: https://subscription.packtpub.com/book/data/9781838987312/6/ch06lvl1sec48/nmf-topic-modeling
installation was done with these commands for mac: 
pip3 install torch torchvision torchaudio
pip install transformers
pip install -U sentence-transformers '''

import re
import string
import pandas as pd
import spacy
from sklearn.cluster import KMeans
from nltk.probability import FreqDist
from sentence_transformers import SentenceTransformer


class K_Means_BERT:
  '''uses the sentence transformers model to encode the data 
    and then the Kmeans model to predict the topic it belongs to''' 

  def __init__(self, reg_relevant_sentences, rea_relevant_sentences, nlp, df_topic_models):
    self.nlp = nlp
    self.stopwords = spacy.lang.en.STOP_WORDS
    self.df_master = df_topic_models
    self.df_reg = self.sentence_df(reg_relevant_sentences, 'reg')
    self.df_rea = self.sentence_df(rea_relevant_sentences, 'rea')
    self.preprocessing()
    self.get_BERT_model()
    self.get_kmeans_model()
    self.cluster_words()

  def sentence_df(self, list_relevant_sentences, origin_string):
    '''creates df with column sentences and flag of their origin'''
    df = pd.DataFrame(list_relevant_sentences, columns =['original_sentence_text'])
    df['origin'] = origin_string
    return df

  def reg_rea_sentence_df(self):
    '''joins rea and reg sentences to one df'''
    self.df= self.df_reg.append(self.df_rea, ignore_index = True)

  def preprocessing(self):
    '''reads in the data and preprocesses it: tokenizes the text, 
    puts it into lowercase, and removes the stopwords and punctuations. 
    Then joins the word arrays, since the sentence transformers model takes a string as input.'''
    self.reg_rea_sentence_df()
    self.processed_text_list = []
    punctuations = string.punctuation
    for index, row in self.df.iterrows():
        doc = self.nlp(row['original_sentence_text'])
        doc = [tok.text for tok in doc if (tok.text not in self.stopwords and tok.pos_ != "PUNCT" and tok.pos_ != "SYM" and tok.text != '\n\n\n')]
        doc = [tok.lower() for tok in doc]
        doc = ' '.join(doc)
        self.processed_text_list.append(doc)

  def get_BERT_model(self):
    '''read in the sentence transformers model and use it to encode the documents. 
    Then, read in the DistilBERT-based model, which is smaller than the regular model.'''
    self.model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    self.encoded_data = self.model.encode(self.processed_text_list)

  def get_kmeans_model(self):
    '''creates the KMeans model, initializing it with 12 clusters (analog to topic modeling) 
    and a random state for model reproducibility.'''
    self.km = KMeans(n_clusters=12, random_state=0)
    self.km.fit(self.encoded_data)

  def get_most_frequent_words(self, text):
    word_list = self.nlp(text)
    word_list = [word.text for word in word_list if word.text not in self.stopwords and word.text not in string.punctuation and re.search('[a-zA-Z]', word.text)]
    freq_dist = FreqDist(word_list)
    top_100 = freq_dist.most_common(100)
    top_100 = [word[0] for word in top_100]
    return top_100

  def cluster_words(self, num_clusters=12):
    '''Prints the most common words by cluster'''
    clusters = self.km.labels_.tolist()
    docs = {'text': self.processed_text_list, 'cluster': clusters}
    frame = pd.DataFrame(docs, index = [clusters])
    for cluster in range(0, num_clusters):
        this_cluster_text = frame[frame['cluster'] == cluster]
        all_text = " ".join(this_cluster_text['text'].astype(str))
        top_100 = self.get_most_frequent_words(all_text)
        print(cluster)
        print(top_100)
    return frame

  def predict_clusters_to_df(self):
    '''applies the model to the texts, returning the corresponding clusters as new column in the df (from topic modeling)'''
    for i, text in enumerate(self.df_master.original_sentence):
        cluster = self.km.predict(self.encoded_data)[i]
        self.df_master.at[i, 'kmeans_sbert_cluster'] = cluster
    return self.df_master

  def coherence_evaluation(self):
    '''Coherence is calculated using Pointwise Mutual Information between each pair of words 
    and then averaging it across all pairs.'''



