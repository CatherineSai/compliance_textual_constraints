import pandas as pd
import numpy as np
from file_paths import *
import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import regex as re
import string
import nltk.data
import re
import yake
from itertools import product
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, sent_tokenize, pos_tag
 

class Key_Phrase:
  def __init__(self, threshold, nlp, reg_relevant_sentences, rea_relevant_sentences):
    self.nlp = nlp
    self.threshold = threshold
    self.df_reg = reg_relevant_sentences
    self.df_rea = rea_relevant_sentences

  def preprocess_text(self, text):
    """calling the other cleaning functions"""
    text = _remove_numbers(text)
    text = _remove_http(text)
    text = _remove_punctuation(text)
    text = _convert_to_lower(text)
    text = _remove_white_space(text)
    text = _remove_short_words(text)
    tokens = _toknizing(text)
    # 2. POS tagging
    pos_map = {'J': 'a', 'N': 'n', 'R': 'r', 'V': 'v'}
    pos_tags_list = pos_tag(tokens)
    #print(pos_tags)
    # 3. Lowercase and lemmatise 
    lemmatiser = WordNetLemmatizer()
    tokens = [lemmatiser.lemmatize(w.lower(), pos=pos_map.get(p[0], 'v')) for w, p in pos_tags_list]
    return tokens

  def _convert_to_lower(text):
    return text.lower()

  def _remove_numbers(text):
    text = re.sub(r'\d+' , '', text)
    return text

  def _remove_http(text):
    text = re.sub("https?:\/\/t.co\/[A-Za-z0-9]*", ' ', text)
    return text

  def _remove_short_words(text):
    text = re.sub(r'\b\w{1,2}\b', '', text)
    return text

  def _remove_punctuation(text):
      punctuations = '''!()[]{};«№»:'"\,`<>./?@=#$-(%^)+&[*_]~'''
      no_punct = ""
      
      for char in text:
          if char not in punctuations:
              no_punct = no_punct + char
      return no_punct
 
  def _remove_white_space(text):
    text = text.strip()
    return text

  def _toknizing(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    ## Remove Stopwords from tokens
    result = [i for i in tokens if not i in stop_words]
    return result
  
  def key_word_extractor (self, text):
    language = "en"
    max_ngram_size = 3
    deduplication_thresold = 0.9
    deduplication_algo = 'seqm'
    windowSize = 1
    numOfKeywords = 2
    kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_thresold, dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords)
    keywords = kw_extractor.extract_keywords(text)
    kw_list = [] 
    for kw, v in keywords:
        kw_list.append(kw)
    return kw_list

  def get_keyword_sim_sent_pairs(self):
    #clean text applying all the text preprocessing functions
    self.df_reg['cleaned_text'] = self.df_reg.text.apply(lambda x: ' '.join(preprocess_text(x)))
    self.df_rea['cleaned_text'] = self.df_rea.text.apply(lambda x: ' '.join(preprocess_text(x)))
    #extract key phrases
    self.df_reg['key_words'] = self.df_reg.cleaned_text.apply(lambda x: key_word_extractor(x))
    self.df_reg['key_words'] = self.df_reg.key_words.apply(lambda x: ' '.join(x))
    self.df_rea['key_words'] = self.df_rea.cleaned_text.apply(lambda x: key_word_extractor(x))
    self.df_rea['key_words'] = self.df_rea.key_words.apply(lambda x: ' '.join(x))
    #combine reg and rea and calculate sim between key phrases 
    result_df = pd.DataFrame()
    # rename columns
    self.df_reg = self.df_reg.rename(columns={"text": "original_sentence_reg", "cleaned_text": "cleaned_text_reg", "key_words": "key_words_reg"})
    self.df_rea = self.df_rea.rename(columns={"text": "original_sentence_rea", "cleaned_text": "cleaned_text_rea", "key_words": "key_words_rea"})
    data = list(product(self.df_reg['original_sentence_reg'], self.df_rea['original_sentence_rea']))
    result_df = result_df.append(data)
    result_df = result_df.rename(columns={0:'original_sentence_reg',1:'original_sentence_rea'})
    result_df = result_df.reset_index(drop=True)
    result_df = pd.merge(result_df, self.df_reg,  how='left', left_on=['original_sentence_reg'], right_on = ['original_sentence_reg'])
    result_df = pd.merge(result_df, self.df_rea,  how='left', left_on=['original_sentence_rea'], right_on = ['original_sentence_rea'])
    for index, row in result_df.iterrows():
        doc_1 = self.nlp(row['key_words_reg'])
        doc_2 = self.nlp(row['key_words_rea'])
        result_df.at[index, 'similarity'] = doc_1.similarity(doc_2)
        # group by reg sentences to get one row per reg sentence; per reg sentence keep row with max similarity score 
        res = result_df[result_df['similarity'] == result_df.groupby('original_sentence_reg')['similarity'].transform('max')]
        # extract the matched sentence-pairs -> for further in depth analysis
        reg_rea_sentence_matches_df = res[res['similarity'] >= self.threshold]
    return reg_rea_sentence_matches_df