'''
### 4. check completness
for the clusters determined with topic modeling a) and k-means sbert clustering b)

a1) caluclate similarity within each cluster between all origin 'reg' and all origin 'rea'

a2) pick most similar pairs above threshold

a3) fill for each sentence the column "match_found" with "yes" (i), "no_missing" (ii), or "no_stricter" (iii)

    --> i) if a constraint from regulation has a pair partner from realization above threshold similarity, it is considered to be addressed in the realization 
    
    --> ii) if a constraint from regulation has no pair partner from realization above threshold similarity, it is considered to be missing in the realization 
    
    --> iii) if a constraint from realization has no pair partner from regulation above threshold similarity, it is considered to be an addition/spezification by the realization, this will be taken into account for deviation calculation (is the realization stricter than necessary?)
    
b1) analog a

b2) analog a

b3) analog a
'''
import pandas as pd
from itertools import product
from file_paths import *


class Constraint_Existence_Check:
  ''' checks completness for the clusters determined with topic modeling and k-means sbert clustering ''' 

  def __init__(self, nlp, df, gamma):
    self.nlp = nlp
    self.df = df
    self.threshold = gamma
    self.topic_model_similarities_df = self.cluster_sentence_pair_similarties('GSDM_topic_model', 'df_topic_model_sentence_similarity.xlsx')
    self.kmeans_bert_similarities_df = self.cluster_sentence_pair_similarties('kmeans_sbert_cluster', 'df_kmeans_bert_sentence_similarity.xlsx')

  def cluster_sentence_pair_similarties(self, grouping_type_column, filename):
    '''creating a df of all sentence combinations reg and rea within a cluster and calculating the similarities (with default spacy (embedding word2vec + cosine sim)) 
    between all resulting reg-rea pairs'''
    df_pairs = self.df[self.df['origin']=='reg']
    df_pairs = df_pairs.sort_values([grouping_type_column], ascending=True)
    df_pairs = df_pairs.rename(columns={"original_sentence": "original_sentence_reg"})
    result_df = pd.DataFrame()
    for cluster in df_pairs[grouping_type_column].unique(): 
        df_rea = self.df[(self.df['origin']=='rea') & (self.df[grouping_type_column]== cluster)]
        df_rea = df_rea.rename(columns={"original_sentence": "original_sentence_rea"})
        df_reg = df_pairs[df_pairs[grouping_type_column]== cluster]
        data = list(product(df_reg['original_sentence_reg'], df_rea['original_sentence_rea']))
        result_df = result_df.append(data)
    result_df = result_df.rename(columns={0:'original_sentence_reg',1:'original_sentence_rea'})
    result_df = result_df.reset_index(drop=True)
    for index, row in result_df.iterrows():
        doc_1 = self.nlp(row['original_sentence_reg'])
        doc_2 = self.nlp(row['original_sentence_rea'])
        #added setup, because Spacy constructs sentence embedding by averaging the word embeddings
        doc_1_no_stopwords = self.nlp(' '.join([str(t) for t in doc_1 if not t.is_stop]))
        doc_2_no_stopwords = self.nlp(' '.join([str(t) for t in doc_2 if not t.is_stop]))
        result_df.at[index, 'similarity'] = doc_1_no_stopwords.similarity(doc_2_no_stopwords)
        #result_df.at[index, 'similarity'] = doc_1.similarity(doc_2)
    result_df.to_excel(join(INTERMEDIATE_DIRECTORY, filename))  
    return result_df

  def split_results_by_similarity(self, result_df, filename_no_match_reg, filename_no_match_rea):
    '''first groups the results df into a df with one row per reg_sentence and it's correcponding highest rea sentence match;
    then splits the resulting df depending similarity value over or under threshold; extracts rea sentences that were not matched above threshold'''
    # group by reg sentences to get one row per reg sentence; per reg sentence keep row with max similarity score 
    res = result_df[result_df['similarity'] == result_df.groupby('original_sentence_reg')['similarity'].transform('max')]
    # extract the matched sentence-pairs -> for further in depth analysis
    reg_rea_sentence_matches_df = res[res['similarity'] >= self.threshold]
    # extract the reg sentences without sufficient match --> for stakeholders to check if it's fine that these are missing 
    reg_sent_without_match_df = res.drop(['original_sentence_rea'], axis=1)
    reg_sent_without_match_df = reg_sent_without_match_df[reg_sent_without_match_df['similarity'] < self.threshold]
    reg_sent_without_match_df.to_excel(join(RESULT_DIRECTORY, filename_no_match_reg)) 
    # check if all sentences from rea where matched, if sentences where not matched, extract them to df --> here potentially rea more strict than necessary
    rea_sent_without_match_df = result_df.drop(['original_sentence_reg'], axis=1)
    rea_sent_without_match_df = rea_sent_without_match_df[rea_sent_without_match_df['similarity'] == rea_sent_without_match_df.groupby('original_sentence_rea')['similarity'].transform('max')]
    rea_sent_without_match_df = rea_sent_without_match_df[rea_sent_without_match_df['similarity'] < self.threshold]
    rea_sent_without_match_df.to_excel(join(RESULT_DIRECTORY, filename_no_match_rea)) 
    return reg_rea_sentence_matches_df
