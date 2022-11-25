
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from file_paths import *



class S_Bert_Sentence_Pairs:
  def __init__(self, gamma):
    self.model = SentenceTransformer('all-MiniLM-L6-v2')
    self.model.max_seq_length = 200
    self.threshold = gamma

  def get_bert_sim_sent_pairs(self, sentence_list_reg, sentence_list_rea):
    '''creates the embeddings of all sentences and calculates the cosine similarity between all sentences; 
    output: a dataframe with all reg sentences and their highest similarity rea sentence'''
    #Compute embedding for both lists
    embeddings1 = self.model.encode(sentence_list_reg, convert_to_tensor=True)
    embeddings2 = self.model.encode(sentence_list_rea, convert_to_tensor=True)
    #Compute cosine-similarits
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
    #create df of best score results
    df_sent_pairs = pd.DataFrame(columns=['original_sentence_reg','original_sentence_rea','sbert_sim_score'])
  
    # returns only best rea match for each reg constraint
    for i in range(len(sentence_list_reg)):
        highest_score = 0
        for j in range(len(sentence_list_rea)):
            if ((cosine_scores[i][j]) > highest_score):
                highest_score = (cosine_scores[i][j])
                best_fit = j
        df_sent_pairs.at[i, 'original_sentence_reg'] = sentence_list_reg[i]
        df_sent_pairs.at[i, 'original_sentence_rea'] = sentence_list_rea[best_fit]
        df_sent_pairs.at[i, 'sbert_sim_score'] = highest_score
    df_sent_pairs = df_sent_pairs[df_sent_pairs['sbert_sim_score'] >= self.threshold]
    return df_sent_pairs
    '''
    # if you want to extract all matches (not only best match) - execute the following code instead
    k = 0
    for i in range(len(sentence_list_reg)):
        for j in range(len(sentence_list_rea)):
            if ((cosine_scores[i][j]) >= self.threshold):
              df_sent_pairs.at[k, 'original_sentence_reg'] = sentence_list_reg[i]
              df_sent_pairs.at[k, 'original_sentence_rea'] = sentence_list_rea[j]
              df_sent_pairs.at[k, 'sbert_sim_score'] = cosine_scores[i][j]
              k = k+1
    return df_sent_pairs
    '''
    
  def get_unmapped_reg_sentences(self, df_sent_pairs, sentence_list_reg):
    '''extracts the reg contranits without suffient partner sentence into extra dfs as unmapped - need to be reviewed manually'''
    # extract the reg sentences without sufficient match --> for stakeholders to check if it's fine that these are missing 
    df_reg = pd.DataFrame (sentence_list_reg, columns = ['original_sentence_reg'])
    for indexa, rowa in df_reg.iterrows():
        for indexb, rowb in df_sent_pairs.iterrows():
          if (rowa['original_sentence_reg'] == rowb['original_sentence_reg']):
            df_reg.at[indexa, 'mapped'] = 1
    df_reg_unmapped = df_reg[df_reg['mapped'] != 1]
    # save case 2 constraints to results
    df_reg_unmapped.to_excel(join(RESULT_DIRECTORY, "df_reg_unmapped.xlsx")) 
    return df_reg_unmapped['original_sentence_reg'].count()

  def get_unmapped_rea_sentences(self, df_sent_pairs, sentence_list_rea):
    '''extracts the rea contranits without suffient partner sentence into extra dfs as unmapped - need to be reviewed manually'''
    # check if all sentences from rea where matched, if sentences where not matched, extract them to df --> here potentially rea more strict than necessary
    self.df_rea = pd.DataFrame (sentence_list_rea, columns = ['original_sentence_rea'])
    #df_sent_pairs_unique_rea = df_sent_pairs.drop(['original_sentence_reg'], axis=1)
    #df_sent_pairs_unique_rea = df_sent_pairs_unique_rea.groupby('original_sentence_rea')['sbert_sim_score'].transform('max')
    list_sent_pairs_unique_rea = df_sent_pairs['original_sentence_rea'].unique()
    df_sent_pairs_unique_rea = pd.DataFrame (list_sent_pairs_unique_rea, columns = ['original_sentence_rea'])
    for indexa, rowa in self.df_rea.iterrows():
        for indexb, rowb in df_sent_pairs_unique_rea.iterrows():
          if (rowa['original_sentence_rea'] == rowb['original_sentence_rea']):
            self.df_rea.at[indexa, 'mapped'] = 1
    df_rea_unmapped = self.df_rea[self.df_rea['mapped'] != 1]
    # save case 3 constraints to results
    df_rea_unmapped.to_excel(join(RESULT_DIRECTORY, "df_rea_unmapped.xlsx")) 
    return df_rea_unmapped['original_sentence_rea'].count()

  def get_mapped_rea_sentences(self):
    '''count number of rea sent mapped (!= number reg sent mapped)'''
    df_rea_mapped = self.df_rea[self.df_rea['mapped'] == 1]
    return df_rea_mapped['original_sentence_rea'].count()
