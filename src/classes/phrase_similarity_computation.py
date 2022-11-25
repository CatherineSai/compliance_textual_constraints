"""this is an adaption to the original phrase extraction and similarity with different phrases (changed november 2022)"""


import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from file_paths import *


class Phrase_Similarity_Computation:

  def __init__(self):
    self.model = SentenceTransformer('all-MiniLM-L6-v2')
    self.model.max_seq_length = 200

  def get_phrase_similarities(self, df):
    '''calculates all phrase similarity pairs between reg and rea sentence pair 
    input: df with original sent as well as the extracted phrases from both reg and rea
    output: input df enhanced by similarity score for each phrase type'''
    print("Start Phrase_Similarity_Computation")
    #create new columns (so they exist for later algorithms even if empty (e.g. sub_2_2_sim might be empty))
    df["sub_sim"] = np.nan
    df["verb_sim"] = np.nan
    df["cond_sim"] = np.nan
    df["obj_sim"] = np.nan
    # fill empty fields with numpy NaN (makes it easier for if statements to check if both are not empty)
    df = df.replace(r'^\s*$', np.nan, regex=True)
    # negation phrase
    for index, row in df.iterrows():
            df.at[index, 'difference_in_negations'] = abs(row['reg_no_neg']- row['rea_no_neg'])
    # subject phrases
    for index, row in df.iterrows():
            if (pd.notnull(row['reg_sub']) & pd.notnull(row['rea_sub'])):
                embeddings1 = self.model.encode(row['reg_sub'], convert_to_tensor=True)
                embeddings2 = self.model.encode(row['rea_sub'], convert_to_tensor=True)
                cosine_score = util.cos_sim(embeddings1, embeddings2)
                df.at[index, 'sub_sim'] = cosine_score.item()
            else:
                    continue
    # verb phrases
    for index, row in df.iterrows():
            if (pd.notnull(row['reg_verb']) & pd.notnull(row['rea_verb'])):
                embeddings1 = self.model.encode(row['reg_verb'], convert_to_tensor=True)
                embeddings2 = self.model.encode(row['rea_verb'], convert_to_tensor=True)
                cosine_score = util.cos_sim(embeddings1, embeddings2)
                df.at[index, 'verb_sim'] = cosine_score.item()
            else:
                    continue
    # conditional phrases
    for index, row in df.iterrows():
            if (pd.notnull(row['reg_cond']) & pd.notnull(row['rea_cond'])):
                embeddings1 = self.model.encode(row['reg_cond'], convert_to_tensor=True)
                embeddings2 = self.model.encode(row['rea_cond'], convert_to_tensor=True)
                cosine_score = util.cos_sim(embeddings1, embeddings2)
                df.at[index, 'cond_sim'] = cosine_score.item()
            else:
                    continue
    # object phrases
    for index, row in df.iterrows():
            if (pd.notnull(row['reg_obj']) & pd.notnull(row['rea_obj'])):
                embeddings1 = self.model.encode(row['reg_obj'], convert_to_tensor=True)
                embeddings2 = self.model.encode(row['rea_obj'], convert_to_tensor=True)
                cosine_score = util.cos_sim(embeddings1, embeddings2)
                df.at[index, 'obj_sim'] = cosine_score.item()
            else:
                    continue
    return df