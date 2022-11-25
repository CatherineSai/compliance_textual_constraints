"""this is the original phrase similarity based on the possibility that one sentence contains multiple constraints (with multiple sub, verb, etc.)"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from file_paths import *


class Phrase_Similarity_Computation:

  def __init__(self):
    self.model = SentenceTransformer('all-MiniLM-L6-v2')
    self.model.max_seq_length = 200

  def get_phrase_similarities(self, df):
    '''calculates all phrase similarity pairs between reg and rea sentence pair '''
    print("Start Phrase_Similarity_Computation")
    #create new columns (so they exist for later algorithms even if empty (e.g. sub_2_2_sim might be empty))
    df["sub_1_1_sim"] = np.nan
    df["sub_1_2_sim"] = np.nan
    df["sub_2_1_sim"] = np.nan
    df["sub_2_2_sim"] = np.nan
    df["verb_1_1_sim"] = np.nan
    df["verb_1_2_sim"] = np.nan
    df["verb_2_1_sim"] = np.nan
    df["verb_2_2_sim"] = np.nan
    df["obj_1_1_sim"] = np.nan
    df["obj_1_2_sim"] = np.nan
    df["obj_2_1_sim"] = np.nan
    df["obj_2_2_sim"] = np.nan
    # fill empty fields with numpy NaN (makes it easier for if statements to check if both are npt empty)
    df = df.replace(r'^\s*$', np.nan, regex=True)
    # negation phrase
    for index, row in df.iterrows():
            df.at[index, 'difference_in_negations'] = abs(row['number_negations_reg']- row['number_negations_rea'])
    # subject phrases
    for index, row in df.iterrows():
            if (pd.notnull(row['subject_phrase_1_reg']) & pd.notnull(row['subject_phrase_1_rea'])):
                embeddings1 = self.model.encode(row['subject_phrase_1_reg'], convert_to_tensor=True)
                embeddings2 = self.model.encode(row['subject_phrase_1_rea'], convert_to_tensor=True)
                cosine_score = util.pytorch_cos_sim(embeddings1, embeddings2)
                df.at[index, 'sub_1_1_sim'] = cosine_score
            else:
                    continue
    for index, row in df.iterrows():
            if (pd.notnull(row['subject_phrase_1_reg']) & pd.notnull(row['subject_phrase_2_rea'])):
                embeddings1 = self.model.encode(row['subject_phrase_1_reg'], convert_to_tensor=True)
                embeddings2 = self.model.encode(row['subject_phrase_2_rea'], convert_to_tensor=True)
                cosine_score = util.pytorch_cos_sim(embeddings1, embeddings2)
                df.at[index, 'sub_1_2_sim'] = cosine_score
            else:
                    continue
    for index, row in df.iterrows():
            if (pd.notnull(row['subject_phrase_2_reg']) & pd.notnull(row['subject_phrase_1_rea'])):
                embeddings1 = self.model.encode(row['subject_phrase_2_reg'], convert_to_tensor=True)
                embeddings2 = self.model.encode(row['subject_phrase_1_rea'], convert_to_tensor=True)
                cosine_score = util.pytorch_cos_sim(embeddings1, embeddings2)
                df.at[index, 'sub_2_1_sim'] = cosine_score
            else:
                    continue
    for index, row in df.iterrows():
            if (pd.notnull(row['subject_phrase_2_reg']) & pd.notnull(row['subject_phrase_2_rea'])):
                embeddings1 = self.model.encode(row['subject_phrase_2_reg'], convert_to_tensor=True)
                embeddings2 = self.model.encode(row['subject_phrase_2_rea'], convert_to_tensor=True)
                cosine_score = util.pytorch_cos_sim(embeddings1, embeddings2)
                df.at[index, 'sub_2_2_sim'] = cosine_score
            else:
                    continue
    # verb phrases
    for index, row in df.iterrows():
            if (pd.notnull(row['verb_phrase_1_reg']) & pd.notnull(row['verb_phrase_1_rea'])):
                embeddings1 = self.model.encode(row['verb_phrase_1_reg'], convert_to_tensor=True)
                embeddings2 = self.model.encode(row['verb_phrase_1_rea'], convert_to_tensor=True)
                cosine_score = util.pytorch_cos_sim(embeddings1, embeddings2)
                df.at[index, 'verb_1_1_sim'] = cosine_score
            else:
                    continue
    for index, row in df.iterrows():
            if (pd.notnull(row['verb_phrase_1_reg']) & pd.notnull(row['verb_phrase_2_rea'])):
                embeddings1 = self.model.encode(row['verb_phrase_1_reg'], convert_to_tensor=True)
                embeddings2 = self.model.encode(row['verb_phrase_2_rea'], convert_to_tensor=True)
                cosine_score = util.pytorch_cos_sim(embeddings1, embeddings2)
                df.at[index, 'verb_1_2_sim'] = cosine_score
            else:
                    continue
    for index, row in df.iterrows():
            if (pd.notnull(row['verb_phrase_2_reg']) & pd.notnull(row['verb_phrase_1_rea'])):
                embeddings1 = self.model.encode(row['verb_phrase_2_reg'], convert_to_tensor=True)
                embeddings2 = self.model.encode(row['verb_phrase_1_rea'], convert_to_tensor=True)
                cosine_score = util.pytorch_cos_sim(embeddings1, embeddings2)
                df.at[index, 'verb_2_1_sim'] = cosine_score
            else:
                    continue
    for index, row in df.iterrows():
            if (pd.notnull(row['verb_phrase_2_reg']) & pd.notnull(row['verb_phrase_2_rea'])):
                embeddings1 = self.model.encode(row['verb_phrase_2_reg'], convert_to_tensor=True)
                embeddings2 = self.model.encode(row['verb_phrase_2_rea'], convert_to_tensor=True)
                cosine_score = util.pytorch_cos_sim(embeddings1, embeddings2)
                df.at[index, 'verb_2_2_sim'] = cosine_score
            else:
                    continue
    # object phrases
    for index, row in df.iterrows():
            if (pd.notnull(row['object_phrase_1_reg']) & pd.notnull(row['object_phrase_1_rea'])):
                embeddings1 = self.model.encode(row['object_phrase_1_reg'], convert_to_tensor=True)
                embeddings2 = self.model.encode(row['object_phrase_1_rea'], convert_to_tensor=True)
                cosine_score = util.pytorch_cos_sim(embeddings1, embeddings2)
                df.at[index, 'obj_1_1_sim'] = cosine_score
            else:
                    continue
    for index, row in df.iterrows():
            if (pd.notnull(row['object_phrase_1_reg']) & pd.notnull(row['object_phrase_2_rea'])):
                embeddings1 = self.model.encode(row['object_phrase_1_reg'], convert_to_tensor=True)
                embeddings2 = self.model.encode(row['object_phrase_2_rea'], convert_to_tensor=True)
                cosine_score = util.pytorch_cos_sim(embeddings1, embeddings2)
                df.at[index, 'obj_1_2_sim'] = cosine_score
            else:
                    continue
    for index, row in df.iterrows():
            if (pd.notnull(row['object_phrase_2_reg']) & pd.notnull(row['object_phrase_1_rea'])):
                embeddings1 = self.model.encode(row['object_phrase_2_reg'], convert_to_tensor=True)
                embeddings2 = self.model.encode(row['object_phrase_1_rea'], convert_to_tensor=True)
                cosine_score = util.pytorch_cos_sim(embeddings1, embeddings2)
                df.at[index, 'obj_2_1_sim'] = cosine_score
            else:
                    continue
    for index, row in df.iterrows():
            if (pd.notnull(row['object_phrase_2_reg']) & pd.notnull(row['object_phrase_2_rea'])):
                embeddings1 = self.model.encode(row['object_phrase_2_reg'], convert_to_tensor=True)
                embeddings2 = self.model.encode(row['object_phrase_2_rea'], convert_to_tensor=True)
                cosine_score = util.pytorch_cos_sim(embeddings1, embeddings2)
                df.at[index, 'obj_2_2_sim'] = cosine_score
            else:
                    continue
    return df