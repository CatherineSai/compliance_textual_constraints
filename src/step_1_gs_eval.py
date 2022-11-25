
import numpy as np
import pandas as pd
from file_paths import *

# read in data
df_gs = pd.read_excel(join(INPUT_DIRECTORY, "goldstandard_88.xlsx"), index_col=0) 
df_sb = pd.read_excel(join(INPUT_DIRECTORY, "df_bert_sent_pairs.xlsx"), index_col=0) 

# clean method input & create rank column
df_sb['sbert_sim_score'] = df_sb['sbert_sim_score'].str[7:]
df_sb['sbert_sim_score'] = df_sb['sbert_sim_score'].str[:-1]
df_sb = df_sb.astype({"sbert_sim_score": float})
df_sb["rank"] = df_sb.groupby("original_sentence_reg")["sbert_sim_score"].rank(ascending=False)

# merging results and gs
df_gs_enhanced = pd.merge(df_gs, df_sb,  how='left', left_on=['original_sentence_reg','original_sentence_rea'], right_on = ['original_sentence_reg','original_sentence_rea'])
# create extra column counting how often reg was matched and merge with gs df
df_count = df_sb[['original_sentence_reg','original_sentence_rea']].groupby(['original_sentence_reg']).count()
df_gs_enhanced = pd.merge(df_gs_enhanced, df_count,  how='left', left_on=['original_sentence_reg'], right_on=['original_sentence_reg'])
# cleaning/renaming
df_gs_enhanced = df_gs_enhanced.rename(columns={"sbert_sim_score": "sim_score_gs_match", "original_sentence_rea_x": "original_sentence_rea", "original_sentence_rea_y": "no_matches_for_this_reg"})
# average precision (AP)
df_gs_enhanced["AP"] = 1/df_gs_enhanced["rank"]
# cleaning
df_gs_enhanced = df_gs_enhanced.fillna(0)

# sum rank for rank 1-4 
df_rank4 = df_sb.loc[(df_sb['rank'] >= 1) & (df_sb['rank'] <= 4)]
df_gs_enhanced["method_rank_4"] = len(df_rank4. index)
# sum rank for rank 1
df_rank1 = df_sb.loc[df_sb['rank'] == 1]
df_gs_enhanced["method_rank_1"] = len(df_rank1. index)
# mean average precision (MAP)
df_gs_enhanced["MAP"] = df_gs_enhanced["AP"].mean()
# count number gs rank within 4
df_gs_enhanced["No_rank_within_4"] = df_gs_enhanced["rank"][(df_gs_enhanced["rank"] > 0) & (df_gs_enhanced["rank"] < 5)].count()

#save to excel  
pd.DataFrame(df_gs_enhanced).to_excel(join(RESULT_DIRECTORY, "goldstandard_evaluation.xlsx"))  