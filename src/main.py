## TEXT CLEANING ########################################
# text cleaning was changed to notebook as coreferee gave errors in OOP. As stated in the readme, before running this script:
# run "NEW_preprocessing_optionA _rea.ipynb"
# run "NEW_preprocessing_optionA _rea.ipynb"


# general imports
import os
import coreferee
import re
import xlrd
import spacy
import pandas as pd
# own path/ class imports
from file_paths import *
from classes.text_cleaning import *
from classes.iso_text_cleaning import *
from classes.topic_modeling import *
from classes.k_means_bert import *
from classes.key_phrase import *
from classes.constraint_existence_check import *
from classes.phrase_extraction import *
from classes.phrase_similarity_computation import *
from classes.s_bert_sentence_pairs import *
from classes.legal_s_bert_sentence_pairs import *
from classes.deviation_counter import *


## Application Selection ########################################START
# choose if step 1 OR step 2 should be executed
step_1 = True #only if step 2 = False
step_2 = False #only if step 1 = True
# choose method 
direct_s_bert = False #if True --> no clustering or other means are implemented, all sentences are comapred with each other via S-Bert
legal_s_bert = True #if True --> implementation like S-BERT but based on legal BERT instead of BERT
s_bert_with_kw = False #if True --> step 1 matching based on keywords instead of sent and S-BERT embedding
clustering = False #if True --> 2 approaches calculated: a) topic model + word2vec + cosine sim; b) bert embeddings + kmeans and word2vev + cosine sim
key_phrase = False # if True --> setp one is performed on only key phrases (identified by tfidf), instead of whole sentences
# choose case study
### GDPR adjusted, ISO not!
iso = False #if False --> running with gdpr setup
# choose thresholds:
gamma_s_bert = 0.2 #0.67 #used for sentence mapping 
gamma_grouping = 0.9 #used for sentence mapping in k-means & topic Model approach
gamma_key_phrase = 0.92 #used for key phrase extraction
gamma_one = 0.26 #used for subject phrase mapping
gamma_two = 0.23 #used for verb phrase mapping
gamma_three = 0.2 #used for object phrase mapping
################################################################# END

# Create the nlp object
nlp = spacy.load('en_core_web_trf')
nlp.add_pipe('coreferee') # resolves coreferences
nlp.add_pipe("merge_entities")

"""
### Previous Preprocessing ###
## parse defined lists of constraint signalwords, stopwords ########################### START
def read_defined_lists(directory): 
  '''reads in defined txts of constraint signalwords, stopwords and realization specific formulations as lists
  Input: .txt
  Output: list'''
  try:
    with open(directory) as f:
      defined_list = f.read().splitlines()
  except FileNotFoundError:
      print("Wrong file or file path.")
      quit()
  return defined_list

if iso:
  signalwords = read_defined_lists(ISO_SIGNALWORDS)
  ISMS_words = read_defined_lists(ISO_REA_SPEZIFICATION1)
  top_management_words = read_defined_lists(ISO_REA_SPEZIFICATION2)
else:
  signalwords = read_defined_lists(GDPR_SIGNALWORDS)
  controller_words = read_defined_lists(GDPR_REA_SPEZIFICATION1)
  data_protection_officer_words = read_defined_lists(GDPR_REA_SPEZIFICATION2)
  management_words = read_defined_lists(GDPR_REA_SPEZIFICATION3)

################################################################# END

## parse documents ############################################ START
def read_documents(directory): 
  '''reads in txts of regulatory and realization documents
  Input: multiple .txt (each a document article)
  Output: dictionary with article name as key and article text as value'''
  doc_dict = dict()
  files = os.listdir(directory)
  try:
    for fi in files:
        if fi.endswith('.txt'):
          with open(directory+'/'+fi,'r') as f:
              doc_dict[re.sub('\.txt', '', fi)] = f.read()
  except FileNotFoundError:
    print("Wrong file or file path to dir.")
    quit()
  return doc_dict

# reading the raw .txt text
if iso:
  reg_paragraphs = read_documents(ISO_REGULATION_INPUT_DIRECTORY) 
  rea_paragraphs = read_documents(ISO_REALIZATION_INPUT_DIRECTORY) 

else: 
  reg_paragraphs = read_documents(GDPR_REGULATION_INPUT_DIRECTORY) 
  rea_paragraphs = read_documents(GDPR_REALIZATION_INPUT_DIRECTORY) 
################################################################# END

#Text cleaning
if iso: 
  itc = Iso_Text_Cleaning(nlp, signalwords, ISMS_words, top_management_words)
  # not adjusted to new coref resolution
  reg_relevant_sentences = itc.get_relevant_sentences(reg_paragraphs)
  rea_relevant_sentences = itc.get_relevant_sentences(rea_paragraphs)
else:
  tc = Text_Cleaning(nlp, signalwords, controller_words, data_protection_officer_words, management_words)
  reg_relevant_sentences = tc.text_prep_reg(reg_paragraphs)
  if rea_only_signal:
    rea_relevant_sentences = tc.text_prep_rea(rea_paragraphs)
  else:
    rea_relevant_sentences = tc.text_prep_rea_no_sig_filter(rea_paragraphs)

  #save all input constraints before matching for evaluation purposes  
  pd.DataFrame(reg_relevant_sentences).to_excel(join(INTERMEDIATE_DIRECTORY, "gdpr_reg_relevant_sentences.xlsx"))  
  pd.DataFrame(rea_relevant_sentences).to_excel(join(INTERMEDIATE_DIRECTORY, "gdpr_rea_relevant_sentences.xlsx"))  
"""
if step_1: 
  #read preprocessed reg and read df and take sent column as list
  df_reg_prep = pd.read_excel(join(INTERMEDIATE_DIRECTORY, "gdpr_reg_preprocessed_optiona.xlsx")) 
  df_rea_prep = pd.read_excel(join(INTERMEDIATE_DIRECTORY, "gdpr_rea_preprocessed_optiona.xlsx")) 
  reg_relevant_sentences = df_reg_prep["reg_sent"].tolist()
  rea_relevant_sentences = df_rea_prep["rea_sent"].tolist()
  reg_relevant_keywords = df_reg_prep["reg_kw_total"].tolist()
  rea_relevant_keywords = df_rea_prep["rea_kw_total"].tolist()

  ## STEP1 ############################################ START
  if direct_s_bert:
    # S-BERT Finding Sentence Pairs 
    sbsp = S_Bert_Sentence_Pairs(gamma_s_bert)
    df_bert_sent_pairs = sbsp.get_bert_sim_sent_pairs(reg_relevant_sentences, rea_relevant_sentences)
    df_bert_sent_pairs.to_excel(join(INTERMEDIATE_DIRECTORY, "df_sbert_step1_results.xlsx"))  
    count_unmapped_reg_sent = sbsp.get_unmapped_reg_sentences(df_bert_sent_pairs, reg_relevant_sentences)
    count_unmapped_rea_sent = sbsp.get_unmapped_rea_sentences(df_bert_sent_pairs, rea_relevant_sentences)
    count_mapped_rea_sent = sbsp.get_mapped_rea_sentences()
    print('S-BERT step 1 finished.')
  elif s_bert_with_kw:
    # S-BERT Finding Sentence Pairs 
    sbsp = S_Bert_Sentence_Pairs(gamma_s_bert)
    df_bert_sent_pairs = sbsp.get_bert_sim_sent_pairs(reg_relevant_keywords, rea_relevant_keywords)
    df_bert_sent_pairs.to_excel(join(INTERMEDIATE_DIRECTORY, "df_kw_sbert_step1_results.xlsx"))  
    print('Keyword S-BERT step 1 finished.')
  elif legal_s_bert:
    # LEGAL S-BERT Finding Sentence Pairs 
    lsbsp = Legal_S_Bert_Sentence_Pairs(gamma_s_bert)
    df_bert_sent_pairs = lsbsp.get_bert_sim_sent_pairs(reg_relevant_sentences, rea_relevant_sentences)
    df_bert_sent_pairs.to_excel(join(INTERMEDIATE_DIRECTORY, "df_legal_nli_sbert_step1_results_all.xlsx"))  
    count_unmapped_reg_sent = lsbsp.get_unmapped_reg_sentences(df_bert_sent_pairs, reg_relevant_sentences)
    count_unmapped_rea_sent = lsbsp.get_unmapped_rea_sentences(df_bert_sent_pairs, rea_relevant_sentences)
    count_mapped_rea_sent = lsbsp.get_mapped_rea_sentences()
    print('Legal S-BERT step 1 finished.')
  elif clustering:
    # Grouping Topic Model
    tm = Topic_Modeling(reg_relevant_sentences, rea_relevant_sentences, nlp)
    df_topic_models = tm.create_topics_dataframe()
    # Grouping Clustering
    kmb = K_Means_BERT(reg_relevant_sentences, rea_relevant_sentences, nlp, df_topic_models)
    df_topic_kmeans_groups = kmb.predict_clusters_to_df()
    df_topic_kmeans_groups.to_excel(join(INTERMEDIATE_DIRECTORY, "df_grouping_results.xlsx"))  
    # Check constraint completness (similarity computation between sentences within clusters)
    cec = Constraint_Existence_Check(nlp, df_topic_kmeans_groups, gamma_grouping)
    topic_model_sentence_pairs_df = cec.split_results_by_similarity(cec.topic_model_similarities_df, 'df_topic_model_reg_sent_without_match.xlsx', 'df_topic_model_rea_sent_without_match.xlsx')
    kmeans_bert_sentence_pairs_df = cec.split_results_by_similarity(cec.kmeans_bert_similarities_df, 'df_kmeans_bert_reg_sent_without_match.xlsx', 'df_kmeans_bert_rea_sent_without_match.xlsx')
    topic_model_sentence_pairs_df.to_excel(join(INTERMEDIATE_DIRECTORY, "topic_model_sentence_pairs_df.xlsx")) 
    kmeans_bert_sentence_pairs_df.to_excel(join(INTERMEDIATE_DIRECTORY, "kmeans_bert_sentence_pairs_df.xlsx")) 
  elif key_phrase: 
    # Finding Sentance Pairs through tfidf keywords
    kp = Key_Phrase(gamma_key_phrase,nlp, reg_relevant_sentences, rea_relevant_sentences)
    df_key_phrase_sent_pairs = kp.get_keyword_sim_sent_pairs()
    df_key_phrase_sent_pairs.to_excel(join(INTERMEDIATE_DIRECTORY, "key_phrase_sent_pairs.xlsx"))  
  ## STEP1 ############################################ END



## STEP2 ############################################ START
  # a) Phrase Extraction (splitting the sentences into Sub/Verb/Obj phrases) from each Sentence Pair
  ## STEP 2a) is done in another repository as it delivers better results with another python version (see readme)
  ## Input for Step 2a) is the result of step 1 and the output should be saved in path "STEP_TWO_EXTRACTION_RESULTS" --> "gs_reg_rea_extracted_phrases.xlsx"

  # b) Calculating Phrase Similarities
elif step_2: 
  df_extracted_phrases = pd.read_excel(join(STEP_TWO_EXTRACTION_RESULTS, "gs_reg_rea_extracted_phrases.xlsx")) 
  psc = Phrase_Similarity_Computation()
  phrase_similarity_s_bert = psc.get_phrase_similarities(df_extracted_phrases)
  phrase_similarity_s_bert.to_excel(join(STEP_TWO_SIMILARITY_RESULTS, "phrase_similarity_s_bert.xlsx")) 
  # Result (checking if deviations of different types can be detected)
"""
#additional script for deviation assessment
  dc = Deviation_Counter(nlp, gamma_one, gamma_two, gamma_three)
  s_bert_master_results_df = dc.get_deviation_flag(similarity_s_bert_constraint_phrases)
  s_bert_master_results_df.to_excel(join(RESULT_DIRECTORY, "s_bert_master_results_df.xlsx")) 
  s_bert_overview_results_df = dc.aggreagte_deviation_count(s_bert_master_results_df, count_unmapped_reg_sent, count_unmapped_rea_sent, count_mapped_rea_sent)
  s_bert_overview_results_df.to_excel(join(RESULT_DIRECTORY, "s_bert_overview_results_df.xlsx")) 
  print('Calculations finished.')
"""
## STEP2 ############################################ END