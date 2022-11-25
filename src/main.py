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

#from classes.iso_text_cleaning import *
#from classes.topic_modeling import *
#from classes.k_means_bert import *
#from classes.key_phrase import *
#from classes.constraint_existence_check import *
from classes.phrase_extraction import *
from classes.phrase_similarity_computation import *
from classes.s_bert_sentence_pairs import *
#from classes.legal_s_bert_sentence_pairs import *
#from classes.deviation_counter import *


## Application Selection ########################################START
# choose method 
direct_s_bert = True #if True --> no clustering or other means are implemented, all sentences are comapred with each other via S-Bert
legal_s_bert = False #if True --> implementation like S-BERT but based on legal BERT instead of BERT
clustering = False #if True --> 2 approaches calculated: a) topic model + word2vec + cosine sim; b) bert embeddings + kmeans and word2vev + cosine sim
key_phrase = False # if True --> setp one is performed on only key phrases (identified by tfidf), instead of whole sentences
# choose case study
### GDPR adjusted, ISO not!
iso = False #if False --> running with gdpr setup
# choose set up
rea_only_signal = False #if False --> gdpr realization input is not filtered to contain only sentences with signalwords
run_step_2_independently = True #if True --> step 2 (phrase level) is  calculated based on the goldstandard of step 1, so it can be evaluated independly of step 1; if False --> step 2 (phrase level) is calculated based on the results of step 1
# choose thresholds:
gamma_s_bert = 0.7 #0.67 #used for sentence mapping 
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
## parse defined lists of constraint signalwords, sequencemarkers and stopwords ########################### START
def read_defined_lists(directory): 
  '''reads in defined txts of constraint signalwords, sequencemarkers and stopwords as lists
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

## calling classes ############################################ START
#Text cleaning
if iso: 
  itc = Iso_Text_Cleaning(nlp, signalwords, ISMS_words, top_management_words)
  # not adjusted to new coref resolution
  reg_relevant_sentences = itc.get_relevant_sentences(reg_para_anaphora_resolved)
  rea_relevant_sentences = itc.get_relevant_sentences(rea_para_anaphora_resolved)
else:
  tc = Text_Cleaning(nlp, signalwords, controller_words, data_protection_officer_words, management_words)
  reg_anaphora_resolved_paragraphs = tc.clean_text(reg_paragraphs)
  reg_relevant_sentences = tc.get_relevant_sentences_reg(reg_anaphora_resolved_paragraphs)
  rea_anaphora_resolved_paragraphs = tc.clean_text(rea_paragraphs)
  if rea_only_signal:
    rea_relevant_sentences = tc.get_relevant_sentences_rea(rea_anaphora_resolved_paragraphs)
  else:
    rea_relevant_sentences = tc.get_relevant_sentences_no_sig_filter(rea_anaphora_resolved_paragraphs)
  #save all input constraints before matching for evaluation purposes  
  pd.DataFrame(reg_relevant_sentences).to_excel(join(INTERMEDIATE_DIRECTORY, "gdpr_reg_relevant_sentences.xlsx"))  
  pd.DataFrame(rea_relevant_sentences).to_excel(join(INTERMEDIATE_DIRECTORY, "gdpr_rea_relevant_sentences.xlsx"))  
"""
"""
if direct_s_bert:
  # S-BERT Finding Sentence Pairs 
  sbsp = S_Bert_Sentence_Pairs(gamma_s_bert)
  df_bert_sent_pairs = sbsp.get_bert_sim_sent_pairs(reg_relevant_sentences, rea_relevant_sentences)
  df_bert_sent_pairs.to_excel(join(INTERMEDIATE_DIRECTORY, "df_bert_sent_pairs.xlsx"))  
  count_unmapped_reg_sent = sbsp.get_unmapped_reg_sentences(df_bert_sent_pairs, reg_relevant_sentences)
  count_unmapped_rea_sent = sbsp.get_unmapped_rea_sentences(df_bert_sent_pairs, rea_relevant_sentences)
  count_mapped_rea_sent = sbsp.get_mapped_rea_sentences()
"""
  ### STEP 2 ###
  ## a) Phrase Extraction (splitting the sentences into Sub/Verb/Obj phrases) from each Sentence Pair

"""
  pe = Phrase_Extraction()
  if run_step_2_independently: #input of step 2 is gs of step 1
    # as step_2 a was build in different environment and delivers other results in this setup, the results of step 2 a) will be imported from the other script
    # df_phrases_reg = pe.extract_phrases(REG_GS_STEP_ONE_INPUT_STEP_TWO, 'reg')
    # df_phrases_rea = pe.extract_phrases(REA_GS_STEP_ONE_INPUT_STEP_TWO, 'rea')
    # read excel as df
    df_extracted_phrases = pd.read_excel(join(STEP_TWO_EXTRACTION_RESULTS, "gs_reg_rea_extracted_phrases.xlsx")) 

  else: # input for step 2 is result of step 1
    pd.DataFrame(df_bert_sent_pairs).to_excel(join(INTERMEDIATE_DIRECTORY, "step1_result_df.xlsx"))  
    wb = xlrd.open_workbook(join(INTERMEDIATE_DIRECTORY,"step1_result_df.xlsx"))
    sh = wb.sheet_by_index(0)
    file_count = 0
    for row in sh.get_rows():
      text = row[0].value
      file = open('reg_{}.txt'.format(file_count), "w")
      file.write(text)
      file.close()
      file_count += 1
    s_bert_constraint_phrases_reg = pe.get_sentence_phrases(df_bert_sent_pairs, 'reg')
    s_bert_constraint_phrases = pe.get_sentence_phrases(s_bert_constraint_phrases_reg, 'rea')
    s_bert_constraint_phrases.to_excel(join(INTERMEDIATE_DIRECTORY, "s_bert_constraint_phrases.xlsx")) 
"""

  ## b) Calculating Phrase Similarities
if direct_s_bert: 
  pe = Phrase_Extraction()
  df_extracted_phrases = pd.read_excel(join(STEP_TWO_EXTRACTION_RESULTS, "gs_reg_rea_extracted_phrases.xlsx")) 
  psc = Phrase_Similarity_Computation()
  phrase_similarity_s_bert = psc.get_phrase_similarities(df_extracted_phrases)
  phrase_similarity_s_bert.to_excel(join(STEP_TWO_SIMILARITY_RESULTS, "phrase_similarity_s_bert.xlsx")) 
  # Result (checking if deviations of different types can be detected)
"""
  dc = Deviation_Counter(nlp, gamma_one, gamma_two, gamma_three)
  s_bert_master_results_df = dc.get_deviation_flag(similarity_s_bert_constraint_phrases)
  s_bert_master_results_df.to_excel(join(RESULT_DIRECTORY, "s_bert_master_results_df.xlsx")) 
  s_bert_overview_results_df = dc.aggreagte_deviation_count(s_bert_master_results_df, count_unmapped_reg_sent, count_unmapped_rea_sent, count_mapped_rea_sent)
  s_bert_overview_results_df.to_excel(join(RESULT_DIRECTORY, "s_bert_overview_results_df.xlsx")) 
  print('Calculations finished.')


elif legal_s_bert:
  # LEGAL S-BERT Finding Sentence Pairs 
  sbsp = Legal_S_Bert_Sentence_Pairs(gamma_s_bert)
  df_bert_sent_pairs = sbsp.get_bert_sim_sent_pairs(reg_relevant_sentences, rea_relevant_sentences)
  df_bert_sent_pairs.to_excel(join(INTERMEDIATE_DIRECTORY, "df_bert_sent_pairs.xlsx"))  
  count_unmapped_reg_sent = sbsp.get_unmapped_reg_sentences(df_bert_sent_pairs, reg_relevant_sentences)
  count_unmapped_rea_sent = sbsp.get_unmapped_rea_sentences(df_bert_sent_pairs, rea_relevant_sentences)
  count_mapped_rea_sent = sbsp.get_mapped_rea_sentences()
  # Phrase Extraction (splitting the sentences into Sub/Verb/Obj phrases) from each Sentence Pair
  idc = In_Depth_Comparison(signalwords, nlp)
  s_bert_constraint_phrases_reg = idc.get_sentence_phrases(df_bert_sent_pairs, 'reg')
  s_bert_constraint_phrases = idc.get_sentence_phrases(s_bert_constraint_phrases_reg, 'rea')
  s_bert_constraint_phrases.to_excel(join(INTERMEDIATE_DIRECTORY, "s_bert_constraint_phrases.xlsx"))  
  # Similarities for Phrases (calculating similarity between the phrases)
  psc = Phrase_Similarity_Computation()
  similarity_s_bert_constraint_phrases = psc.get_phrase_similarities(s_bert_constraint_phrases)
  similarity_s_bert_constraint_phrases.to_excel(join(INTERMEDIATE_DIRECTORY, "similarity_s_bert_constraint_phrases.xlsx")) 
  # Result (checking if deviations of different types can be detected)
  dc = Deviation_Counter(nlp, gamma_one, gamma_two, gamma_three)
  s_bert_master_results_df = dc.get_deviation_flag(similarity_s_bert_constraint_phrases)
  s_bert_master_results_df.to_excel(join(RESULT_DIRECTORY, "s_bert_master_results_df.xlsx")) 
  s_bert_overview_results_df = dc.aggreagte_deviation_count(s_bert_master_results_df, count_unmapped_reg_sent, count_unmapped_rea_sent, count_mapped_rea_sent)
  s_bert_overview_results_df.to_excel(join(RESULT_DIRECTORY, "s_bert_overview_results_df.xlsx")) 
  print('Calculations finished.')

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

  # In depth comparison (on phrase level) of Sentence Pairs (splitting the sentences into Sub/Verb/Obj phrases)
  idc = In_Depth_Comparison(signalwords, nlp)
  topic_model_constraint_phrases_reg = idc.get_sentence_phrases(topic_model_sentence_pairs_df, 'reg')
  topic_model_constraint_phrases = idc.get_sentence_phrases(topic_model_constraint_phrases_reg, 'rea')
  kmeans_bert_constraint_phrases_reg = idc.get_sentence_phrases(kmeans_bert_sentence_pairs_df, 'reg')
  kmeans_bert_constraint_phrases = idc.get_sentence_phrases(kmeans_bert_constraint_phrases_reg, 'rea') 
  # Similarities for Phrases
  psc = Phrase_Similarity_Computation(nlp)
  similarity_topic_model_constraint_phrases = psc.get_phrase_similarities(topic_model_constraint_phrases)
  similarity_kmeans_bert_constraint_phrases = psc.get_phrase_similarities(kmeans_bert_constraint_phrases)
  similarity_topic_model_constraint_phrases.to_excel(join(INTERMEDIATE_DIRECTORY, "similarity_topic_model_constraint_phrases.xlsx"))  
  similarity_kmeans_bert_constraint_phrases.to_excel(join(INTERMEDIATE_DIRECTORY, "similarity_kmeans_bert_constraint_phrases.xlsx")) 

elif key_phrase: 
  # Finding Sentance Pairs through tfidf keywords
  kp = Key_Phrase(gamma_key_phrase,nlp, reg_relevant_sentences, rea_relevant_sentences)
  df_key_phrase_sent_pairs = kp.get_keyword_sim_sent_pairs()
  df_key_phrase_sent_pairs.to_excel(join(INTERMEDIATE_DIRECTORY, "key_phrase_sent_pairs.xlsx"))  

"""