import os
import re
import spacy
from string import punctuation, whitespace
import pandas as pd
from file_paths import *
from classes.phrase_extraction_helper import *


class Phrase_Extraction:

  def __init__(self):
    # acreate new nlp pipeline (as the main one is different to this) and add own phrase_spans component to nlp pipeline
    nlp_pe = spacy.load('en_core_web_trf')
    nlp_pe.add_pipe('merge_noun_chunks')
    nlp_pe.add_pipe('merge_entities')
    nlp_pe.add_pipe('phrase_spans') #all the calculations are imported via this custom component from helper script
    #nlp_pe.to_disk(join(INTERMEDIATE_DIRECTORY, "nlp_phrase_extract_model"))
    self.nlp_pe = nlp_pe

  def extract_phrases(self, input_path, doc_name):
    # read all .txt files from the input folder
    documents = _read_documents(input_path)
    result_df = pd.DataFrame()
    # extract each sentence and save in results folder
    for file in documents:
        sentence = documents[file]
        doc = self.nlp_pe(sentence)
        all_spans = doc.spans['sc']
        spans_dict = dict()
        for label, spans in groupby(all_spans, lambda span: span.label_):
            # merge all spans of one category into one string
            merged = ' '.join([span.text.strip(punctuation + whitespace) for span in spans])
            spans_dict[label] = merged
        # output df with extracted spans
        df = pd.DataFrame(
            {
                "EXTRACTED": [sentence] + [spans_dict.get(category, '') for category in categories],
            },
            index = ['SENTENCE'] + categories
        )
        # transpose, add negation column
        df_out = df.transpose()
        df_out["{}_no_neg".format(doc_name)] = df_out.apply(lambda row : get_number_of_negations_in_sentence(row['VERB']), axis = 1)
        result_df = pd.concat([result_df, df_out])
    # change names of dfs
    result_df.columns = result_df.columns.str.replace('SENTENCE', '{}_original_sentence'.format(doc_name))
    result_df.columns = result_df.columns.str.replace('SUBJECT', '{}_sub'.format(doc_name))
    result_df.columns = result_df.columns.str.replace('VERB', '{}_verb'.format(doc_name))
    result_df.columns = result_df.columns.str.replace('TIME', '{}_time'.format(doc_name))
    result_df.columns = result_df.columns.str.replace('CONDITION', '{}_cond'.format(doc_name))
    result_df.columns = result_df.columns.str.replace('OBJECT', '{}_obj'.format(doc_name))
    pd.DataFrame(result_df).to_excel(join(STEP_TWO_EXTRACTION_RESULTS, "{}_step_2_extraction_results.xlsx".format(doc_name))) 

def get_number_of_negations_in_sentence(text):
    '''extracts the number of explicit negations from a sentence'''
    no_negations = 0
    nlp_neg = spacy.load('en_core_web_trf')
    doc = nlp_neg(text)
    for token in doc:   
        if (token.dep_ == 'neg'):
            no_negations =+ 1
    return no_negations

def _read_documents(directory): 
    '''reads in txts of regulatory and realization documents
    Input: multiple .txt files (each a sentence)
    Output: dictionary with file name as key and its content as value'''
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
  



