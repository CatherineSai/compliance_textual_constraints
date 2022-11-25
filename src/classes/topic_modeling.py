
'''
BACKGROUND INFO - adjusted from this source: https://towardsdatascience.com/short-text-topic-modelling-lda-vs-gsdmm-20f1db742e14  
There are two topic modelling approaches applied to short-text documents [...]: Latent Dirichlet Allocation (LDA) 
and Gibbs Sampling Dirichlet Multinomial Mixture (GSDMM). LDA assumes that each document consists of multiple topics and calculates 
the contribution of each topic to the document. GSDMM, on the other hand, is specifically aimed at detecting topics in smaller documents 
and assumes only one topic per document.
--> since we work on sentence level, it can be assumed that there is only one topic per sentence --> GSDMM seems to be the better fit!
'''
#from gsdmm import MovieGroupProcess
import pandas as pd
import numpy as np
import inspect
import spacy


class Topic_Modeling:
  def __init__(self, reg_relevant_sentences, rea_relevant_sentences, nlp):
    self.nlp = nlp
    self.threshold = 0.3
    self.df_reg = self.sentence_df(reg_relevant_sentences, 'reg')
    self.df_rea = self.sentence_df(rea_relevant_sentences, 'rea')

  def sentence_df(self, list_relevant_sentences, origin_string):
    '''creates df with column sentences and flag of their origin'''
    df = pd.DataFrame(list_relevant_sentences, columns =['original_sentence_text'])
    df['origin'] = origin_string
    return df

  def reg_rea_sentence_df(self):
    '''joins rea and reg sentences to one df'''
    self.df= self.df_reg.append(self.df_rea, ignore_index = True)

  def preprocessing_sentences(self):
    '''preprocesses the senteces into a df with one word per row and the words lemma form, and additional attributes'''
    docs = list(self.nlp.pipe(self.df.original_sentence_text))
    cols = ['doc_id', 'token','token_order', 'lemma', 'is_stop', 'is_punct']
    df = []
    for ix, doc in enumerate (docs):
        meta = self.extract_token_plus_meta(doc)
        meta = pd.DataFrame(meta)
        meta.columns = cols [1:]
        meta = meta.assign(doc_id=ix).loc[:,cols]
        df.append(meta)
    self.meta_df = pd.concat(df)

  def extract_token_plus_meta(self, doc:spacy.tokens.doc.Doc):
    '''helper function for preprocessing_sentences'''
    return [(i.text,i.i,i.lemma_, i.is_stop, i.is_punct) for i in doc]

  def cleaning_words(self):
    '''removing word that are labeled as stopwords or punctuations, dropping columns not needed anymore'''
    self.meta_df = self.meta_df.loc[(self.meta_df['is_stop'] == False) & (self.meta_df['is_punct'] == False)]
    self.meta_df= self.meta_df.drop(['token','token_order','is_stop','is_punct'], axis=1)

  def unique_lemma_list_of_lists(self):
    '''changing the meta_df (with one row per word) back to one row per sentence (doc_id) with one column lemma 
    which contains a list of the sentences remaining lemmas, this column is then extracted to a list of all the 
    lemma lists and duplicate lemmas within each sentence list are removed (requirement by the gsdmm) '''
    joined_lemma_df = self.meta_df.groupby('doc_id').agg(lambda x: x.tolist())
    list_of_lemma_list = joined_lemma_df["lemma"].tolist()
    self.unique_lemma_per_sentence_list = [list(set(word_list)) for word_list in list_of_lemma_list]

  def vocab_len(self):
    '''calculate len of total vocab - needed input for gsdmm topic modeling'''
    vocab = set(x for sentence_joined_lemma in self.unique_lemma_per_sentence_list for x in sentence_joined_lemma)
    self.n_terms = len(vocab)

  def gsdmm_topic_modeling(self):
    '''apply gsdmm - for installation refere to https://github.com/rwalk/gsdmm'''
    self.mgp = MovieGroupProcess(K=12, alpha=0.01, beta=0.01, n_iters=30)
    self.mgp.fit(self.unique_lemma_per_sentence_list, self.n_terms)

  def show_resulting_topics(self):
    '''gets clusters and corresponding top words for each'''
    doc_count = np.array(self.mgp.cluster_doc_count)
    self.top_clusters = doc_count.argsort()[-12:][::-1]# topics sorted by the number of document they are allocated to
    self.top_words(10)

  def top_words(self, num_words):
    '''helper function for show_resulting_topics'''
    for cluster in self.top_clusters:
        sort_dicts = sorted(self.mgp.cluster_word_distribution[cluster].items(), key=lambda k: k[1], reverse=True)[:num_words]
        print(f'Cluster {cluster}: {sort_dicts}')

  def topic_dict(self):
    '''creating a dictionary with the topics'''
    self.topic_dict = {}
    topic_names = ['0','1','2','3','4','5','6','7','8','9','10','11']
    for i, topic_num in enumerate( self.top_clusters):
        self.topic_dict[topic_num]=topic_names[i] 

  def create_topics_dataframe(self):
    '''map original sentences to topic clusters in df --> column "GSDM_Topic_model"'''
    self.reg_rea_sentence_df()
    self.preprocessing_sentences()
    self.cleaning_words()
    self.unique_lemma_list_of_lists()
    self.vocab_len()
    self.gsdmm_topic_modeling()
    self.show_resulting_topics()
    self.topic_dict()
    result = pd.DataFrame(columns=['id','origin','original_sentence','GSDM_topic_model'])
    for i, sentences in enumerate(self.df.original_sentence_text):
        result.at[i, 'id'] = self.df.index[i]
        result.at[i, 'origin'] = self.df.origin[i]
        result.at[i, 'original_sentence'] = self.df.original_sentence_text[i]
        prob = self.mgp.choose_best_label(self.unique_lemma_per_sentence_list[i])
        if prob[1] >= self.threshold:
            result.at[i, 'GSDM_topic_model'] = self.topic_dict[prob[0]]
        else:
            result.at[i, 'GSDM_topic_model'] = 'other'
    return result


