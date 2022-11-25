

test_text1 = "Prior to giving consent, the data subject shall be informed thereof."
test_text2 = "in such a case, The processor shall inform the controller of that legal requirement before processing, unless that law prohibits such information on important grounds of public interest."
test_text3 = "If applicable standards or legal provisions require the processor to carry out the processing contrary to the controller's instructions, or if these provisions prevent the processor from meeting the processor obligations under this Policy or under the agreement on processing on behalf, then the processor shall immediately inform the processor controller unless the legal provision in question forbids such notification."


'''
# trial key phrase extraction with KeyBert
from keybert import KeyBERT
# choosing an S-BERT Model --> in future best based on LEGAL BERT
kw_model = KeyBERT(model='all-MiniLM-L6-v2')
keywords = kw_model.extract_keywords(test_text1, keyphrase_ngram_range=(1, 3), stop_words='english', highlight=False, top_n=3)
keywords_list= list(dict(keywords).keys())
print(keywords_list)

# - ngram range is always the same (but sometimes 1/2/3 words makes more sense)
# - often words duplicate in different ngram combinations
# - quite slow for just one sent
# + seems to extract key words/phrases
'''


# trial key phrase extraction with KeyBert
import yake
kw_extractor = yake.KeywordExtractor(top=3, stopwords=None)
keywords = kw_extractor.extract_keywords(test_text2)
for kw, v in keywords:
  print("Keyphrase: ",kw, ": score", v)
# - doesn't seem to extract key words/phrases from a semantical point of view (like test sent 1 - no data no subject)
# + extracts phrases of different length (1-3 words)
# + very fast
