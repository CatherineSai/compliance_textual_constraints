
from spacy.matcher import Matcher


class Iso_Text_Cleaning:
  def __init__(self, nlp, signalwords, ISMS_words, top_management_words):
    self.nlp = nlp
    self.signalwords = signalwords
    self.ISMS_words = ISMS_words
    self.top_management_words = top_management_words
  
  def find_paragraph_references(self):
    '''Matcher finding mentions like "article 5", "paragraphs 3 and 7", "No 1338/2008" '''
    paragraph_list = []
    # Import the Matcher and initialize it with the shared vocabulary
    matcher = Matcher(self.nlp.vocab)
    # Write a pattern for adjective plus one or two nouns
    pattern1 = [{"LEMMA": {"IN": ["paragraph"]}}, {'POS': 'NUM'}]
    pattern2 = [{"LEMMA": {"IN": ["paragraph"]}}, {'POS': 'NUM'}, {}, {'POS': 'NUM'}]
    # Add the pattern to the matcher and apply the matcher to the doc
    matcher.add('Referenz_simplifier1', [pattern1])
    matcher.add('Referenz_simplifier2', [pattern2])
    for para in self.cleaned_paragraphs_list:
        doc = self.nlp(para)
        matches = matcher(doc)
        # Iterate over the matches and append to list
        for match_id, start, end in matches:
            for i in range(len(doc)):
                try:
                    if (doc[i].text in doc[start:end].text):
                        paragraph_list.append(doc[start:end].text) 
                except:
                    continue
    return set(paragraph_list)

  def find_article_references(self):
    '''Matcher finding mentions like "article 5", "paragraphs 3 and 7", "No 1338/2008" ''' 
    article_list = []
    # Import the Matcher and initialize it with the shared vocabulary
    matcher = Matcher(self.nlp.vocab)
    # Write a pattern for adjective plus one or two nouns
    pattern3 = [{"LEMMA": {"IN": ["article"]}}, {'POS': 'NUM'}]
    pattern4 = [{"LEMMA": {"IN": ["article"]}}, {'POS': 'NUM'}, {}, {'POS': 'NUM'}]
    # Add the pattern to the matcher and apply the matcher to the doc
    matcher.add('Referenz_simplifier3', [pattern3])
    matcher.add('Referenz_simplifier4', [pattern4])
    for para in self.cleaned_paragraphs_list:
        doc = self.nlp(para)
        matches = matcher(doc)
        # Iterate over the matches and append to list
        for match_id, start, end in matches:
            for i in range(len(doc)):
                try:
                    if (doc[i].text in doc[start:end].text):
                        article_list.append(doc[start:end].text) 
                except:
                    continue 
    return set(article_list)
    
  def find_number_specification_references(self):
    '''Matcher finding mentions like "article 5", "paragraphs 3 and 7", "No 1338/2008" '''
    number_specification_list = []
    # Import the Matcher and initialize it with the shared vocabulary
    matcher = Matcher(self.nlp.vocab)
    # Write a pattern for adjective plus one or two nouns
    pattern5 = [{'ORTH': 'No'}, {'POS': 'NUM'}]
    # Add the pattern to the matcher and apply the matcher to the doc
    matcher.add('Referenz_simplifier5', [pattern5])
    for para in self.cleaned_paragraphs_list:
        doc = self.nlp(para) 
        matches = matcher(doc)
        # Iterate over the matches and append to list
        for match_id, start, end in matches:
            for i in range(len(doc)):
                try:
                    if (doc[i].text in doc[start:end].text):
                        number_specification_list.append(doc[start:end].text) 
                except:
                    continue
    return set(number_specification_list)

  def replace_found_references(self, paragraph_references, article_references, number_specification_references):
    '''Matcher finding mentions like "article 5", "paragraphs 3 and 7", "No 1338/2008" '''           
    self.cleaned_paragraphs_list3 = []
    self.cleaned_paragraphs_list4 = []
    self.cleaned_paragraphs_list5 = []
    for para in self.cleaned_paragraphs_list:
        new_para = para
        for item in paragraph_references:
            new_para = new_para.replace(item, 'paragraph')
        self.cleaned_paragraphs_list3.append(new_para)
    for para in self.cleaned_paragraphs_list3:
        new_para = para
        for item in article_references:
            new_para = new_para.replace(item, 'article')
        self.cleaned_paragraphs_list4.append(new_para)
    for para in self.cleaned_paragraphs_list4:
        new_para = para
        for item in number_specification_references:
            new_para = new_para.replace(item, "")
        self.cleaned_paragraphs_list5.append(new_para)


  def substitude_specific_realization_formulations(self):
    '''replaces realization specific words (like a company name as "daimler group") with a general term like just "company"
    or "Chief Officer Corporate Data Protection" with "controller"
    Input: List of Paragraphs
    Output: List of Paragraphs'''
    self.cleaned_paragraphs_list6 = []
    self.cleaned_paragraphs_list7 = []
    for para in self.cleaned_paragraphs_list5:
        new_para = para
        for item in self.ISMS_words:
            new_para = new_para.replace(item, 'information security management system')
        self.cleaned_paragraphs_list6.append(new_para)
    for para in self.cleaned_paragraphs_list6:
        new_para = para
        for item in self.top_management_words:
            new_para = new_para.replace(item, 'top management')
        self.cleaned_paragraphs_list7.append(new_para)

  def get_relevant_sentences(self, cleaned_paragraphs_list):
    '''creates a list with sentences from all paragaraphs, only keeping those sentences that contain at least one signalword
    Note: this wasn't already done in the function get_relevant_paragraphs because this way the anaphora resolution can be added
    Input: List of Paragraphs (as doc objects)
    Output: List of Sentences'''
    self.cleaned_paragraphs_list = cleaned_paragraphs_list
    paragraph_references = self.find_paragraph_references()
    article_references = self.find_article_references()
    number_specification_references = self.find_number_specification_references()
    self.replace_found_references(paragraph_references, article_references, number_specification_references)
    self.substitude_specific_realization_formulations()
    self.relevant_sentences = []
    for para in self.cleaned_paragraphs_list7:
        doc = self.nlp(para) 
        sentences = doc.sents
        for idx, sentence in enumerate(sentences):
            for token in sentence: 
                if (token.text in self.signalwords):
                    self.relevant_sentences.append(sentence.text.strip())
                    break
    return self.relevant_sentences