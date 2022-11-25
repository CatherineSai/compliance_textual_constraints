from spacy.matcher import Matcher


class Phrase_Extraction:

  def __init__(self, signalwords, nlp):
    self.signalwords = signalwords
    self.nlp = nlp
    self.object_list = ["dobj", "dative", "attr", "oprd", "pobj", "prep"]
    self.verb_extensions = ["auxpass", "prep", "acomp", "pobj", "neg"]
    self.connectors = ["and", "or"]

  def get_sentence_phrases(self, df, doc_name):
    '''extract the phrases for the matched sentences for an in depth analysis of deviation'''
    print("Start Phrase_Extraction")
    name = doc_name
    df["number_negations_{}".format(name)] = ""
    df["subject_phrase_1_{}".format(name)] = ""
    df["verb_phrase_1_{}".format(name)] = ""
    df["object_phrase_1_{}".format(name)] = ""
    df["subject_phrase_2_{}".format(name)] = ""
    df["verb_phrase_2_{}".format(name)] = ""
    df["object_phrase_2_{}".format(name)] = ""
    for index, row in df.iterrows():
        # get conctraint phrases for reg sentences
        sentence = row["original_sentence_{}".format(name)]
        doc = self.nlp(sentence)
        self.get_number_of_constraints(doc)
        self.subject_phrase = self.get_subject_phrase(doc)
        self.root_token = self.find_root_of_sentence(doc)
        self.root_phrase = self.find_root_phrase(doc)
        self.other_verb_phrases = self.find_other_verb_phrases(doc)
        self.object_phrase = self.get_object_phrase(doc)
        # writing to df
        negation_item = self.get_number_of_negations_in_sentence(doc)
        df.at[index,"number_negations_{}".format(name)] = negation_item
        i = 0
        for i in range(self.signalword_count):
            if i == 0:
                verb_item = self.root_phrase
                try:
                    subject_item = self.subject_phrase[i].text
                except:
                    subject_item = self.object_phrase[i].text
                try:
                    object_item = self.object_phrase[i].text
                except: 
                    object_item = self.subject_phrase[i].text
                df.at[index,"subject_phrase_1_{}".format(name)] = subject_item
                df.at[index,"verb_phrase_1_{}".format(name)] = verb_item
                df.at[index,"object_phrase_1_{}".format(name)] = object_item
            else: 
                try:
                    subject_item = self.subject_phrase[i].text
                except:
                    try:
                        subject_item = self.subject_phrase[i-1].text
                    except:
                        subject_item = self.object_phrase[i].text
                try: 
                    verb_item = self.other_verb_phrases[i]
                except:
                    try:
                        verb_item = self.other_verb_phrases[i-1]
                    except:
                        verb_item = self.root_phrase
                try:
                    object_item = self.object_phrase[i].text
                except:
                    try:
                        object_item = self.object_phrase[i-1].text
                    except: 
                        object_item = self.subject_phrase[i].text
                df.at[index,"subject_phrase_2_{}".format(name)] = subject_item
                df.at[index,"verb_phrase_2_{}".format(name)] = verb_item
                df.at[index,"object_phrase_2_{}".format(name)] = object_item
    return df

  def get_number_of_constraints(self, doc):
    '''depending on the number of signalwords per sentence there can be multiple constraints (one per signalword)'''
    self.signalword_count = 0
    # check if more object_phrases than signalword occurances 
    for token in doc:
        for element in self.signalwords:
            if token.text == element:
                self.signalword_count = self.signalword_count+1  

  def get_number_of_negations_in_sentence(self, doc):
    '''extracts the number of explicit negations from a sentence'''
    no_negations = 0
    for token in doc:   
        if (token.dep_ == 'neg'):
            no_negations =+ 1
    return no_negations

  def get_subject_phrase(self, doc):
    '''finds the subject phrase of the sentence, if there are more subject phrases than signalwords, deletes subject phrases fursest from signalword'''
    fursest_sub_position = 0
    subj_spans = []
    for token in doc:   
        if ("subj" in token.dep_):
            subtree = list(token.subtree)
            start = subtree[0].i
            end = subtree[-1].i + 1
            subj_spans.append(doc[start:end])
    if len(subj_spans) > self.signalword_count:
        for l in range(len(subj_spans)):
            try:
                sub_pos = subj_spans[l].start       
                while len(subj_spans) > self.signalword_count:
                    # get signalword position
                    for token in doc:  
                        if token.text in self.signalwords:
                            signal_word_pos = token.i
                            print (signal_word_pos)
                            if abs(signal_word_pos-sub_pos) > fursest_sub_position:
                                fursest_sub_position = l
                    # delete subject phrase furthest away from signalword
                    del subj_spans[fursest_sub_position]
            except:
                continue
    return subj_spans

  def find_root_of_sentence(self, doc):
    '''find the dependency root token of the sentence'''
    root_token = None
    for token in doc:
        if (token.dep_ == "ROOT"):
            root_token = token
    return root_token

  def find_root_phrase(self, doc):
    '''find the dependency root phrase of the sentence, e.g. instead of be (ROOT) --> be necessary'''
    root_phrase = []
    for token in doc:
        if (token.dep_ == "ROOT"):
            root_phrase.append(doc[token.i])
            ''' #extensions of verb phrase at current stage lead to worse similarity, this need further work in future 
            if doc[token.i -1].dep_ in self.verb_extensions:
                root_phrase.append(doc[token.i -1])
            if doc[token.i -2].dep_ in self.verb_extensions:
                root_phrase.append(doc[token.i -2])
            try:
                if doc[token.i + 1].dep_ in self.verb_extensions:
                    root_phrase.append(doc[token.i + 1])
            except:
                continue
            try:
                if doc[token.i + 1].text in self.connectors:
                    root_phrase.append(doc[token.i + 1])
                    root_phrase.append(doc[token.i + 2])
            except:
                continue
            try:
                if doc[token.i + 2].dep_ in self.verb_extensions:
                    root_phrase.append(doc[token.i + 2])
            except:
                continue
            '''
    return (" ".join([t.text for t in root_phrase]))            

  def find_other_verb_phrases(self, doc):
    other_verb_phrases = []
    for token in doc:
        new_phrase = []
        if (token.pos_ == "VERB" and token.dep_ != "ROOT"):
            new_phrase.append(token)
            ''' #extensions of verb phrase at current stage lead to worse similarity, this need further work in future 
            if doc[token.i -2].dep_ in self.verb_extensions:
                new_phrase.append(doc[token.i -2])
            if doc[token.i -1].dep_ in self.verb_extensions:
                new_phrase.append(doc[token.i -1])
            try:
                if doc[token.i + 1].dep_ in self.verb_extensions:
                    new_phrase.append(doc[token.i + 1])
            except:
                continue
            try:
                if doc[token.i + 1].text in self.verb_extensions:
                    new_phrase.append(doc[token.i + 1])
                    new_phrase.append(doc[token.i + 2])
            except:
                continue
            try:
                if doc[token.i + 2].dep_ in self.verb_extensions:
                    new_phrase.append(doc[token.i + 2])
            except:
                continue
            '''
            verb_phrase = (" ".join([t.text for t in new_phrase]))
            other_verb_phrases.append(verb_phrase)
    #delete items from other verbs phrases if they are overlapping with subject phrase
    for i in range(len(other_verb_phrases)):
        try:
            while (other_verb_phrases[i] in self.subject_phrase.text):
                del other_verb_phrases[i]
        except:
            continue
    ''' 
    # needs more work - sometimes list index range error  
    # only keep as many other_verb_phrases as signalwords, delete front ones (as they are more likely to overlap with subject phrase)
    for i in range(len(other_verb_phrases)):           
        while len(other_verb_phrases)+1 > self.signalword_count:
            del other_verb_phrases[i]
    ''' 
    return other_verb_phrases

  def get_object_phrase(self, doc):
    obj_spans = []
    for token in doc:
        for element in self.object_list:
            if (element in token.dep_):
                subtree = list(token.subtree)
                start = subtree[0].i
                end = subtree[-1].i + 1
                obj_spans.append(doc[start:end])
    #delete substrings from object phrases
    for i in range(len(obj_spans)-1):
        try:
            while (obj_spans[i].text in obj_spans[i+1].text):
                del obj_spans[i]
        except:
            continue
        try:
            while (obj_spans[i+1].text in obj_spans[i].text): 
                del obj_spans[i+1]   
        except:
            continue
    return self.object_phrase_cleaning(doc, obj_spans)

  def object_phrase_cleaning(self, doc, object_phrase):
    '''1. deletes object phrases that are overlapping with subject_phrase, root_phrase or other_verbs_phrase 
    2. checks if there are still more object-phrases than signalwords found: deletes first object phrases in list until there are only as many as signal words'''           
    #delete items from object phrases if they are overlapping with subject phrase, root phrase or other_verb_phrases
    for i in range(len(object_phrase)):
        try:
            while (object_phrase[i].text in self.subject_phrase.text):
                del object_phrase[i]
        except:
            continue
        try:
            while (object_phrase[i].text in self.root_phrase):
                del object_phrase[i]
        except:
            continue
        try:
            for k in range(len(self.other_verb_phrases)):
                while (object_phrase[i].text in self.other_verb_phrases[k]):
                    del object_phrase[i]
        except:
            continue
    # only keep as many object_phrases as signalwords, delete front ones (as they are more likely to overlap with subject phrase)
    for i in range(len(object_phrase)):           
        while len(object_phrase) > self.signalword_count:
            del object_phrase[i]

    # Import the Matcher and initialize it with the shared vocabulary
    matcher = Matcher(self.nlp.vocab)
    # Write a pattern for adjective plus one or two nouns
    pattern = [{'POS': 'DET'},{'POS': 'NOUN'}, {'POS': 'ADJ', 'OP': '?'}, {'POS': 'ADP'}, {'POS': 'DET'}, {'POS': 'NOUN'}, {'POS': 'ADJ'}]
    # Add the pattern to the matcher and apply the matcher to the doc
    matcher.add('ADJ_NOUN_PATTERN', [pattern])
    matches = matcher(doc)
    # Iterate over the matches and print the span text
    for match_id, start, end in matches:
        for i in range(len(object_phrase)):
            try:
                if (object_phrase[i].text in doc[start:end].text):
                    object_phrase = []
                    [object_phrase.append(doc[start:end].text)]
            except:
                continue
    return object_phrase

