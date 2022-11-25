from spacy import Language
from spacy.tokens import Doc
from functools import reduce
from itertools import groupby
from spacy.tokens import Span, Doc

# dependencies that we take without further conditions, except for dep_cond
dep_signal = ['aux']
dep_subj = ['nsubj', 'csubj', 'nsubjpass', 'csubjpass']
dep_obj = ['dobj', 'pobj', 'iobj', 'dative', 'agent', 'ccomp', 'xcomp', 'attr']
dep_verb = ['neg', 'auxpass', 'prt', 'cc', 'conj', 'acomp']
dep_cond = ['acl', 'relcl', 'advcl', 'nmod', 'npadvmod', 'nounmod', 'advmod', 'prep'] # these could also be non-conditions, we do additional triggers
# annotation labels of spans
categories = ['SUBJECT', 'SIGNAL', 'VERB', 'TIME', 'CONDITION', 'OBJECT', 'OP_SUBJECT', 'OP_SIGNAL', 'OP_VERB', 'OP_TIME', 'OP_CONDITION', 'OP_OBJECT']
# defined trigger words in addition to dependency tags
signal = ['should', 'shall', 'must', 'may']
condition_trigger = ['without', 'within', 'where not', 'where', 'when not', 'when', 'upon', 'until not', 'until', 'unless not', 'unless and until', 'unless', 'timely', 'taking into account', 'subject to', 'regardless of', 'provided that not', 'provided that', 'prior to', 'only if', 'not to exceed', 'not subject to', 'not later than', 'not equal to', 'not earlier than' 'no later than', 'not earlier than', 'no more than', 'no less than', 'no later than', 'no earlier than', 'more than or equal to', 'more than', 'minimum of', 'minimum', 'maximum of', 'maximum', 'lesser than', 'lesser of', 'lesser', 'less than or equal to', 'less than', 'least of', 'least', 'later than', 'last of', 'irrespective of', 'in the case of', 'in the absence of', 'if not', 'if', 'highest', 'greatest of', 'greater than or equal to', 'greater than', 'greater of', 'greater', 'first of', 'extended', 'expressly', 'except', 'exceeds', 'exceed', 'exactly', 'equal to', 'earlier than', 'during', 'conditioned upon', 'conditioned on', 'before', 'at the time when', 'at the time', 'at the latest', 'at most', 'at least', 'as soon as', 'as long as', 'after']
time_point = ['years', 'year', 'weeks', 'week', 'seconds', 'second', 'period', 'periods', 'months', 'month', 'minutes', 'minute', 'hours', 'hour', 'days', 'day']
relative_words = ['which', 'who', 'that', 'whose', 'whom', 'where', 'what']

# Phrase is a list of tokens that would represent e.g. a subject phrase, etc.
class Phrase:
    def __init__(self, tokens_or_root=[], has_skips=False):
        self.has_skips = has_skips # this phrase is not guaranteed to be fully gapless in the sentence

        if type(tokens_or_root) is list: # list of tokens
            self.root = None
            self.tokens = tokens_or_root
        else: # a root that we would expand
            self.root = tokens_or_root
            self.tokens = _expand_subtree(self.root)

    def as_span(self, label):
        '''
        return a list of Span objects and their labels, such that each gapless index sequences are mapped to one span object e.g. [[3:7], [10:11], [15:17]]
        '''
        indices = list(set(t.i for t in self.tokens))
        indices.sort()
        spans = []
        # function found on stackoverflow https://stackoverflow.com/questions/4628333/converting-a-list-of-integers-into-range-in-python
        for group_id, group in groupby(enumerate(indices), lambda pair: pair[1] - pair[0]):
            group = list(group)
            spans.append(Span(self.tokens[0].doc, group[0][1], group[-1][1]+1, label))
        return spans

    def as_str(self, lower=True):
        text = ' '.join([t.text for t in self.tokens]) if self.has_skips else ''.join([t.text_with_ws for t in self.tokens])
        return text.lower() if lower else text
    
    def merge_as_str(phrases):
        if phrases is None:
            return ''
        return ' | '.join([str(phrase) for phrase in phrases])

    def merge(phrases):
        if phrases is None:
            return Phrase([])
        return reduce(lambda x, y: x+y, phrases, Phrase([]))

    def get_children(self, as_set=False):
        '''children of a phrase are all of their individual children minus their own inner chunk'''
        all_children = set()
        for token in self.tokens:
            all_children |= set(token.children)
        
        all_children -= set(self.tokens)
        if not as_set:
            all_children = list(all_children)
            all_children.sort(key=_by_index)
        return all_children

    def has_trigger(self, triggers):
        string = self.as_str()
        for t in triggers:
            # check if it's in the string
            if string.find(t) != -1:
                return True
        return False

    def starts_with_trigger(self, triggers):
        string = self.as_str()
        for t in triggers:
            # check if it's at the front of the string
            if string.find(t) == 0:
                return True
        return False
    
    def __getitem__(self, key):
        return self.tokens[key]
    
    def __len__(self):
        return len(self.tokens)
    
    def __str__(self):
        return self.as_str(lower=False)
    
    def __add__(self, other):
        return Phrase(self.tokens + other.tokens if other is not None else [], has_skips=True)
    
    def __sub__(self, other):
        if type(other) is Phrase:
            blacklist = set(other.tokens)
            return Phrase([t for t in self.tokens if t not in blacklist], has_skips=True)
        else: #type(other) is Token:
            return Phrase([t for t in self.tokens if t != other], has_skips=True)


# Extracted contains the extracted subject phrases, verb phrases, etc.
class Extracted:
    def __init__(self, whole_phrase, subject_phrases=None, signal_word=None, verb_phrase=None, times=None, conditions=None, objects=None):
        '''
        whole_phrase, subject_phrases, times, conditions: [Phrase];
        signal_word, verb_phrase: Phrase;
        objects: [Extracted] OR [Phrase]
        '''
        self.whole_phrase = whole_phrase
        self.subject_phrases = subject_phrases
        self.signal_word = signal_word
        self.verb_phrase = verb_phrase
        self.times = times
        self.conditions = conditions
        self.objects = objects

    def as_tuple(self):
        return (self.whole_phrase, self.subject_phrases, self.signal_word, self.verb_phrase, self.times, self.conditions, self.objects)

    def as_row(self):
        def str_or_empty(p):
            return str(p) if p is not None else ''

        wp = str_or_empty(self.whole_phrase)
        sp = Phrase.merge_as_str(self.subject_phrases)
        sw = str_or_empty(self.signal_word)
        vp = str_or_empty(self.verb_phrase)
        t = Phrase.merge_as_str(self.times)
        c = Phrase.merge_as_str(self.conditions)
        
        op, op_subj, op_sig, op_verb, op_time, op_cond, op_obj = [], [], [], [], [], [], []
        for object in self.objects:
            if type(object) is Extracted:
                obj = object.as_tuple()
                op.append(str_or_empty(obj[0]))
                op_subj.append(Phrase.merge_as_str(obj[1]))
                op_sig.append(str_or_empty(obj[2]))
                op_verb.append(str_or_empty(obj[3]))
                op_time.append(Phrase.merge_as_str(obj[4]))
                op_cond.append(Phrase.merge_as_str(obj[5]))
                op_obj.append(Phrase.merge_as_str(obj[6]))
            else: # type(object) is Phrase
                op.append(str_or_empty(object))
        op = Phrase.merge_as_str(op)
        op_subj = Phrase.merge_as_str(op_subj)
        op_sig = Phrase.merge_as_str(op_sig)
        op_verb = Phrase.merge_as_str(op_verb)
        op_time = Phrase.merge_as_str(op_time)
        op_cond = Phrase.merge_as_str(op_cond)
        op_obj = Phrase.merge_as_str(op_obj)

        return (wp, sp, sw, vp, t, c, op, op_subj, op_sig, op_verb, op_time, op_cond, op_obj)
        
    def as_span(self):
        sp = Phrase.merge(self.subject_phrases).as_span('SUBJECT')
        sw = self.signal_word.as_span('SIGNAL')
        vp = self.verb_phrase.as_span('VERB')
        t = Phrase.merge(self.times).as_span('TIME')
        c = Phrase.merge(self.conditions).as_span('CONDITION')
        
        op, op_subj, op_sig, op_verb, op_time, op_cond, op_obj = [], [], [], [], [], [], []
        for object in self.objects:
            if type(object) is Extracted:
                obj = object.as_tuple()
                op.append(obj[0])
                op_subj.append(Phrase.merge(obj[1]))
                op_sig.append(obj[2])
                op_verb.append(obj[3])
                op_time.append(Phrase.merge(obj[4]))
                op_cond.append(Phrase.merge(obj[5]))
                op_obj.append(Phrase.merge(obj[6]))
            else: # type(object) is Phrase
                op.append(object)
        op = Phrase.merge(op).as_span('OBJECT')
        op_subj = Phrase.merge(op_subj).as_span('OP_SUBJECT')
        op_sig = Phrase.merge(op_sig).as_span('OP_SIGNAL')
        op_verb = Phrase.merge(op_verb).as_span('OP_VERB')
        op_time = Phrase.merge(op_time).as_span('OP_TIME')
        op_cond = Phrase.merge(op_cond).as_span('OP_CONDITION')
        op_obj = Phrase.merge(op_obj).as_span('OP_OBJECT')

        return (sp, sw, vp, t, c, op, op_subj, op_sig, op_verb, op_time, op_cond, op_obj)


# helper functions
def _by_depth(token):
    return depth_of[token]
def _by_index(token):
    return token.i
def _find_root(doc):
    for token in doc:
        if token.dep_.lower() in ['root']:
            return token
def _phrases_from_roots(roots):
    '''given a list of tokens, return a list of its subtrees as phrases (thus, the tokens are roots of the subtrees)'''
    return [Phrase(root) for root in roots]
def _tokens_with_dep(tokens, deps):
    return [t for t in tokens if t.dep_ in deps]
def _expand_subtree(token):
    tokens = [child for child in token.subtree]
    tokens.sort(key=_by_index) # sort in the order of appearance in the sentence
    return tokens
# depth of each token in the dependency tree (e.g. depth_of[root] = 0)
depth_of = dict()
def _assign_depth(token, current_depth=0):
    depth_of[token] = current_depth
    # assign the next depth to all children recursively
    for child in token.children:
        _assign_depth(child, current_depth+1)
def _last_pos_of(string, substrings):
    """find the last occurrence of a substring in a string"""
    last_pos, substr = max([(string.lower().rfind(s), s) for s in substrings], key=lambda pair: pair[0], default=(-1, ''))
    substr = string[last_pos:last_pos+len(substr)]
    return last_pos, substr

def add_span_label_vocabs(nlp):
    """add the span labels to the vocabs of nlp"""
    nlp.vocab.strings.add('OP_SUBJECT')
    nlp.vocab.strings.add('OP_SIGNAL')
    nlp.vocab.strings.add('OP_VERB')
    nlp.vocab.strings.add('OP_TIME')
    nlp.vocab.strings.add('OP_CONDITION')
    nlp.vocab.strings.add('OP_OBJECT')

def find_verb_deco(root):
    # find verb (root) and its decorations, e.g. be, not, etc.
    direct_decos = _tokens_with_dep(root.children, dep_verb) + [root]
    # decide whether the adverbial modifier is a verb or a condition
    for token in _tokens_with_dep(root.children, ['advmod']):
        if not Phrase(token).has_trigger(condition_trigger):
            direct_decos.append(token)   
    direct_decos.sort(key=_by_index)
    # recursively find all of my decorations's own decorations (e.g. know not very well)
    all_decos = []
    for deco in direct_decos:
        if deco == root:
            all_decos.append(root)
        else:
            all_decos += find_verb_deco(deco)
    return all_decos

def find_times_conds(potential_cond):
    times = []
    conds = []
    for cond_phrase in potential_cond:
        if len(cond_phrase) > 1:
            if cond_phrase.has_trigger(condition_trigger) and cond_phrase.as_str().strip() not in condition_trigger:
                if cond_phrase.has_trigger(time_point):
                    times.append(cond_phrase)
                else:
                    conds.append(cond_phrase)
    return times, conds

def extract(root, extract_obj_recursively=True):
    '''
    extract the whole phrase, subject, signal word, verb, time, condition, and object
    if extract_obj_recursively is True, then objects is a list of Extracted
    if extract_obj_recursively is False, then objects is a list of Phrase
    '''
    # find verb phrase
    verb_phrase = Phrase(find_verb_deco(root), has_skips=True)
    # find signal word
    signal_word = Phrase(_tokens_with_dep(root.children, dep_signal))
    if signal_word.as_str().strip() not in signal: # not in the whitelist, add to verb instead
        verb_phrase += signal_word
        verb_phrase.tokens.sort(key=_by_index)
        signal_word = Phrase([])
    # subject, object and conditions should be a child of the verb phrase
    verb_phrase_children = verb_phrase.get_children()
    # find subject phrases
    subj_phrases = _phrases_from_roots(_tokens_with_dep(verb_phrase_children, dep_subj))
    # find object phrases
    obj_phrases = _phrases_from_roots(_tokens_with_dep(verb_phrase_children, dep_obj))
    # prep could either be object or condition/time
    for phrase in _phrases_from_roots(_tokens_with_dep(verb_phrase_children, ['prep'])):
        if not phrase.starts_with_trigger(condition_trigger):
            obj_phrases.append(phrase)
    # find conditions and time constraints
    potential_cond_roots = set(_tokens_with_dep(verb_phrase_children, dep_cond)) - set(obj.root for obj in obj_phrases)
    times, conds = find_times_conds(_phrases_from_roots(potential_cond_roots))

    # for each objects (given their roots), extract their subj, signal, verb, conds, times, and obj
    objects = []
    if extract_obj_recursively:
        for obj_phrase in obj_phrases:
            # clausal compliments we can directly extract as sentence
            if obj_phrase.root.dep_ in ['ccomp', 'xcomp']:
                extracted_obj = extract(obj_phrase.root, extract_obj_recursively=False)
                # if the extracted part has no new subject, the verbs are mergeable and there's only 1 object, merge the verbs with the main and re-extract
                extracted_signal_verb = extracted_obj.signal_word + extracted_obj.verb_phrase
                if extracted_obj.subject_phrases == [] \
                    and verb_phrase[-1].nbor(1) == extracted_signal_verb[0] \
                    and len(extracted_obj.objects) == 1:
                    verb_phrase += extracted_signal_verb
                    times += extracted_obj.times
                    conds += extracted_obj.conditions
                    # extract one layer deeper to get the object's object
                    extracted_obj = extract(extracted_obj.objects[0].root, extract_obj_recursively=False)
                objects.append(extracted_obj)

            # non-clausal objects might still contain relative clauses (or acl) or conditions
            else:
                # look for a relative clause and only take the highest level clause, in case it is nested (lowest depth)
                rel_clause = min(_tokens_with_dep(obj_phrase, ['relcl', 'acl']), default=None, key=_by_depth)
                if rel_clause is not None:
                    rel_clause = Phrase(rel_clause)
                    # extract only the relative clause (not the whole object phrase!)
                    extracted_obj = extract(rel_clause.root, extract_obj_recursively=False)
                    # analyze the rest of the object phrase
                    rest_of_obj_phrase = obj_phrase - rel_clause # obj phrase WITHOUT the rel clause
                    if obj_phrase.root.dep_ == 'prep': # for prepositional object phrases, remove the preposition (e.g. for)
                        rest_of_obj_phrase -= obj_phrase.root
                    # find any potential conditions in the rest of the object phrase
                    t, c = find_times_conds(_phrases_from_roots(_tokens_with_dep(rest_of_obj_phrase, dep_cond)))
                    t = [phrase for phrase in t if phrase.starts_with_trigger(condition_trigger)]
                    c = [phrase for phrase in c if phrase.starts_with_trigger(condition_trigger)]
                    extracted_obj.times += t
                    extracted_obj.conditions += c
                    # find the referred noun phrase
                    ref_noun_phrase = rest_of_obj_phrase - (Phrase.merge(t) + Phrase.merge(c))
                    # find the relative word (which, who, etc.) and prepend it with the referred noun phrase
                    for phrases in [extracted_obj.subject_phrases, extracted_obj.objects]:
                        for i, phrase in enumerate(phrases):
                            # find the position of the first relative word in the phrase (if any)
                            rel_index = next((i for i, t in enumerate(phrase) if Phrase(t).has_trigger(relative_words)), None)
                            if rel_index is not None:
                                phrases[i] = Phrase(phrase[:rel_index]) + ref_noun_phrase + Phrase(phrase[rel_index:])
                    # only the relative clause was extracted, but the whole phrase should still be the whole object phrase
                    extracted_obj.whole_phrase = obj_phrase
                    objects.append(extracted_obj)
                else:
                    # no clauses found to extract, but still look for conditions and times
                    t, c = find_times_conds(_phrases_from_roots(_tokens_with_dep(obj_phrase, dep_cond)))
                    t = [phrase for phrase in t if phrase.starts_with_trigger(condition_trigger)]
                    c = [phrase for phrase in c if phrase.starts_with_trigger(condition_trigger)]
                    objects.append(Extracted(obj_phrase, None, None, None, t, c, None))
    else: # do not analyse my objects, instead just return the object phrases
        objects = obj_phrases
    return Extracted(Phrase(root), subj_phrases, signal_word, verb_phrase, times, conds, objects)


## Apply functions to create phrase span addition for pipeline ############################START
def extract_sentence(doc):
    root = _find_root(doc)
    _assign_depth(root)
    return extract(root)

# register custom extension attributes
Doc.set_extension('extracted', default=None)
Doc.set_extension('replacements', default=None)

@Language.component('phrase_spans')
def phrase_spans(doc):
    root = _find_root(doc)
    _assign_depth(root)
    extracted = extract(root)
    spans = reduce(lambda x,y: x+y, extracted.as_span(), []) # all spans in one list
    doc.spans['sc'] = spans # create a new span group in the doc
    doc._.extracted = extracted # store the extracted sentence in the doc
    return doc

phrase_spans.key = 'sc'
########################################################################################## END

