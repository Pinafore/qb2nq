import logging
import argparse
import json
from collections import Counter

import traceback
import re
from collections import defaultdict
from collections import Iterable

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from nltk import Tree

import spacy
import neuralcoref

nlp = spacy.load('en_core_web_sm')
neuralcoref.add_to_pipe(nlp)

def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
    else:
        return node.orth_


# Heuristics for NQlike quality checking
class HeuristicsTransformer:
  def __init__(self, config, lat_lookup):
    self.current_analysis = {}    
    self.valid_verbs = config["valid_verbs"]
    self.wh_words = config["wh_words"]
    self.strictly_valid_verbs = config["strictly_valid_verbs"]
    self.to_trim = config["to_trim"]
    self.pos_pronouns = config["pos_pronouns"]
    self.pronouns = config["pronouns"]
    self.imperative_pattern = {re.compile(x) for x in config["imperative"]}
    self.regexp_trims = dict((re.compile(x), y) for x, y in config["remove_dict"].items())
    self.non_last_sent_transform_dict = config["non_last_sent_transform_dict"]
    self.answer_type_dict = lat_lookup

class add_question_word_if_no_pronouns:
    def __init__(self, qb_id: int, question: str, lexical_answer_type: str, question_determiner: str):
        #-> Iterable[str]: Cannot force return type because of error 'ABCMeta' object is not subscriptable
        
        # input: questions after the parse tree steps and before transformation
        q = question[0].lower()+question[1:]
    
        question_test = self.current_analysis[question]["spacy"]
        pronouns_tags = {"PRON", "WDT", "WP", "WP$", "WRB", "VEZ"}
        # check whether there are any pronouns or not in the sentence q
        flag = True
        for token in question_test:
          if token.tag_ in pronouns_tags:
            flag = False
            break
    
        if flag == True:
          # no pronouns in the question
    
          # check wether answer type is singular or plural
          answer_type = self.answer_type_dict[qb_id]
          processed_text = nlp(answer_type)
          lemma_tags = {"NNS", "NNPS"}
    
          sigular_plural_flags = True # singular
          for token in processed_text:
            if token.tag_ == 'NNPS':
              sigular_plural_flags = False # plural
              break
    
          # check if the first toke is VERB
          if question_test[0].pos_ == 'VERB' and question_test[1].pos_ != 'PART' and question_test[2].pos_ != 'AUX':
            replacement = 'which '+answer_type+' '
            q = replacement+q
          else:
            if sigular_plural_flags == False:
              # plural
              replacement = 'which '+answer_type+' are '
              q = replacement+q
            else:
              # singular
              replacement = 'which '+answer_type+' is '
              q = replacement+q
        # capitalize the first letter of each sentence
        q = q[0].upper()+q[1:]
        yield q
        
  # Heuristic 1 remove punctuation patterns at the beginning and the end of the question [" ' ( ) , .]
class remove_regexp_patterns:
    def __init__(self, qb_id: int, question: str, lexical_answer_type: str, question_determiner: str):
        #-> Iterable[str]: Cannot force return type because of error 'ABCMeta' object is not subscriptable
        """
        Remove punctuation patterns at the beginning and the end of the question
        """
    
        question = question
        for pattern, replacement in self.regexp_trims.items():
          question = pattern.sub(replacement, question)
        yield question.replace("  ", "").strip()

  # Heuristic 2 -- name this answer type correction
class imperative_to_question:
    def __init__(self, qb_id: int, question: str, lexical_answer_type: str, question_determiner: str):
        #-> Iterable[str]: Cannot force return type because of error 'ABCMeta' object is not subscriptable
        """
        Convert "-- name this" patterns to "which"
        """
        for pattern in self.imperative_pattern:
          if pattern.search(question):
            parse = self.current_analysis[question]["spacy"]
            # find the mention, the first noun after identify, name, or give
            verb_position = min(x.i for x in parse if x.text.lower() in ["name", "give", "identify"])
            mention = parse[verb_position + 1].head
            
            # is there a relative clause or an appositive?
            if 'relcl' in [x.dep_ for x in mention.children]:
              # find the relative clause head
              relative_head = [x for x in mention.children if x.dep_ == "relcl"]
              if len(relative_head) > 1:
                logging.warn("Two relative clauses for an 'identify' construction, and we don't know how to handle that")
                return
              relative_head = relative_head[0]
              continuation = " ".join(x.text for x in parse[relative_head.left_edge.i+1:relative_head.right_edge.i+1])
              yield "%s %s %s" % (question_determiner, lexical_answer_type, continuation)
            elif len(parse) > mention.i + 1 and parse[mention.i + 1].text == ',':
              # If there is an appostive, then turn it into a question
              continuation = " ".join(x.text for x in parse[(mention.i + 2):])
              yield "%s is %s" % (lexical_answer_type, continuation)   
            else:
              # If not, just cut the "For 10 ... points [name/identify]" and yield that
              reduced = pattern.sub("", question).strip()
              yield reduced
    
              # and the question version
              yield "%s is the %s" % (question_determiner, reduced)

  # Heuristic 3 semicolon
class drop_after_punctuation:
    def __init__(self, qb_id: int, question: str, lexical_answer_type: str, question_determiner: str):
        #-> Iterable[str]: Cannot force return type because of error 'ABCMeta' object is not subscriptable
        """
        Remove contents after semicolon in NQlike
        """
        for pattern in [re.compile("[;,!?.].*$"), re.compile("^[;,!?.].*")]:
          if pattern.search(question):
            question = pattern.sub('', question)
            yield question
    
  
  # Heuristic 5 remove repetition of the subject âis thisâ
class count_num_of_verbs:
    def __init__(self,text, strictly = False):
        """
        count the number of verbs
        """
        verb_tags = []
        if strictly:
          verb_tags = self.strictly_valid_verbs
        else:
          verb_tags = self.valid_verbs
        tokens = self.current_analysis[text]["nltk_tokens"]
        tagged = self.current_analysis[text]["nltk_tags"]
        counted = Counter(tag for word,tag in tagged)
        num_of_verb = 0
        for v in verb_tags:
          num_of_verb = num_of_verb + counted[v]
        return num_of_verb

  def remove_rep_subject(self, qb_id: int, question: str, lexical_answer_type: str, question_determiner: str):
    #-> Iterable[str]: Cannot force return type because of error 'ABCMeta' object is not subscriptable
    """
    remove is this... pattern
    """
    to_clean = " is this [a-zA-Z]*\s"
    if re.search(to_clean, question):
      # the sentence has to have 1 verb at least otherwise this will not be done
      if (self.count_num_of_verbs(question) > 1):
        question = re.sub(to_clean, ' ', question)
    yield question

  # Heuristic 6 change be determiner to s possession
  def remove_BE_determiner(self, qb_id: int, question: str, lexical_answer_type: str, question_determiner: str):
    #-> Iterable[str]: Cannot force return type because of error 'ABCMeta' object is not subscriptable
    """
    change is his/is her/is its to 's
    """
    to_clean = "( is his )|( is her )|( is its )"
    if re.search(to_clean, question):
      question = re.sub(to_clean, '\'s ', question)
    yield question

  # function to add space before punctuation
  def add_space_before_punctuation(self, qb_id: int, question: str, lexical_answer_type: str, question_determiner: str):
    #-> Iterable[str]: Cannot force return type because of error 'ABCMeta' object is not subscriptable
    """
    add space before punctuation because in NQ there's space before all types of punctuation
    """
    tokens = self.current_analysis[question]["nltk_tokens"]
    q = ' '.join(tokens)
    yield q

  def fix_no_verb(self, qb_id: int, question: str, lexical_answer_type: str, question_determiner: str):
    # -> Iterable[str]: Cannot force return type because of error 'ABCMeta' object is not subscriptable
    if (self.count_num_of_verbs(question, True) == 0):
      tokens = self.current_analysis[question]["nltk_tokens"]
      tagged = self.current_analysis[question]["nltk_tags"]
      ind = 0

      for tk,tg in tagged:
        if tg == 'NN' or tg == 'NNP':
          tokens.insert(ind+1,'is')
          break
        elif tg == 'NNS' or tg == 'NNPS':
          tokens.insert(ind+1,'are')
          break
        ind = ind + 1
        
      yield ' '.join(tokens)
    else:
      yield question

  # Heuristic 8 remove repetitive be verb when there's more verbs
  def remove_repeat_verb(self, qb_id: int, question: str, lexical_answer_type: str, question_determiner: str):
    #-> Iterable[str]: Cannot force return type because of error 'ABCMeta' object is not subscriptable
    """
    remove is he/is she/is it
    """
    to_clean = "( is he )|( is she )|( is it )"
    if re.search(to_clean, question):
      if (self.count_num_of_verbs(question) > 1):
        question = re.sub(to_clean, ' ', question)
    yield question

  # Heuristic 9 First verb after which in continuous tense
  def convert_continuous_to_present(self, qb_id: int, question: str, lexical_answer_type: str, question_determiner: str):
    #-> Iterable[str]: Cannot force return type because of error 'ABCMeta' object is not subscriptable
    """
    if the first verb is in continuous tense, change it to nomal
    """
    verb_tags = self.valid_verbs
    tokens = self.current_analysis[question]["nltk_tokens"]
    tagged = self.current_analysis[question]["nltk_tags"]

    ind = 0
    for tk,tg in tagged:
      if tg in verb_tags:
        if tg == 'VBG':
          try:
            old_tk, old_tg = tagged[ind-1]
            if old_tg == 'NN' or old_tg == 'NNP':
              tokens[ind] = re.sub('ing','s',tokens[ind])
              question = ' '.join(tokens)
            else:
              tokens[ind] = re.sub('ing','',tokens[ind])
              question = ' '.join(tokens)
          except:
            break
          break
        else:
            break
      ind = ind + 1
    yield question

  # Heuristic11 convert this to which
  def no_wh_words(self, qb_id: int, question: str, lexical_answer_type: str, question_determiner: str):
    #-> Iterable[str]: Cannot force return type because of error 'ABCMeta' object is not subscriptable
    result = question
    wh_words = self.wh_words
    wh_re = re.compile("|".join(wh_words))
    if not wh_re.search(question):
      # no wh_words
      if len(question) > 1 and qb_id in self.answer_type_dict:
          answer_type = self.answer_type_dict[qb_id] # get the answer type from qb_id
          # whether starting from VERB or not
          wn_list = wn.synsets(question.split()[0])
          if not wn_list==[]:
              tag = wn.synsets(question.split()[0])[0].pos()
              if tag == 'v':
                  result = 'which '+answer_type+question
              else:
                  result = 'which '+answer_type+' is '+question
      else:
          if not qb_id in self.answer_type_dict:
              logging.warn("Missing answer type %i" % qb_id)
          result = re.sub('this', 'which', question, 1)
    yield result

  # Heuristic12
  def replace_this_is(self, qb_id: int, question: str, lexical_answer_type: str, question_determiner: str):
    #-> Iterable[str]: Cannot force return type because of error 'ABCMeta' object is not subscriptable
    """
    Replace 'this' to 'which'+answer_type within 'this is' pattern.
    """
    x = question
    index = x.find('this is')
    if index!=-1:
      # adding answer type
      if qb_id in self.answer_type_dict:
        answer_type = self.answer_type_dict[qb_id] # get the answer type from qb_id
        replacement = 'which '+answer_type
        result = re.sub('this is', replacement+' is', x, 1)
        question = result
      else:
        logging.warn("Missing answer type %i" % qb_id)
        result = re.sub('this', 'which', x, 1)
        question = result
    yield question

  # Heuristic14: double auxiliary words
  def remove_extra_AUX(self, qb_id: int, question: str, lexical_answer_type: str, question_determiner: str):
    #-> Iterable[str]: Cannot force return type because of error 'ABCMeta' object is not subscriptable
    """
    Remove extra auxiliary words.
    """
    x = question
    doc_dep = self.current_analysis[question]["spacy"]
    lemma_lst = []
    tokem_text_lst = []
    for k in range(len(doc_dep)):
      lemma_lst.append(doc_dep[k].lemma_)
      tokem_text_lst.append(doc_dep[k].text)
    if lemma_lst.count('be') == 2:
      index = lemma_lst.index('be')
      if lemma_lst[index+1] == '-PRON-' and lemma_lst[index+2] == 'be':
        # two non-conjunctional be verbs with pronoun in between
        del tokem_text_lst[index+1]
        del tokem_text_lst[index+1]
        result = " ".join(tokem_text_lst)
        question = result
      else:
        # two conjunction BE verbs or two non-conjunctional be verbs without pronoun in between
        del tokem_text_lst[index]
        result = " ".join(tokem_text_lst)
        question = result
    yield question

  # Heuristic15: WDT+BE patterns
  def replace_which_with_this(self, qb_id: int, question: str, lexical_answer_type: str, question_determiner: str):
    #-> Iterable[str]: Cannot force return type because of error 'ABCMeta' object is not subscriptable
    """
    Convert 'which' to 'that' and check if no 'which' present anymore, if so, convert 'this' to 'which'.
    """
    x = question
    index1 = x.find('which is where')
    index2 = x.find('which is why')
    if index1 != -1:
      result = re.sub('which is where', 'that is where', x)
      q = result
    elif index2 != -1:
      result = re.sub('which is why', 'that is why', x)
      question = result
    else:
      result = x
      # check if no 'which' present anymore
    index = result.find('which')
    if index==-1:
      result = re.sub('this', 'which', result, 1)
      question = result
    yield question

  def rejoin_contractions(self, qb_id: int, question: str, lexical_answer_type: str, question_determiner: str):
    # -> Iterable[str]: Cannot force return type because of error 'ABCMeta' object is not subscriptable
    for separated, together in {"who 's": "whose", "ca n't": "can't", "wo n't": "won't"}.items():
      if separated in question:
        question = question.replace(separated, together)
    yield question

  def split_conjunctions(self, qb_id: int, question: str, lexical_answer_type: str, question_determiner: str):
    #-> Iterable[str]: Cannot force return type because of error 'ABCMeta' object is not subscriptable
    # First, find the verbs 
    parse = self.current_analysis[question]["spacy"]
    
    root_verb = [x for x in parse if x.pos_ == "VERB" and not any(1 for _ in x.ancestors)]
    verbs = [x for x in parse if x.pos_ == "VERB" and x.head in root_verb]

    # See if they have any coordinating conjunctions as dependents
    verb_conj = set()
    for verb in verbs:
      for child in verb.children:
        if child.dep_ == 'cc' and child.pos_ == "CCONJ":
          verb_conj.add((verb, child))

    if len(verb_conj) > 1:
      logging.warn("Multiple conjunctions in sentence and we don't know what to do: " + question)
      return

    # If so, then we need to know if they are independent clauses
    for verb, conj in verb_conj:
      # Check to see if this is the second verb and if it has no ancestors
      if verb.i > verbs[0].i and not any(1 for _ in verb.ancestors):
        # If so, we have two independent clauses, so yield the two
        # parts on either side of the conjunction
        yield " ".join(x.text for x in parse if x.i < conj.i)
        yield " ".join(x.text for x in parse if x.i > conj.i)
      elif verb.i < verbs[-1].i and verbs[-1].dep_ == "conj":
        # Otherwise, if this verb is child of another verb with "conj"
        # relation, we can have two sentences with the same subject

        # Get what came before verb and doesn't modify verb
        left_tokens = [x for x in parse if x.i < verb.i and not
                         (x.head == verb and (x.pos_ == "ADVERB" or x.pos_ == "AUX"))]

        # Get possible completions
        first_verb = [x for x in parse if x.i < conj.i and not x in left_tokens]
        second_verb = [x for x in parse if x.i > conj.i]

        # Return those
        yield " ".join(x.text for x in left_tokens + first_verb)
        yield " ".join(x.text for x in left_tokens + second_verb)
                                              
          
          
        
      
          

    # If there are nouns and verbs on both sides of it, the just iterate on those


    # If there are only verbs, duplicate the subject
    None
    
  # Heuristic16: 
  # WDT tag: which/what
  # WRB tag: where/why/when
  def add_question_word(self, qb_id: int, question: str, lexical_answer_type: str, question_determiner: str):
    #-> Iterable[str]: Cannot force return type because of error 'ABCMeta' object is not subscriptable
    """
    Adding 'which+answer_type' at the beginning when no WDT/WRB present.
    """
    x = question
    doc_dep = self.current_analysis[question]["spacy"]
    tag_lst = []
    tokem_text_lst = []
    for k in range(len(doc_dep)):
      tag_lst.append(doc_dep[k].tag_)
      tokem_text_lst.append(doc_dep[k].text)
    if ('WRB' in tag_lst)!=True and ('WDT' in tag_lst)!=True:
      # adding answer type at the beginning
      qb_id = str(qb_id)
      if qb_id in self.answer_type_dict:
        answer_type = self.answer_type_dict[qb_id] # get the answer type from qb_id
        result = 'which '+answer_type+' is '+x
        question = result
      else:
        logging.warn(str(qb_id) + 'is not in the frequency table!')
    yield question

  # Heuristic17: VERB/AUX at the beginning of the sample while missing the object
  def add_subject(self, qb_id: int, question: str, lexical_answer_type: str, question_determiner: str):
    #-> Iterable[str]: Cannot force return type because of error 'ABCMeta' object is not subscriptable
    """
    Adding 'which+answer_type' at the beginning when starting with VERB/AUX and missing the subject.
    """

    doc_dep = self.current_analysis[question]["spacy"]

    if len(doc_dep) > 1 and doc_dep[0].pos_ in {"AUX", "VERB"}:
      if qb_id in self.answer_type_dict:
        yield "which %s %s" % (self.answer_type_dict[qb_id], question)
      else:
        logging.warn(str(qb_id) + 'is not in the frequency table!')
        
    assert not question.startswith("which none is")
    yield question

  # Heuristic18: 'which none is' patterns
  def which_none_is(self, qb_id: int, question: str, lexical_answer_type: str, question_determiner: str):
    #-> Iterable[str]: Cannot force return type because of error 'ABCMeta' object is not subscriptable
    """
    Convert 'which none is' to 'what is'.
    """
    x = question
    index = x.find('which none is')
    if index != -1:
      qb_id = str(qb_id)
      if qb_id in self.answer_type_dict.keys():
        answer_type = self.answer_type_dict[qb_id] # get the answer type from qb_id
        result = re.sub('which none is', 'which '+answer_type+' is', x)
        question = result
      else:
        logging.warn(qb_id+'is not in the frequency table!')
    yield question

  # Heuristic19: 'what is which' pattern
  def what_is_which(self, qb_id: int, question: str, lexical_answer_type: str, question_determiner: str):
    #-> Iterable[str]: Cannot force return type because of error 'ABCMeta' object is not subscriptable
    """
    Remove "what is" from "what is which".
    """
    x = question
    index = x.find('what is which')
    if index != -1:
      result = re.sub('what is which', 'which', x)
      question = result
    yield question

  def cache_analysis(self, question, verbose=False):
    """
    Parse the sentence for downstream heuristics.  This prevents repeated computation in each of the heuristics.
    """
    tokens = nltk.word_tokenize(question.lower())
    text = nltk.Text(tokens)
    tagged = nltk.pos_tag(text)

    spacy_parse = nlp(question)
    if verbose:
      for ii in spacy_parse.sents:
        logging.debug(to_nltk_tree(ii.root))
          
    self.current_analysis[question] = {"spacy": spacy_parse, "nltk_tokens": tokens, "nltk_tags": tagged}

    
  def __call__(self, qb_id, answer, chunk_id, question, lexical_answer_phrase, question_determiner, suppress_errors=False):
    self.current_analysis = {}
    finished = set()
    applied_transformations = []
    self.cache_analysis(question)

    while len(finished) < len(self.current_analysis):
      unchecked = [x for x in self.current_analysis if x not in finished]
      for qq in unchecked:
        assert not qq in finished
        for method_name, replace in [("split_conjunctions", False),
                                     ("imperative_to_question", True),                              
                                     ("remove_regexp_patterns", False),
                                     ("drop_after_punctuation", False),
                                     ("convert_continuous_to_present", False),
                                     ("no_wh_words", False),
                                     ("replace_this_is", False),
                                     ("replace_which_with_this", False),
                                     # "add_question_word",
                                     # "add_subject",
                                     ("which_none_is", False),
                                     ("what_is_which", True),
                                     # "remove_extra_AUX",
                                     ("remove_rep_subject", True),
                                     ("remove_BE_determiner", True),
                            # "fix_no_verb",
                                     ("add_space_before_punctuation", True),
                                     ("rejoin_contractions", True)]:
          method = getattr(self, method_name)

          if suppress_errors:
            try:
              results = method(qb_id, qq, lexical_answer_phrase, question_determiner)
            except Exception as exc:
              logging.error(traceback.format_exc())
              logging.error(exc)
              results = []
              continue  
          else:
            results = method(qb_id, qq, lexical_answer_phrase, question_determiner)

          for new_question in results:
              if new_question != qq:
                logging.debug("Adding new question [%s]: %s" % (method_name, new_question))

                row = {}
                row["qanta_id"] = qb_id
                row["original"] = question
                row["answer"] = answer
                row["parent"] = qq
                row["chunk_id"] = chunk_id
                row["question"] = new_question.lower()
                row["transform"] = method_name
                applied_transformations.append(row)
                
                self.cache_analysis(new_question)

          if replace:
            finished.add(qq)


    self.current_analysis = None
    return applied_transformations


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Demo heuristic functions")
  parser.add_argument('--qb_path', type=str,
                      default='qanta.train.2021.12.20.json',
                      help="path of the qb dataset")
  parser.add_argument('--config_file', type=str, default='config.json',
                      help="File with data that configures extraction")

  args = parser.parse_args()
  # Load dataset

  with open(args.config_file) as json_file:
    config = json.load(json_file)
  
  heuristics = HeuristicsTransformer(config, {0: "character", 1: "thing", 94: "ruler", 102: "novel", 104: "organ"})
  logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

  for qid, example in [(102, "An automobile in what novel was purchased in Fuller by Bessie for her marriage to 16-year - old Dude , and led to the death of both an African - American man and Grandmother Lester ."),
                       (94, "He demanded compensation for the family of Jacob Kaiser and forced another group to end its alliance with Austria in an armistice that he negotiated to end a war in which no battles occurred."),
                       (0, ' what character is first encountered in the Spouter - Inn where the landlord thinks he may be late because " he ca n\'t sell his head , " and his coffin helps save the narrator after the ship he \'s on sinks . \xa0'),
                       (19, "The protagonist of one of who 's works gives a jar to his friend instead of repaying a loan , and later dies after embezzling money in an attempt to buy out a courtesan 's contract ."),
                           (1, "'!$  % ##@# @@# ftp FTP ftp--- For 10 points, for 10 points , For 10 points --- for 20 points: for 10 points--this is a real, legit question !@#!@@#@%%%....?"),
                           (104, "For 10 points , name this organ , home to the islets of Langerhans ."),
                           (8, "For 10 points , identify this French government that succeeded the Second Empire ."),
                           (98, "For 10 points , identify this Sanskrit - language author of \" The Cloud Messenger \" and The Recognition of Shakuntala ."),
                           (59, "For 10 points , name this mortal lover of Zeus and mother of Dionysus ."),
                           (20, "For 10 points , name this Italian author of The Name of the Rose and Foucault 's Pendulum ."),
                           (17, "For 10 points , name this country , the home of the director of The Seventh Seal , Ingmar Bergman ."),
                           (119, "For 10 points , name this 1859 essay condemning infringements on personal freedom , written by John Stuart Mill ."),
                           (378, "Beginning with \" April is the cruellest month \" , for 10 points , name this long poem in five sections by T.S. Eliot ."),
                           (361,  " who asserted \" the air is a spongy body / a promiscuous faceless being \" in \" The Balcony \" and asked \" Do I believe in man / or in the stars ? \" in another poem .")]:

      transformations = heuristics(qid, "??", 0, example, "what thing", "what")
      for transform in transformations:
        logging.debug("===============================")
        logging.debug(transform["original"])
        logging.debug(transform["parent"])
        logging.debug(transform["transform"])           
        logging.debug(transform["question"])
    
