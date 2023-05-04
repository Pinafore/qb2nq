import re
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
import logging
import nltk
from syntax import *
import pyinflect
import spacy
from collections import Counter
from gingerit.gingerit import GingerIt
from syntax_check import *

nlp = spacy.load('en_core_web_sm')

class ConditionalHeuristic:
  def __init__(self, analysis):
    with open("config.json") as json_file:
      config = json.load(json_file)
    self.current_analysis = {}
    self.replace = True
    self.valid_verbs = config["valid_verbs"]
    ##print(self.valid_verbs)
    self.wh_words = config["wh_words"]
    ##print(self.wh_words)
    self.strictly_valid_verbs = config["strictly_valid_verbs"]
    ##print(self.strictly_valid_verbs)
    self.to_trim = config["to_trim"]
    self.pos_pronouns = config["pos_pronouns"]
    self.pronouns = config["pronouns"]
    self.regexp_trims = dict((re.compile(x), y) for x, y in config["remove_dict"].items())
    self.imperative_pattern = {re.compile(x) for x in config["imperative"]}
    self.non_last_sent_transform_dict = config["non_last_sent_transform_dict"]
    self.answer_type_dict = {0: "character", 1: "thing", 94: "ruler", 102: "novel", 104: "organ",19: "character",8:"government",98:"person",59:"person",20:"person",17:"country",119:"essay",378:"poem",904:"book"}
  
  def precondition(self, question):
    return True

  def postcondition(self, question):
    return True

def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
    else:
        return node.orth_

class add_question_word_if_no_pronouns(ConditionalHeuristic):
    def precondition(self, qb_id: int, question: str, lexical_answer_type: str, question_determiner: str):
      return True
    def __call__(self, qb_id: int, question: str, lexical_answer_type: str, question_determiner: str):
        #print ("Hello",lexical_answer_type)
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
class remove_regexp_patterns(ConditionalHeuristic):
    def my_name(cls_): 
      return cls_.__name__ 
    def precondition(self, qb_id: int, question: str, lexical_answer_type: str, question_determiner: str):
      #print()
      flag=False
      for pattern, replacement in self.regexp_trims.items():
        if pattern.search(question):
          #print("found pattern",pattern)
          return True
      return flag
    def __call__(self, qb_id: int, question: str, lexical_answer_type: str, question_determiner: str):
        #-> Iterable[str]: Cannot force return type because of error 'ABCMeta' object is not subscriptable
        """
        Remove punctuation patterns at the beginning and the end of the question
        """
        print("Remove regex patterns")
        flag=self.precondition(qb_id,question,lexical_answer_type,question_determiner)
        if flag==True:
          question = question
          #print("loop items: ",self.regexp_trims.items())
          for pattern, replacement in self.regexp_trims.items():
            #print("pattern ",pattern," and replacement",replacement)
            #print("Replacing: ",pattern)
            if pattern.search(question):
              question = pattern.sub(replacement, question)
            #print("Question: ",question)
          prev=question.replace("  ", "").strip()
          #print(prev)
          final_q=self.postcondition(prev)
          yield final_q
        else:
          yield question
    
    def postcondition(self,question):
      len1=GingerIt().parse(question)['result']
      return len1
  # Heuristic 2 -- name this answer type correction
class imperative_to_question(ConditionalHeuristic):
    def my_name(cls_): 
        return cls_.__name__ 
    
    def cache_analysis(self, question, verbose=False):
      """
      Parse the sentence for downstream heuristics.  This prevents repeated computation in each of the heuristics.
      """
      print("code in cachex")
      tokens = nltk.word_tokenize(question.lower())
      text = nltk.Text(tokens)
      tagged = nltk.pos_tag(text)

      spacy_parse = nlp(question)
      if verbose:
        for ii in spacy_parse.sents:
          logging.debug(to_nltk_tree(ii.root))
            
      self.current_analysis[question] = {"spacy": spacy_parse, "nltk_tokens": tokens, "nltk_tags": tagged}
      print(self.current_analysis[question]["spacy"])
    def precondition(self, qb_id: int, question: str, lexical_answer_type: str, question_determiner: str):
      #print("imperative",question)
      for pattern in self.imperative_pattern:
          if pattern.search(question):
            return True
      return False
    def __call__(self, qb_id: int, question: str, lexical_answer_type: str, question_determiner: str):
        #-> Iterable[str]: Cannot force return type because of error 'ABCMeta' object is not subscriptable
        """
        Convert "-- name this" patterns to "which"
        """
        #print("Heuristic 2: Imperative to question")
        self.cache_analysis(question)
        flag=self.precondition(qb_id,question,lexical_answer_type,question_determiner)
        if flag==True:
          for pattern in self.imperative_pattern:
            if pattern.search(question):
              parse = self.current_analysis[question]["spacy"]
              #print("parse",parse)
              # find the mention, the first noun after identify, name, or give
              """for pattern in self.imperative_pattern:
                if pattern.find(question):
                  print("printing test: ",pattern.search(question))"""
              #begin_word=x.i for x in parse if x.text in self.imperative_pattern
              #print("begin word",begin_word)
              verb_position = min(x.i for x in parse if x.text.lower() in ["name", "give", "identify"])
              #print("verb postition: ",verb_position)
              mention = parse[verb_position + 1].head
              #print("mention: ",mention)
              #print("length: ",len(parse))
              #print("mention+1: ",mention.i + 1)
              #print("next: ",parse[mention.i + 1].text)
              #len(parse) > mention.i + 1 and parse[mention.i + 1].text == ','
              
              # is there a relative clause or an appositive?
              if 'relcl' in [x.dep_ for x in mention.children]:
                # find the relative clause head
                child=mention.children
                """for x in child:
                  print("child",x)
                  print("x_dep::: ",x.dep_)"""
                #print("x_dep",(x.dep_ for x in mention.children))
                relative_head = [x for x in mention.children if x.dep_ == "relcl"]
                #print("relative head",relative_head)
                if len(relative_head) > 1:
                  #print("len",len(relative_head))
                  logging.warn("Two relative clauses for an 'identify' construction, and we don't know how to handle that")
                  return
                relative_head = relative_head[0]
                #print("left: ",relative_head.left_edge)
                #print("right: ",relative_head.right_edge)
                continuation = " ".join(x.text for x in parse[relative_head.left_edge.i+1:relative_head.right_edge.i+1])
                #print("continution::",continuation)
                #print(question_determiner, " and lex",lexical_answer_type)
                #print("Before postcondition: ")
                prev_q="%s %s" % (lexical_answer_type, continuation)
                #print(prev_q)
                #print("After postcondition")
                final_q=self.postcondition(prev_q)
                yield final_q
                first_part=" ".join(x.text for x in parse[:(mention.i)])
                #print("first_part: ",first_part)
                #question=first_part.replace(mention.text,"")
                question_determiner_mod=" "+question_determiner+" is"
                second_q=re.sub(pattern,question_determiner,first_part)
                #print("After removal: ",second_q)
                continuation2=" ".join(x.text for x in parse[(mention.i+2):])
                second_q=second_q+" "+mention.text+" "+continuation
                #second_q=
                #print("second q in elif: ",second_q)
                yield self.postcondition(second_q)
                #yield "%s %s %s" % (question_determiner, lexical_answer_type, continuation)
              elif len(parse) > mention.i + 1 and parse[mention.i + 1].text == ',':
                # If there is an appostive, then turn it into a question
                continuation1="".join(x.text for x in parse[:(mention.i)])
                #print("first part",continuation1)
                continuation = " ".join(x.text for x in parse[(mention.i + 2):])
                #print("conjunction: ",continuation)
                #print("lexical answer: ",lexical_answer_type)
                prev_q="%s is %s" % (lexical_answer_type, continuation)  
                #print("Before precondition")
                #print(prev_q)
                #print("After postcondition") 
                final_q=self.postcondition(prev_q)
                yield final_q
                first_part=" ".join(x.text for x in parse[:(mention.i)])
                #print("first_part: ",first_part)
                #question=first_part.replace(mention.text,"")
                if parse[mention.i+1].tag_=="IN":
                  #print("Found instance: ",parse[mention.i+1].tag_)
                  question_determiner_mod=" "+question_determiner+" is"
                  second_q=re.sub(pattern," "+question_determiner_mod,first_part)
                  add_part=second_q+" "+mention.text
                else:
                  question_determiner_mod=" "+question_determiner
                  second_q=re.sub(pattern," "+question_determiner_mod,first_part)
                  add_part=second_q+" "+mention.text+" is "
                #print("After removal: ",second_q)
                continuation2=" ".join(x.text for x in parse[(mention.i+2):])
                second_q=add_part+" "+continuation2
                #second_q=
                #print("second q in elif: ",second_q)
                yield self.postcondition(second_q)
                #yield "%s is %s" % (lexical_answer_type, continuation)   
              else:
                # If not, just cut the "For 10 ... points [name/identify]" and yield that
                continuation1="".join(x.text for x in parse[:(mention.i)])
                #print("first part",continuation1)
                #print("Before precondition")
                reduced = pattern.sub("", question).strip()
                #print(reduced)
                #print("After postcondition")
                final_q=self.postcondition(reduced)
                #yield final_q
                #yield reduced
                #print("Before precondition")
                prev_q="%s is the %s" % (question_determiner, reduced)
                #print(prev_q)
                #print("After postcondition")
                final_q=self.postcondition(prev_q)
                yield final_q
                first_part=" ".join(x.text for x in parse[:(mention.i)])
                #print("first_part: ",first_part)
                #question=first_part.replace(mention.text,"")
                if parse[mention.i+1].tag_=="IN":
                  #print("Found instance: ",parse[mention.i+1].tag_)
                  question_determiner_mod=" "+question_determiner+" is"
                  second_q=re.sub(pattern," "+question_determiner_mod,first_part)
                  add_part=second_q+" "+mention.text
                else:
                  question_determiner_mod=" "+question_determiner
                  second_q=re.sub(pattern," "+question_determiner_mod,first_part)
                  add_part=second_q+" "+mention.text+" is "
                #print("After removal: ",second_q)
                continuation2=" ".join(x.text for x in parse[(mention.i+1):])
                second_q=add_part+" "+continuation2
                #print("second q in else: ",second_q)
                yield self.postcondition(second_q)
                # and the question version
                #yield "%s is the %s" % (question_determiner, reduced)
    def postcondition(self,question):
      #len=syntax_checker(question)
      #print("length of syntax check: ",len)
      #return len
      return question
  # Heuristic 3 semicolon
class drop_after_punctuation(ConditionalHeuristic):
    def my_name(cls_): 
      return cls_.__name__ 
    def precondition(self, qb_id: int, question: str, lexical_answer_type: str, question_determiner: str):
      for pattern in [re.compile("[;,!?].*$"), re.compile("^[;,!?].*")]:
          if pattern.search(question):
            #print("patterns",pattern)
            return True
          else:
            return False
    def __call__(self, qb_id: int, question: str, lexical_answer_type: str, question_determiner: str):
        #-> Iterable[str]: Cannot force return type because of error 'ABCMeta' object is not subscriptable
        """
        Remove contents after semicolon in NQlike
        """
        print("Drop after punctuation")
        question_after=question
        flag=self.precondition(qb_id,question,lexical_answer_type,question_determiner)
        if flag==True:
          for pattern in [re.compile("[;,!?].*$"), re.compile("^[;,!?].*")]:
            question = pattern.sub('', question)
        #print("question before precondition: ",question)
        question_final=self.postcondition(question_after,question)
        yield question_final
        """for pattern in [re.compile("[;,!?.].*$"), re.compile("^[;,!?.].*")]:
          if pattern.search(question):
            question = pattern.sub('', question)
            yield question"""
    def postcondition(self,question,question_prev):
      temp="%s" %question_prev
      temp=re.sub('".*?"', '', temp)
      #print(temp)
      sent = nlp(temp)
      has_noun = 1
      has_verb = 1
      for token in sent:
        #print("dep: ",token.pos_)
        if token.pos_ in ["NOUN", "PROPN", "PRON","NUM","DET"]:
          #print("dep: ",token.dep_)
          if token.dep_=="nsubj" or token.dep_=="nsubjpass":
            has_noun -= 1
        elif token.pos_ == "VERB":
          has_verb -= 1
        elif token.pos_=="AUX" and token.dep_=="ROOT":
          has_verb -= 1
      if has_noun >= 1 or has_verb >= 1:
        return question
      check2=is_quote_ok(question_prev)
      #print("chech2",check2)
      if check2==True:
        return syntax_checker(question_prev)
      else:
        return syntax_checker(question)


  
  # Heuristic 5 remove repetition of the subject âis thisâ
def count_num_of_verbs(self,text, strictly = False):
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

class remove_rep_subject(ConditionalHeuristic):
    def my_name(cls_): 
      return cls_.__name__ 
    def count_num_of_verbs(self,text, strictly = False):
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

    def precondition(self, qb_id: int, question: str, lexical_answer_type: str, question_determiner: str):
      to_clean = " is this [a-zA-Z]*\s"
      #print("To clean", to_clean)
      if re.search(to_clean, question):
        # the sentence has to have 1 verb at least otherwise this will not be done
        if (self.count_num_of_verbs(question) > 1):
          return True
        else:
          return False
      else:
        return False
    def __call__(self, qb_id: int, question: str, lexical_answer_type: str, question_determiner: str):
    #-> Iterable[str]: Cannot force return type because of error 'ABCMeta' object is not subscriptable
      """
      remove is this... pattern
      """
      flag=self.precondition(qb_id,question,lexical_answer_type,question_determiner)
      if flag==True:
        to_clean = " is this [a-zA-Z]*\s"
        question = re.sub(to_clean, ' ', question)
        yield self.postcondition(question)
      else:
        yield question
    def postcondition(self,question):
      len=syntax_checker(question)
      #print("length of syntax check: ",len)
      return len
  # Heuristic 6 change be determiner to s possession
class remove_BE_determiner(ConditionalHeuristic):
    def my_name(cls_): 
      return cls_.__name__ 
    def precondition(self, qb_id: int, question: str, lexical_answer_type: str, question_determiner: str):
      to_clean = "( is his )|( is her )|( is its )"
      if re.search(to_clean, question):
        return True
      else:
        return False
    def __call__(self, qb_id: int, question: str, lexical_answer_type: str, question_determiner: str):
    #-> Iterable[str]: Cannot force return type because of error 'ABCMeta' object is not subscriptable
        """
        change is his/is her/is its to 's
        """
        flag=self.precondition(qb_id,question,lexical_answer_type,question_determiner)
        to_clean = "( is his )|( is her )|( is its )"
        if flag==True:
          question = re.sub(to_clean, '\'s ', question)
          yield self.postcondition(question)
        else:
          yield question
    def postcondition(self,question):
      #print("Question before pre: ",question)
      if question.find("\'s\'s")!=-1:
        question=question.replace("\'s\'s","\'s")
      len=syntax_checker(question)
      #print("length of syntax check: ",len)
      return len

  # function to add space before punctuation
class add_space_before_punctuation(ConditionalHeuristic):
    def my_name(cls_): 
      return cls_.__name__ 
    def __call__(self, qb_id: int, question: str, lexical_answer_type: str, question_determiner: str):
       #-> Iterable[str]: Cannot force return type because of error 'ABCMeta' object is not subscriptable
        """
        add space before punctuation because in NQ there's space before all types of punctuation
        """
        #print("before:",question)
        tokens = self.current_analysis[question]["nltk_tokens"]
        q = " ".join(tokens)
        #print("after:",q)
        yield q
    def postcondition(self,question):
      len=syntax_checker(question)
      #print("length of syntax check: ",len)
      return len

class fix_no_verb(ConditionalHeuristic):
  def __call__(self, qb_id: int, question: str, lexical_answer_type: str, question_determiner: str):
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
class remove_repeat_verb(ConditionalHeuristic):
  def __call__(self, qb_id: int, question: str, lexical_answer_type: str, question_determiner: str):
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
class convert_continuous_to_present(ConditionalHeuristic):
  def my_name(cls_): 
    return cls_.__name__ 
  def precondition(self, qb_id: int, question: str, lexical_answer_type: str, question_determiner: str):
    verb_tags = self.valid_verbs
    tokens = self.current_analysis[question]["nltk_tokens"]
    tagged = self.current_analysis[question]["nltk_tags"]
    parse = self.current_analysis[question]["spacy"]
    #print("tokens: ",tokens," tagged: ",tagged)
    flag=False
    ind = 0
    cur=0
    for tk,tg in tagged:
      cur=cur+1
      #print("tk: ",tk," tags ",tg)
      if tg in verb_tags:
        if tg == 'VBG':
            old_tk, old_tg = tagged[ind-1]
            for x in parse:
              #print("x",x.tag_, " and ",x.head.text)
              if x.head.text==tk:
                if x.tag_=='NN' or x.tag_=='NNP'or x.tag_=="NNS" or x.tag=='NNPS':
                  flag=True
                  prev_tk=parse[cur-2]
                  #print("token",prev_tk.text," and tags ",prev_tk.dep_," and tense ",prev_tk.tag_)
                  if prev_tk.dep_=='aux' or prev_tk.dep_=='auxpass':
                    flag=True
    return flag
  def __call__(self, qb_id: int, question: str, lexical_answer_type: str, question_determiner: str):
    #-> Iterable[str]: Cannot force return type because of error 'ABCMeta' object is not subscriptable
    """
    if the first verb is in continuous tense, change it to nomal
    """
    flag=self.precondition(qb_id,question,lexical_answer_type,question_determiner)
    if flag==True:
      verb_tags = self.valid_verbs
      tokens = self.current_analysis[question]["nltk_tokens"]
      tagged = self.current_analysis[question]["nltk_tags"]
      parse = self.current_analysis[question]["spacy"]
      #print("tokens: ",tokens," tagged: ",tagged)

      ind = 0
      cur=0
      for tk,tg in tagged:
        cur=cur+1
        #print("tk: ",tk," tags ",tg)
        if tg in verb_tags:
          if tg == 'VBG':
              old_tk, old_tg = tagged[ind-1]
              for x in parse:
                #print("x",x.tag_, " and ",x.head.text)
                if x.head.text==tk:
                  if x.tag_=='NN' or x.tag_=='NNP' or x.tag_=="NNS" or x.tag=='NNPS':
                    prev_tk=parse[cur-2]
                    #print("token",prev_tk.text," and tags ",prev_tk.dep_," and tense ",prev_tk.tag_)
                    if prev_tk.dep_=='aux' or prev_tk.dep_=='auxpass':
                      if prev_tk.tag_=='VBZ':
                        lem=nlp(parse[cur-1].lemma_)
                        for tok in lem:
                          replaced_verb = tok._.inflect("VBZ")
                      elif prev_tk.tag_=='VBP':
                        lem=nlp(parse[cur-1].lemma_)
                        for tok in lem:
                          replaced_verb = tok._.inflect("VBP")
                      elif prev_tk.tag_=='VBD':
                        lem=nlp(parse[cur-1].lemma_)
                        for tok in lem:
                          replaced_verb = tok._.inflect("VBD")
                          #print("replaced",replaced_verb," tokens: ",tokens[cur-1])
                      tokens[cur-1] = replaced_verb
                      question = ' '.join(tokens)
                      tokens[cur-2] = " "
                      question = ' '.join(tokens)
                      break
                      #print("found: ",x.text)
                      #tokens[cur-2] = re.sub(prev_tk.text," ",tokens[cur-2])
        ind = ind + 1
      final_q=self.postcondition(question)
      yield final_q
    else:
      yield question
  def postcondition(self,question):
      len=syntax_checker(question)
      #print("length of syntax check: ",len)
      return len
  # Heuristic11 convert this to which
class no_wh_words(ConditionalHeuristic):
  def my_name(cls_): 
      return cls_.__name__ 
  
  def cache_analysis(self, question, verbose=False):
      """
      Parse the sentence for downstream heuristics.  This prevents repeated computation in each of the heuristics.
      """
      print("code in cachex")
      tokens = nltk.word_tokenize(question.lower())
      text = nltk.Text(tokens)
      tagged = nltk.pos_tag(text)

      spacy_parse = nlp(question)
      if verbose:
        for ii in spacy_parse.sents:
          logging.debug(to_nltk_tree(ii.root))
            
      self.current_analysis[question] = {"spacy": spacy_parse, "nltk_tokens": tokens, "nltk_tags": tagged}
      print(self.current_analysis[question]["spacy"])
  def precondition(self, qb_id: int, question: str, lexical_answer_type: str, question_determiner: str):
    #print("question here",question.lower())
    flag=False
    wh_words = self.wh_words
    wh_re = re.compile("|".join(wh_words))
    #print(wh_re)
    if not wh_re.search(question.lower()):
      flag=True
    else:
      if question.lower().find("this")!=-1:
        flag=True
      else:
        flag=True
        temp_f=False
        sent=nlp(question.lower())
        for x in sent:
          loc=x.i+1
          if loc<len(sent):
            #print("text",x.text,"next: ",sent[loc].pos_," with text ",sent[loc].text,end="  ")
            #print()
            #print("")
            loc3=x.i+2
            if loc3<len(sent):
              if x.text=="which" and (sent[loc3].pos_=="NOUN" or sent[loc3].pos_=="PROPN") and sent[loc].text=="\"":
                #print("senr",sent[loc].text)
                flag=False
              if x.text=="what" and (sent[loc3].pos_=="NOUN" or sent[loc3].pos_=="PROPN") and sent[loc].text=="\"":
                #print("senr",sent[loc].text)
                flag=False
            if x.text=="which" and (sent[loc].pos_=="NOUN" or sent[loc].pos_=="PROPN"):
              #print("senr",sent[loc].text)
              flag=False
            if x.text.lower()=="what" and (sent[loc].pos_=="NOUN" or sent[loc].pos_=="PROPN"):
              #print("senr",sent[loc].text)
              flag=False
            if x.text=="when" and (sent[loc].pos_=="VERB" or sent[loc].pos_=="AUX"):
              #print("senr",sent[loc].text)
              flag=False
            if x.text=="where" and (sent[loc].pos_=="VERB" or sent[loc].pos_=="AUX"):
              #print("senr",sent[loc].text)
              flag=False
            if x.text.lower()=="whose" or x.text.lower()=="who":
              #print("pos",x.text," ",x.pos_)
              #print("senr",sent[loc].text)
              flag=False
              """pr=["He","She","he","she"]
              for y in sent:
                if y.text in pr:
                  flag=True"""
            #if x.text.lower()=="who":

            if x.text=="what" or x.text=="which":
              lox=x.i+1
              loc2=x.i
              for y in sent.noun_chunks:
                #print("in pre: ",y.text,end=" ")
                if y.start==lox:
                  #print("1st if",y.text)
                  flag=False
                if loc2>=y.start and loc2<y.end and ((y.start+1)!=y.end):
                  #print("2nd if",y.text)
                  flag=False
            wh=["where", "who", "what", "when", "why", "how", "which"]
            if x.text in wh:
              #print("x.text",x.text)
              a=question.lower().split(x.text)[0]
              #print("printing part ",a)
              b=question.lower().split(x.text)[1]
              """if is_quote_ok(a)==False or is_quote_ok(b)==False:
                flag=False"""
          if x.dep_=="relcl":
            #print("text found",x.text," with pos",x.pos_)
            temp_f=True  
          #flag=True
    return flag
  def __call__(self, qb_id: int, question: str, lexical_answer_type: str, question_determiner: str):
    #-> Iterable[str]: Cannot force return type because of error 'ABCMeta' object is not subscriptable
    self.cache_analysis(question)
    result = question
    wh_words = self.wh_words
    wh_re = re.compile("|".join(wh_words))
    question_b="%s"%question
    flag=self.precondition(qb_id,question,lexical_answer_type,question_determiner)
    #print("Question: ",question_b," flag: ",flag)
    if flag==False:
      return question_b
    if flag==True:
      #print("Question in function: ",question)
      if len(question) <1 and qb_id in self.answer_type_dict:
        answer_type = self.answer_type_dict[qb_id] # get the answer type from qb_id
          # whether starting from VERB or not:
        #answer_type = self.answer_type_dict[qb_id] # get the answer type from qb_id
        # whether starting from VERB or not
        #print("split: ",question.split()[0])
        """wn_list = wn.synsets(question.split()[0])
        #print("wh_list: ",wn_list)
        #answer_type=" thing "
        if not wn_list==[]:
          tag = wn.synsets(question.split()[0])[0].pos()
          print("here: ",wn.synsets(question.split()[0])[0])
          if tag == 'v':
            result = 'which '+answer_type+question
          else:
              result = 'which '+answer_type+' is '+question"""
      else:
        #if not qb_id in self.answer_type_dict:
        #logging.warn("Missing answer type %i" % qb_id)
        text=nlp(question.lower())

        """fil = [i for i in text.ents]
        for i in fil:
          print("text: ",i.text," label ",i.label_)"""
        #print("labels: ",fil.label_)
        n_flag=True
        for l in range (len(list(text.noun_chunks))-1):
          i=list(text.noun_chunks)[l]
          #print("checking: ",i.text,end=" ")
          if i.text.lower().find("this")!=-1:
            #print("Found")
            n_flag=False
            next=text[i.end].pos_
            next_tag=text[i.end].tag_
            next_dep=text[i.end].dep_
            next_element=text[i.end]
            #if next_tag=="prep"
            item=list(text.noun_chunks)[l+1]
            #print("replacing: ",i.text, i.start, " and ",i.end)
            #print("next item: ",text[i.end].text," and pos ",next," tag ",next_tag," dep ",next_dep)
            fla=True
            if fla==False:
              replaced=i.text.replace("this","which")
              result = re.sub(i.text, replaced, question.lower(), 1)
            elif next=="PUNCT":
              #print("PUNC found")
              if (i.end+1)<len(text):
                item_next= text[i.end].pos_
                if item_next!="NOUN":
                  replaced=i.text.replace("this ","which ")
                  result = re.sub(i.text, replaced, question, 1)
                else:
                  replaced=i.text.replace("this ","which ")
                  #print("replaced in punc ",replaced)
                  replaced=replaced+" is"
                  to_replaced=i.text+" , "
                  result = re.sub(to_replaced, replaced, question, 1)
            elif (i.start==(i.end-1)):
              if item.start==i.end:
                replaced=i.text.replace("this ","which ")
              else:
                replaced=i.text.replace("this","what")
              result = re.sub(i.text, replaced, question, 1)
            elif next=="VERB" or next=="AUX" or next=="SCONJ" or next=="CCONJ" or next_tag=="IN":
              replaced=i.text.replace("this ","which ")
              result = re.sub(i.text, replaced, question.lower(), 1)
            elif next_tag=="IN":
              t_f=False
              for k in text:
                if k.i>=next_element.i and (k.pos_=="VERB" or k.pos_=="AUX"):
                  #print("Verb found")
                  t_f=True
              if t_f==True:
                replaced=i.text.replace("this ","which ")
              else:
                replaced=i.text.replace("this ","which is ")
              result = re.sub(i.text, replaced, question.lower(), 1)
            else:
              if item.start!=(i.end+1):
                #print("next item: ",item.text)
                replaced=i.text.replace("this ","which ")
                replaced=replaced+" is"
              else:
                #print("Question here",i.text)
                replaced=i.text.replace("this ","which is ")
                #print("rep ",replaced)
              #print("rep ",replaced)
              result = re.sub(i.text, replaced, question.lower(), 1)
              #print("result",result)
        #print("n flag val",n_flag)
        if n_flag==True:
          index=question.lower().find("this ")
          if index!=-1:
            result = re.sub('this ', 'which ', question.lower(), 1)
          else:
            parse = self.current_analysis[question]["spacy"]
    
            root_verb = [x for x in parse if x.pos_ == "VERB" and not any(1 for _ in x.ancestors)]
            if len(root_verb)==0:
              root_verb = [x for x in parse if x.pos_ == "AUX" and not any(1 for _ in x.ancestors)]
            root_verb = [x for x in parse if x.pos_ == "VERB"]
            mod = [x for x in parse if x.pos_ == "NOUN" and x.i<x.head.i]
            mod2 = [x for x in parse if x.pos_ == "PRON" and x.head in root_verb and x.i<x.head.i]
            article=["A","An","The","the","a","an"]
            wh_words_list=["where", "who", "what", "when", "why", "how", "which","another"]
            wh_done=False
            #print("mod",mod)
            for x in parse:
              #print("parse: ",x.text," pos ",x.pos_," head ",x.head,x.tag_,end="  ")
              if x.pos_ == "PRON" and x.head in root_verb and x.i<x.head.i:
                #print("Condition here first pron",x.text.lower()," tag ",x.tag_)
                if x.tag_=="PRP":
                    #print("Condition here",x.text.lower())
                    if x.i==0 and x.text.lower()=="it":
                      #print("Code in pos 0")
                      #result=question.replace(" "+x.text+" "," what thing ")
                      result=re.sub(x.text.lower()+" ","what thing ", question.lower(), 1)
                    elif x.text.lower()=="it":
                      #print("Code in it")
                      #result=question.replace(" "+x.text+" "," what thing ")
                      result=re.sub(" "+x.text.lower()+" "," what thing ", question.lower(), 1)
                    else:
                      #result=question.replace(x.text,"who")
                      if x.i==0:
                        result=re.sub(x.text.lower()+" ", "who ", question.lower(), 1)
                      else:
                        result=re.sub(" "+x.text.lower()+" ", " who ", question.lower(), 1)
                      #result=re.sub(x.text.lower(), "what", question.lower(), 1)
                      #print("result: ",result)
                    wh_done=True
                    break
                elif x.tag_=="PRP$" and (x.text.lower() not in wh_words_list):
                  if x.i==0:
                    #print("Condition here",x.text.lower())
                    result=re.sub(x.text.lower()+" ", "whose ", question.lower(), 1)
                  else:
                    result=re.sub(" "+x.text.lower()+" ", " whose ", question.lower(), 1)
                  #print("result: ",result)wh_done=True
                  wh_done=True
                  break

            #print("1st",wh_done)
            wh_words_list=["where", "who", "what", "when", "why", "how", "which","another"]
            pro_list=["his","her"]
            if wh_done==False:
              #print("Code in det list")
              for x in parse:
                if x.pos_ == "DET" and (x.text.lower() not in wh_words_list)  and (x.text not in article) and (x.text.lower() in pro_list) and x.head in mod and x.i<x.head.i:
                  #print("x repl in first one: ",x.text)
                  #result=question.replace(x.text,"whose")
                  if x.i==0:
                    result=re.sub(x.text.lower()+" ", "whose ", question.lower(), 1)
                  else:
                    result=re.sub(" "+x.text.lower()+" ", " whose ", question.lower(), 1)
                  #print("result: ",result)
                  wh_done=True
                  break
            if wh_done==False:
              #print("Code in det list")
              for x in parse:
                if x.pos_ == "DET" and (x.text.lower() not in wh_words_list)  and (x.text not in article) and x.head in mod and x.i<x.head.i:
                  #print("x repl: ",x.text)
                  #result=question.replace(x.text,"whose")
                  if x.i==0:
                    result=re.sub(x.text.lower()+" ", "whose ", question.lower(), 1)
                  else:
                    result=re.sub(" "+x.text.lower()+" ", " whose ", question.lower(), 1)
                  #print("result: ",result)
                  wh_done=True
                  break
            #print("Before second one: ",wh_done)
            if wh_done==False:
              for x in parse:
                #print("Code here in sec pron",x.text," POS ",x.pos_)
                if x.pos_=="PRON":
                  if x.tag_=="PRP":
                    if x.i==0 and x.text.lower()=="it":
                      #print("Code in pos 0")
                      #result=question.replace(" "+x.text+" "," what thing ")
                      result=re.sub(x.text.lower()+" ","what thing ", question.lower(), 1)
                    elif x.text.lower()=="it":
                      #print("Code in it")
                      #result=question.replace(" "+x.text+" "," what thing ")
                      result=re.sub(" "+x.text.lower()+" "," what thing ", question.lower(), 1)
                    else:
                      #result=question.replace(x.text,"who")
                      if x.i==0:
                        result=re.sub(x.text.lower()+" ", "who ", question.lower(), 1)
                      else:
                        result=re.sub(" "+x.text.lower()+" ", " who ", question.lower(), 1)
                      #result=re.sub(x.text.lower(), "who", question.lower(), 1)
                      #result=re.sub(x.text.lower(), "what", question.lower(), 1)
                      #print("result: ",result)
                    wh_done=True
                    break
                  elif x.tag_=="PRP$" and (x.text.lower() not in wh_words_list):
                    #print("Condition here",x.text.lower())
                    if x.i==0:
                      #print("Condition here",x.text.lower())
                      result=re.sub(x.text.lower()+" ", "whose ", question.lower(), 1)
                    else:
                      result=re.sub(" "+x.text.lower()+" ", " whose ", question.lower(), 1)
                    #result=re.sub(x.text.lower(), "whose", question.lower(), 1)
                    #print("result: ",result)
                    wh_done=True
                    break
                if x.pos_ == "DET"and (x.text.lower() not in article) and (x.text.lower() in pro_list) and (x.text.lower() not in wh_words_list):
                  #print("x repl: ",x.text)
                  #result=question.replace(x.text,"whose")
                  if x.i==0:
                    result=re.sub(x.text.lower()+" ", "whose ", question.lower(), 1)
                  else:
                    result=re.sub(" "+x.text.lower()+" ", " whose ", question.lower(), 1)
                  #result=re.sub(x.text.lower(), "whose", question.lower(), 1)
                  wh_done==True
                  break
            if wh_done==False:
              #print("Code in det list")
              for x in parse:
                if x.pos_ == "DET"and (x.text.lower() not in article) and (x.text.lower() not in wh_words_list):
                  #print("x repl: ",x.text)
                  #result=question.replace(x.text,"whose")
                  if x.i==0:
                    result=re.sub(x.text.lower()+" ", "whose ", question.lower(), 1)
                  else:
                    result=re.sub(" "+x.text.lower()+" ", " whose ", question.lower(), 1)
                  #result=re.sub(x.text.lower(), "whose", question.lower(), 1)
                  wh_done==True
                  break
            #print(wh_done)
            """if wh_done==False:
              #print("Code in noun")
              for x in parse:
                if x.pos_ == "NOUN" and x.head in root_verb and x.i<x.head.i:
                  #result=question.replace(x.text,"what")
                  print("noun: ",x.text)
                  result=re.sub(x.text.lower(), "what", question.lower(), 1)
                  wh_done=True
                  break"""
            if wh_done==False:
              #print("Code in noun")
              ap_list=["here,","Here,","here","Here"]
              for x in parse:
                if x.text in ap_list:
                  #result=question.replace(x.text,"what")
                  #print("here test: ",x.text)
                  if x.i==0:
                    result=re.sub(x.text.lower()+" ", "where ", question.lower(), 1)
                  else:
                    result=re.sub(" "+x.text.lower()+" ", " where ", question.lower(), 1)
                  wh_done=True
                  break
                
            #final_wh=False
            if wh_done==False:
              #print("Code in noun")
              if not wh_re.search(result.lower()):
                result = "Where "+result
                    #wh_done=True
                #result = re.sub("its", "which", question.lower(), 1)
                    
                """loc=x.i
                    for y in parse.noun_chunks:
                      if x.i>=y.start and x.i<y.end:
                        result=question.replace(y.text,"what")"""
              #print("list: ",mod," and ",mod2)


        #print("Modified question: ",result)
        yield self.postcondition(result,question_b)

        #result = re.sub('that', 'which', question, 1)

  def postcondition(self,question,question_b):
      #len1=GingerIt().parse(question)['result']
      question_list=question.lower().split(" ")
      question_b_list=question_b.lower().split(" ")
      for i in range (len(question_b_list)):
        #print(question_list[i]," and ",question_b_list[i])
        if question_list[i]!=question_b_list[i]:
          #print("Not eq")
          a=""
          b=""
          for j in range (len(question_list)):
            if j<i:
              a=a+"".join(question_list[j])
            else:
              b=b+"".join(question_list[j])
          #a=question.split(question_list[i])[0]
          #b=question.split(question_list[i])[1]
          #print(a," and ",b)
          """if is_quote_ok(a)==False or is_quote_ok(b)==False:
            return question_b"""
      doc=nlp(question)
      c=0
      result='%s' % question
      #print("Len in doc",len(doc))
      for i in doc:
        c=c+1
        if c<(len(doc)):
          #print("c: ",c," ps ",list(doc)[c].pos_," og ",list(doc)[c].text)
          if i.text=="another" and list(doc)[c].pos_=="NOUN":
            result=question.replace(i.text,"a")
      rem=['also','later','in another',"another of","another"]
      for i in rem:
        if result.lower().find(i)!=-1:
          result=result.replace(i,"")
      len1=result
      return len1
      #len=syntax_checker(question)
      #print("length of syntax check: ",len)
      #return len
      # Heuristic12
class replace_this_is(ConditionalHeuristic):
  def my_name(cls_): 
      return cls_.__name__ 
  def precondition(self, qb_id: int, question: str, lexical_answer_type: str, question_determiner: str):
    flag=False
    x = question
    index = x.find('this is')
    if index!=-1:
      flag=True
    return flag
  def __call__(self, qb_id: int, question: str, lexical_answer_type: str, question_determiner: str):
    #-> Iterable[str]: Cannot force return type because of error 'ABCMeta' object is not subscriptable
    """
    Replace 'this' to 'which'+answer_type within 'this is' pattern.
    """
    flag=self.precondition(qb_id,question,lexical_answer_type,question_determiner)
    x = question
    if flag==True:
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
      yield self.postcondition(question)
    else:
      yield question
  def postcondition(self,question):
      len=syntax_checker(question)
      #print("length of syntax check: ",len)
      return len
  # Heuristic14: double auxiliary words
class remove_extra_AUX(ConditionalHeuristic):
  def __call__(self, qb_id: int, question: str, lexical_answer_type: str, question_determiner: str):
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
class replace_which_with_this(ConditionalHeuristic):
  def my_name(cls_): 
      return cls_.__name__ 
  def precondition(self, qb_id: int, question: str, lexical_answer_type: str, question_determiner: str):
    flag=False
    x = question
    text=nlp(x)
    for token in text:
      if token.text=="which":
        flag=True
    return flag
  def __call__(self, qb_id: int, question: str, lexical_answer_type: str, question_determiner: str):
    #-> Iterable[str]: Cannot force return type because of error 'ABCMeta' object is not subscriptable
    """
    Convert 'which' to 'that' and check if no 'which' present anymore, if so, convert 'this' to 'which'.
    """
    flag=self.precondition(qb_id,question,lexical_answer_type,question_determiner)
    if flag==True:
      x = question
      text=nlp(x)
      """for i in text:
        print("i: ",i.text," tag: ",i.tag_," dep ",i.dep_," and pos: ",i.pos_)"""
      #for i in text.noun_chunks:
        #print("i: ",i.text)
      for token in text:
        if token.text=="which":
          which_rep=True
          #print("loc: ",token.i)
          for j in text.noun_chunks:
            #print(j.text)
            if token.i>=j.start and token.i<j.end:
              which_rep=False
          if token.i==0:
            which_rep=False
          if which_rep==True:
            question = re.sub('which', 'that', question)
            #print("phrase: ",j.text," start: ",j.start," and end of span: ",j.end)
            #if token.i>=j.start
        # check if no 'which' present anymore
      """index = result.find('which')
      if index==-1:
        result = re.sub('this', 'which', result, 1)
        question = result"""
      
      yield self.postcondition(question)
    else:
      yield question
  def postcondition(self,question):
    len=syntax_checker(question)
    #print("length of syntax check: ",len)
    return len

class rejoin_contractions(ConditionalHeuristic):
  def my_name(cls_): 
      return cls_.__name__
  def __call__(self, qb_id: int, question: str, lexical_answer_type: str, question_determiner: str):
    # -> Iterable[str]: Cannot force return type because of error 'ABCMeta' object is not subscriptable
    for separated, together in {"who 's": "whose", "ca n't": "can't", "wo n't": "won't"}.items():
      if separated in question:
        question = question.replace(separated, together)
    yield self.postcondition(question)
  def postcondition(self,question):
    len=syntax_checker(question)
    #print("length of syntax check: ",len)
    return len

class split_conjunctions(ConditionalHeuristic):
  def my_name(cls_): 
    return cls_.__name__ 
  def cache_analysis(self, question, verbose=False):
    """
    Parse the sentence for downstream heuristics.  This prevents repeated computation in each of the heuristics.
    """
    print("code in cachex")
    tokens = nltk.word_tokenize(question.lower())
    text = nltk.Text(tokens)
    tagged = nltk.pos_tag(text)

    spacy_parse = nlp(question)
    if verbose:
      for ii in spacy_parse.sents:
        logging.debug(to_nltk_tree(ii.root))
          
    self.current_analysis[question] = {"spacy": spacy_parse, "nltk_tokens": tokens, "nltk_tags": tagged}
    print(self.current_analysis[question]["spacy"])
  def precondition(self, qb_id: int, question: str, lexical_answer_type: str, question_determiner: str):
    #-> Iterable[str]: Cannot force return type because of error 'ABCMeta' object is not subscriptable
    # First, find the verbs 
    flag=False
    parse = self.current_analysis[question]["spacy"]
    for x in parse:
      #print(x," and pos: ",x.pos_)
      if x.pos_=="CCONJ" or x.pos_=="SCONJ":
        flag=True
    return flag
  def __call__(self, qb_id: int, question: str, lexical_answer_type: str, question_determiner: str):
    self.cache_analysis(question)
    backup=question
    # If so, then we need to know if they are independent clauses
    #print("Heuristic 1: Split Conjunctions")
    #print("prev vals: ",self.current_analysis)
    sem_part=""
    prev="%s" %question
    #print(self.current_analysis[question])
    if question.find(";")!=-1:
      sem_part=" ; "+question.split(";")[1]
      self.current_analysis[prev]["spacy"]=nlp(question.split(";")[0])
      question=question.split(";")[0]
      #print(self.current_analysis[prev]["spacy"])
    #print("prev vals: ",self.current_analysis[prev])
    parse = self.current_analysis[prev]["spacy"]
    flag=self.precondition(qb_id,prev,lexical_answer_type,question_determiner)
    if flag==True:
        #tagged = self.current_analysis[question]["nltk_tags"]
        root=[x for x in parse if (not any(1 for _ in x.ancestors))]

        #print("Finding root: ",root)
        #parse2=parse.remove(root)
        root_verb = [x for x in parse if x.pos_ == "VERB" and (not any(1 for _ in x.ancestors))]
        for x in parse:
          #print("root",x.text," and pos: ",x.pos_)
          if x.pos_ == "VERB" and (not any(1 for _ in x.ancestors)):
            root_verb=[x]
            break
          elif x.pos_ == "AUX" and (not any(1 for _ in x.ancestors)):
            #print("code entered in aux")
            root_verb=[x]
            break
            #verbs = [x for x in parse if x.pos_ == "VERB" and x.head in root_verb]
          elif x.pos_ == "VERB" and (any(1 for _ in x.ancestors)):
            flag_root=True
            for y in x.ancestors:
              if y.pos_=="VERB":
                flag_root=False
            if flag_root==True:
              root_verb=[x]
        #print("root list final",root_verb)
        flag_verb=False
        for x in parse:
          if x.dep_=="ROOT" and x.pos_=="VERB":
            flag_verb=True
          verbs = [x for x in parse if x.pos_ == "VERB" and x.head in root_verb]
          #print("root:",root_verb[0])
          if len(root_verb)>0:
            if root_verb[0] not in verbs and len(root_verb)>0:
              #print("code here",root_verb[0])
              verbs.extend(root_verb)
        else:
            #print("value if",flag_verb," abd ",root_verb)
            verbs=[]
            if len(root_verb)>0:
              verbs.append(root_verb[0])
            verbs.extend([x for x in parse if x.pos_ == "VERB" and x.head in root_verb])
        #print("verbs",verbs)
        verb_conj = set()
        for verb in verbs:
          #print("verb: ",verb)
          for child in verb.children:
            #print("child before if: ",child," and pos ",child.pos_," and dep ",child.dep_)
            if (child.dep_ == 'cc' or child.dep_=='mark') and (child.pos_ == "CCONJ" or child.pos_=="SCONJ"):
              #print("verb: ",verb," and child: ",child)
              verb_conj.add((verb, child))

        #print("verb conj: ",verb_conj)
        if len(verb_conj) > 1:
          logging.warn("Multiple conjunctions in sentence and we don't know what to do: " + question)
        
        #return 
        #parse = self.current_analysis[question]["spacy"]
        ##print("#printing: ",parse)
        #root_verb = [x for x in parse if x.pos_ == "VERB" and not any(1 for _ in x.ancestors)]
        ##print("root verb: ",root_verb)
        #verbs = [x for x in parse if x.pos_ == "VERB" and x.head in root_verb]
        ##print("verbs: ",verbs)

        # See if they have any coordinating conjunctions as dependents
        
          
        for verb, conj in verb_conj:
          #print("verb dep in for loop: ",verbs[-1].dep_," text: ",verbs[-1].text)
          # Check to see if this is the second verb and if it has no ancestors
          if verb.i > verbs[0].i and not any(1 for _ in verb.ancestors):
            # If so, we have two independent clauses, so yield the two
            # parts on either side of the conjunction
            #print("Code in the fisr if")
            first_q=" ".join(x.text for x in parse if x.i < conj.i)
            #print("Before postcondition: ")
            #print(first_q)
            final_first_q=self.postcondition(first_q,backup)
            #print("After postcondition: ")
            yield final_first_q
            #yield " ".join(x.text for x in parse if x.i < conj.i)
            second_q=" ".join(x.text for x in parse if x.i > conj.i)
            #print("Before postcondition: ")
            #print(second_q)
            final_second_q=self.postcondition(second_q,backup)
            #print("After postcondition")
            yield final_second_q
            #yield " ".join(x.text for x in parse if x.i > conj.i)
          elif verb.i < verbs[-1].i and verbs[-1].dep_ == "conj":
            # Otherwise, if this verb is child of another verb with "conj"
            # relation, we can have two sentences with the same subject

            # Get what came before verb and doesn't modify verb
            """print("Code in the second if")
            for x in parse:
              if x.i<verb.i:
                print("text: ",x.text," head: ",x.head, " pos: ",x.pos_)"""
            left_tokens = [x for x in parse if x.i < verb.i and not
                            (x.head == verb and (x.pos_ == "ADV" or x.pos_ == "AUX"))]

            # Get possible completions

            first_verb = [x for x in parse if x.i < conj.i and not x in left_tokens]
            second_verb = [x for x in parse if x.i > conj.i]
            #print("Second verb ")
                  
            """for b in temp_nlp:
              if b.i==0:
                if b.pos_=="PRON" or b.pos_="NOUN":
                  second="""
            #print("first: ",first_verb," second ",second_verb," left tok: ",left_tokens)
            # Return those
            first_q=" ".join(x.text for x in left_tokens + first_verb)
            #print("Before postcondition: ")
            #print(first_q)
            final_first_q=self.postcondition(first_q,backup)
            #print("After postcondition")
            yield final_first_q
            #yield " ".join(x.text for x in left_tokens + first_verb)
            nsub_flag=False
            #print("verbs here:",verbs)
            for x in second_verb:
              #print("text",x.text," pos ",x.pos_," dep ",x.dep_," tag_ ",x.tag_)
              #print("testing: ",x.text," with val: ",nsub_flag)
              if x in verbs:
                break
              if x.dep_=="nsubj" or x.dep_=="nsubjpass":
                #print("code in nsubtrue")
                nsub_flag=True
            if nsub_flag==True:
              second_q=" ".join(x.text for x in second_verb) 
            else:
              #print("code in not nsubtrue")
              second_q=" ".join(x.text for x in left_tokens+second_verb)
              verb_im=["give","identity","name"]
              flag_pos=False
              p=list(self.imperative_pattern)[0]
              #print("Original ",backup)
              for pattern in self.imperative_pattern:
                #print("pattern",pattern)
                if pattern.search(backup):
                  #print("Found in question",backup)
                  p=pattern
                  flag_pos=True
              
              #print(p)
              if flag_pos==True:
                s=""
                loc=9999
                ver_c=0
                #print("Searching ",p.search(second_q))
                if p.search(second_q) is None:
                  #print("Found in postcondition",second_q)
                  k="%s"%s
                  for i in nlp(backup):
                    if i.text in verb_im:
                      ver_c=ver_c+1
                      loc=i.i
                    elif i.pos_=="VERB" and i.i>loc:
                      ver_c=ver_c+1
                    if ver_c<2:
                      s=s+" "+i.text
                  #result=s+" "+result
                  second_q=s+" ".join(x.text for x in second_verb) 

              #second_q=" ".join(x.text for x in left_tokens+second_verb) 
            #print("Before precondition")
            #print(second_q+sem_part)
            yield self.postcondition(second_q+sem_part,backup)
            #print("types: ",type(second_verb))
            if nsub_flag==True:
              for x in second_verb:
                #print(x.text," pos here ",x.pos_)
                if x in verbs:
                  break
                if (x.dep_=="nsubj" or x.dep_=="nsubjpass") and x.pos_=="PRON":
                  #second_verb.remove(x)
                  loc=second_verb.index(x)+1
                  #print("After removal, ",second_verb)
                  second_q=" ".join(x.text for x in left_tokens+second_verb[loc:]) 
                  #print("Before postcondition in 3rd part: ")
                  #print(second_q+sem_part)
                  final_second_q=self.postcondition(second_q+sem_part,backup)
                  #print("After postcondition in 3rd part")
                  yield final_second_q
                  break
            #yield " ".join(x.text for x in left_tokens + second_verb) 
        # If there are nouns and verbs on both sides of it, the just iterate on those
        # If there are only verbs, duplicate the subject
        None
      # Heuristic16: 
      # WDT tag: which/what
      # WRB tag: where/why/when
    else:
      yield question
  def postcondition(self,question1,question_prev):
    check2=is_quote_ok(question1)
    #check2=True
    if check2==False:
      return syntax_checker(question_prev)
    doc=nlp(question1)
    c=0
    result='%s' % question1
    #print("Len in doc",len(doc))
    for i in doc:
      c=c+1
      if c<(len(doc)):
        #print("c: ",c," ps ",list(doc)[c].pos_," og ",list(doc)[c].text)
        if i.text=="another" and list(doc)[c].pos_=="NOUN":
          result=question1.replace(i.text,"a")
    rem=['also','later','in another',"another"]
    for i in rem:
      if result.lower().find(i)!=-1:
        result=result.replace(i,"")

    #print("New q: ",result)
    """c=0
    for i in doc.noun_chunks:
      k=i.end
      if k<len(doc):
        if doc[k].pos_=="PRON":
          question=question.replace(doc[k].text,"",1)
          break
        if (doc[k].text=="\"" and k+1<len(doc) and doc[k+1].pos_=="PRON"):
          question=question.replace(doc[k+1].text,"",1)
          break
    for i in doc.noun_chunks:
      l=i.start
      if l>0:
        if doc[l-1].pos_=="PRON":
          question=question.replace(doc[k].text,"",1)
    c=0
    for i in doc:
      c=c+1
      if c<(len(doc)):
        if i.pos_=="PRON" and list(doc)[c].pos_=="PRON":
          question=question.replace(i.text,"",1)
          break"""
    """verb_im=["give","identity","name"]
    flag_pos=False
    p=list(self.imperative_pattern)[0]
    print("Original ",question_prev)
    for pattern in self.imperative_pattern:
      print("pattern",pattern)
      if pattern.search(question_prev):
        print("Found in question",question_prev)
        p=pattern
        flag_pos=True
    
    print(p)
    if flag_pos==True:
      s=""
      loc=9999
      ver_c=0
      print("Searching ",p.search(result))
      if p.search(result) is None:
        print("Found in postcondition",result)
        k="%s"%s
        for i in nlp(question_prev):
          if i.text in verb_im:
            ver_c=ver_c+1
            loc=i.i
          elif i.pos_=="VERB" and i.i>loc:
            ver_c=ver_c+1
          if ver_c<2:
            s=s+" "+i.text
        result=s+" "+result"""

            
    #len1=GingerIt().parse(result)['result']
    #len1=syntax_checker(question)
    #print("length of syntax check: ",len)
    return result
    
class add_question_word(ConditionalHeuristic):
  def __call__(self, qb_id: int, question: str, lexical_answer_type: str, question_determiner: str):
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
class add_subject(ConditionalHeuristic):
  def __call__(self, qb_id: int, question: str, lexical_answer_type: str, question_determiner: str):
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
class which_none_is(ConditionalHeuristic):
  def my_name(cls_): 
    return cls_.__name__ 
  def precondition(self, qb_id: int, question: str, lexical_answer_type: str, question_determiner: str):
    flag=False
    x = question
    index = x.find('which none is')
    if index != -1:
      flag=True
    return flag
  def __call__(self, qb_id: int, question: str, lexical_answer_type: str, question_determiner: str):
    #-> Iterable[str]: Cannot force return type because of error 'ABCMeta' object is not subscriptable
    """
    Convert 'which none is' to 'what is'.
    """
    x = question
    flag=self.precondition(qb_id,question,lexical_answer_type,question_determiner)
    if flag==True:
      qb_id = int(qb_id)
      #print("id: ",qb_id)
      #print("keys: ",self.answer_type_dict.keys())
      if qb_id in self.answer_type_dict.keys():
        answer_type = self.answer_type_dict[qb_id] # get the answer type from qb_id
        result = re.sub('which none is', 'which '+answer_type+' is', x)
        question = result
      else:
        logging.warn(qb_id+'is not in the frequency table!')
      yield self.postcondition(question)
    else:
      yield question
  def postcondition(self,question):
    len=syntax_checker(question)
    #print("length of syntax check: ",len)
    return len

  # Heuristic19: 'what is which' pattern
class what_is_which(ConditionalHeuristic):
  def my_name(cls_): 
    return cls_.__name__ 
  def precondition(self, qb_id: int, question: str, lexical_answer_type: str, question_determiner: str):
    flag=False
    x = question
    index = x.find('what is which')
    if index != -1:
      flag=True
    return flag
  def __call__ (self, qb_id: int, question: str, lexical_answer_type: str, question_determiner: str):
    #-> Iterable[str]: Cannot force return type because of error 'ABCMeta' object is not subscriptable
    """
    Remove "what is" from "what is which".
    """
    x = question
    text=nlp(question)
    flag=self.precondition(qb_id,question,lexical_answer_type,question_determiner)
    if flag==True:
      for i in range (len(list(text.noun_chunks))-1):
        first=(list(text.noun_chunks))[i].text
        second=(list(text.noun_chunks))[i+1].text
        if first=="what" and second.find("which")!=-1:
          result = re.sub(first+" is ", '', x)
          result = re.sub(second,second+" is ", result)
          break
        #print("nouns: ",i.text)
      question = result
      yield self.postcondition(question)
    else:
      yield question

  def postcondition(self,question):
    len=syntax_checker(question)
    #print("length of syntax check: ",len)
    return len
