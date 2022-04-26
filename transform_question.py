#Step1. import libraries

import numpy as np
import pandas as pd
import json
import string
import nltk
import time
import os
import re
import random
import argparse
import spacy
import neuralcoref
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
import tensorflow as tf
from transformers import DistilBertTokenizerFast
from transformers import TFDistilBertForSequenceClassification
from functools import partial
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

from collections import Counter
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('omw-1.4')

nlp = spacy.load('en_core_web_sm')
neuralcoref.add_to_pipe(nlp)

# Heuristics for NQlike quality checking
class HeuristicsTransformer:
  def __init__(self,config):
    with open(config["last_sent_word_transform_30000"], 'r') as f:
      self.last_sent_word_transform_30000 = json.load(f)
    with open(config["answer_type_dict_before_parse_tree_nq_like_test_v_3"]) as json_file:
      self.answer_type_dict = json.load(json_file)
    self.valid_verbs = config["valid_verbs"]
    self.wh_words = config["wh_words"]
    self.strictly_valid_verbs = config["strictly_valid_verbs"]
    self.to_trim = config["to_trim"]
    self.pos_pronouns = config["pos_pronouns"]
    self.pronouns = config["pronouns"]
    self.bad_patterns = config["bad_patterns"]
    self.non_last_sent_transform_dict = config["non_last_sent_transform_dict"]
    self.remove_dict = config["remove_dict"]

  # Heuristic 1 remove punctuation patterns at the beginning and the end of the question [" ' ( ) , .]
  def clean_marker(self,q):
    """
    Remove punctuation patterns at the beginning and the end of the question
    """
    to_clean = r"\"|\'|\(|\)|,|\.|\s"
    has_heuristic = False
    q_array = nltk.word_tokenize(q.lower())
    array_leng = len(q_array)
    while re.match(to_clean, q_array[array_leng-1]):
      q_array = q_array[:array_leng-1]
      array_leng = array_leng - 1
      has_heuristic = True

    while re.match(to_clean, q_array[0]):
      q_array = q_array[1:]
      array_leng = array_leng - 1
      has_heuristic = True
    if has_heuristic:
      q = ' '.join(q_array)
    return q

  # Heuristic 2 -- name this answer type correction
  def clean_answer_type(self,q):
    """
    Convert "-- name this" patterns to "which"
    """
    to_clean = "-- name this"
    if re.search(to_clean, q):
      start_with = "^-- name this"
      # if start with -- name this converts to which
      if re.search(start_with, q):
        q = re.sub(start_with, 'which', q)
      else:
        q = re.sub(to_clean, 'the', q)
    return q

  # Heuristic 3 semicolon
  def drop_after_semicolon(self,q):
    """
    Remove contents after semicolon in NQlike
    """
    to_clean = ";.*"
    if re.search(to_clean, q):
      q = re.sub(to_clean, '', q)
    return q

  # Heuristic 4 remove pattern issues
  def remove_pattern(self,q):
    """
    Remove bad patterns in NQlike
    """
    to_clean = self.bad_patterns
    q = re.sub(to_clean, '', q)
    return q

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
    tokens = nltk.word_tokenize(text.lower())
    text = nltk.Text(tokens)
    tagged = nltk.pos_tag(text)
    counted = Counter(tag for word,tag in tagged)
    num_of_verb = 0
    for v in verb_tags:
      num_of_verb = num_of_verb + counted[v]
    return num_of_verb

  def remove_rep_subject(self,q):
    """
    remove is this... pattern
    """
    to_clean = " is this [a-zA-Z]*\s"
    if re.search(to_clean, q):
      # the sentence has to have 1 verb at least otherwise this will not be done
      if (self.count_num_of_verbs(q) > 1):
        q = re.sub(to_clean, ' ', q)
    return q

  # Heuristic 6 change be determiner to s possession
  def remove_BE_determiner(self,q):
    """
    change is his/is her/is its to 's
    """
    to_clean = "( is his )|( is her )|( is its )"
    if re.search(to_clean, q):
      q = re.sub(to_clean, '\'s ', q)
    return q

  # function to add space before punctuation
  def add_space_before_punctuation(self,q):
    """
    add space before punctuation because in NQ there's space before all types of punctuation
    """
    tokens = nltk.word_tokenize(q.lower())
    q = ' '.join(tokens)
    return q

  # Heuristic 7 add be verb to questions without verb
  def add_verb(self,text):
    """
    add BE verb when there's no verb in the entire question
    """
    tokens = nltk.word_tokenize(text.lower())
    text = nltk.Text(tokens)
    tagged = nltk.pos_tag(text)
    ind = 0
    for tk,tg in tagged:
      if tg == 'NN' or tg == 'NNP':
        tokens.insert(ind+1,'is')
        break
      elif tg == 'NNS' or tg == 'NNPS':
        tokens.insert(ind+1,'are')
        break
      ind = ind + 1
    return ' '.join(tokens)

  def fix_no_verb(self,q):
    if (self.count_num_of_verbs(q, True) == 0):
      q = self.add_verb(q)
    return q

  # Heuristic 8 remove repetitive be verb when there's more verbs
  def remove_repeat_verb(self,q):
    """
    remove is he/is she/is it
    """
    to_clean = "( is he )|( is she )|( is it )"
    if re.search(to_clean, q):
      if (self.count_num_of_verbs(q) > 1):
        q = re.sub(to_clean, ' ', q)
    return q

  # Heuristic 9 First verb after which in continuous tense
  def convert_continuous_to_present(self,q):
    """
    if the first verb is in continuous tense, change it to nomal
    """
    verb_tags = self.valid_verbs
    text = q
    tokens = nltk.word_tokenize(text.lower())
    text = nltk.Text(tokens)
    tagged = nltk.pos_tag(text)
    ind = 0
    for tk,tg in tagged:
      if tg in verb_tags:
        if tg == 'VBG':
          try:
            old_tk, old_tg = tagged[ind-1]
            if old_tg == 'NN' or old_tg == 'NNP':
              tokens[ind] = re.sub('ing','s',tokens[ind])
              q = ' '.join(tokens)
            else:
              tokens[ind] = re.sub('ing','',tokens[ind])
              q = ' '.join(tokens)
          except:
            break
          break
        else:
            break
      ind = ind + 1
    return q

  # Heuristic 10 fix "name which" "identify which"
  def remove_name_which(self,q):
    """
    remove name which/identify which
    """
    to_clean = "identify which|name which"
    if re.search(to_clean, q):
      q = re.sub(to_clean, 'which', q)
    return q

  # function counts the number of of questions with 1,2,3 words
  def count_word_freq(self,q_lst):
    count_1 = 0
    count_2 = 0
    count_3 = 0
    for q in q_lst:
      q_array = q.split()
      if len(q_array) == 1:
        count_1 = count_1 + 1
      if len(q_array) == 2:
        count_2 = count_2 + 1
      if len(q_array) == 3:
        count_3 = count_3 + 1
    return (count_1,count_2,count_3)

  # Heuristic11 convert this to which
  def no_wh_words(self,qb_id, q):
    result = q
    wh_words = self.wh_words
    wh_re = re.compile("|".join(wh_words))
    if not wh_re.search(q):
      # no wh_words
      qb_id = str(qb_id)
      if qb_id in self.answer_type_dict.keys():
          answer_type = self.answer_type_dict[qb_id] # get the answer type from qb_id
          # whether starting from VERB or not
          wn_list = wn.synsets(q.split()[0])
          if not wn_list==[]:
              tag = wn.synsets(q.split()[0])[0].pos()
              if tag == 'v':
                  result = 'which '+answer_type+q
              else:
                  result = 'which '+answer_type+' is '+q
      else:
          result = re.sub('this', 'which', q, 1)
    return result

  # Heuristic12
  def this_is_pattern(self,qb_id, q):
    """
    Replace 'this' to 'which'+answer_type within 'this is' pattern.
    """
    x = q
    index = x.find('this is')
    if index!=-1:
      # adding answer type
      qb_id = str(qb_id)
      if qb_id in self.answer_type_dict.keys():
        answer_type = self.answer_type_dict[qb_id] # get the answer type from qb_id
        replacement = 'which '+answer_type
        result = re.sub('this is', replacement+' is', x, 1)
        q = result
      else:
        # answer type is not in the dict
        result = re.sub('this', 'which', x, 1)
        q = result
    return q

  # Heuristic13: 'is/are' at the end of questions (after cleaning the wrong punc at the end of the sample)
  def remove_end_be_verbs(self,q):
    """
    Remove 'is/are' at the end of NQklike questions.
    """
    x = q
    x = x.strip()
    if x[-3:] == ' is':
      result = x[:-3]
      q = result
    elif x[-4:] == ' are':
      result = x[:-4]
      q = result
    return q

  # Heuristic14: double auxiliary words
  def remove_extra_AUX(self,q):
    """
    Remove extra auxiliary words.
    """
    x = q
    doc_dep = nlp(x)
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
        q = result
      else:
        # two conjunction BE verbs or two non-conjunctional be verbs without pronoun in between
        del tokem_text_lst[index]
        result = " ".join(tokem_text_lst)
        q = result
    return q

  # Heuristic15: WDT+BE patterns
  def WDT_BE_pattern(self,q):
    """
    Convert 'which' to 'that' and check if no 'which' present anymore, if so, convert 'this' to 'which'.
    """
    x = q
    index1 = x.find('which is where')
    index2 = x.find('which is why')
    if index1 != -1:
      result = re.sub('which is where', 'that is where', x)
      q = result
    elif index2 != -1:
      result = re.sub('which is why', 'that is why', x)
      q = result
    else:
      result = x
      # check if no 'which' present anymore
    index = result.find('which')
    if index==-1:
      result = re.sub('this', 'which', result, 1)
      q = result
    return q

  # Heuristic16: 
  # WDT tag: which/what
  # WRB tag: where/why/when
  def no_WDT_and_WRB(self,qb_id, q):
    """
    Adding 'which+answer_type' at the beginning when no WDT/WRB present.
    """
    x = q
    doc_dep = nlp(x)
    tag_lst = []
    tokem_text_lst = []
    for k in range(len(doc_dep)):
      tag_lst.append(doc_dep[k].tag_)
      tokem_text_lst.append(doc_dep[k].text)
    if ('WRB' in tag_lst)!=True and ('WDT' in tag_lst)!=True:
      # adding answer type at the beginning
      qb_id = str(qb_id)
      if qb_id in self.answer_type_dict.keys():
        answer_type = self.answer_type_dict[qb_id] # get the answer type from qb_id
        result = 'which '+answer_type+' is '+x
        q = result
      else:
        print(qb_id+'is not in the frequency table!')
    return q

  # Heuristic17: VERB/AUX at the beginning of the sample while missing the object
  def VERB_AUX_at_beginning(self,qb_id, q):
    """
    Adding 'which+answer_type' at the beginning when starting with VERB/AUX and missing the object.
    """
    x = q
    doc_dep = nlp(x)
    pos_lst = []
    tokem_text_lst = []
    for k in range(len(doc_dep)):
      pos_lst.append(doc_dep[k].pos_)
      tokem_text_lst.append(doc_dep[k].text)
    if pos_lst[0]=='AUX' or pos_lst[0]=='VERB':
      # adding answer type at the beginning
      qb_id = str(qb_id)
      if qb_id in self.answer_type_dict.keys():
        answer_type = self.answer_type_dict[qb_id] # get the answer type from qb_id
        result = 'which '+answer_type+' '+x
        q = result
      else:
        print(qb_id+'is not in the frequency table!')
    return q

  # Heuristic18: 'which none is' patterns
  def which_none_is(self,qb_id, q):
    """
    Convert 'which none is' to 'what is'.
    """
    x = q
    index = x.find('which none is')
    if index != -1:
      qb_id = str(qb_id)
      if qb_id in self.answer_type_dict.keys():
        answer_type = self.answer_type_dict[qb_id] # get the answer type from qb_id
        result = re.sub('which none is', 'which '+answer_type+' is', x)
        q = result
      else:
        print(qb_id+'is not in the frequency table!')
    return q

  # Heuristic19: 'what is which' pattern
  def what_is_which(self,q):
    """
    Remove "what is" from "what is which".
    """
    x = q
    index = x.find('what is which')
    if index != -1:
      result = re.sub('what is which', 'which', x)
      q = result
    return q

  # quality checking for each NQlike question
  def quality_check(self,qb_id, q):
    try:
      q = self.remove_name_which(q)
      q = self.clean_marker(q)
      q = self.clean_answer_type(q)
      q = self.drop_after_semicolon(q)
      q = self.convert_continuous_to_present(q)
      q = self.no_wh_words(qb_id, q)
      q = self.this_is_pattern(qb_id, q)
      q = self.WDT_BE_pattern(q)
      q = self.no_WDT_and_WRB(qb_id, q)
      q = self.VERB_AUX_at_beginning(qb_id, q)
      q = self.which_none_is(qb_id, q)
      q = self.what_is_which(q)
      q = self.remove_end_be_verbs(q)
      q = self.remove_extra_AUX(q)
      q = self.remove_pattern(q)
      q = self.remove_rep_subject(q)
      q = self.remove_BE_determiner(q)
      q = self.remove_repeat_verb(q)
      q = self.fix_no_verb(q)
      q = self.add_space_before_punctuation(q)
    except:
      pass
    return q

class AnswerTypeClassifier:

    def __init__(self, last_sent_word_transform_30000, tokenizer, loaded_model):
        self.last_sent_word_transform_30000 = last_sent_word_transform_30000
        self.tokenizer = tokenizer
        self.loaded_model = loaded_model

    def get_answer_type_group(self, test_sentence):
        predict_input = self.tokenizer.encode(test_sentence,
                                      truncation=True,
                                      padding=True,
                                      return_tensors="tf")
        tf_output = self.loaded_model.predict(predict_input)[0]
        tf_prediction = tf.nn.softmax(tf_output, axis=1)
        labels = ['NON_PERSON','PERSON']
        label = tf.argmax(tf_prediction, axis=1)
        label = label.numpy()
        return labels[label[0]]

    def answer_type_classifier_training(self):
        # No need to rerun the answer type classifier to replicate results as we are providing checkpoints for the same
        # the checkpoints are already provided in the corresponding folder
    
        #A PERSON:
        #     replace 'he/she/who/him' and 'He/She/Who/Him' with 'which + answer_type + is/are'
        #     replace 'his/whose/she's/he's' and 'His/Whose/She's/He's' with 'which + answer_type's'
    
        #A THING:
        #     replace 'it/this/these' and 'It/This/These' with 'which + answer_type + is/are'
        #     replace 'it's' and 'It's' with 'which + answer_type's'
        person_list = []
        label_list = []
        for v in self.last_sent_word_transform_30000['who is the']:
          person_list.append(v)
          label_list.append('PERSON')
    
        non_person_list = []
        for v in self.last_sent_word_transform_30000['which is the']:
          non_person_list.append(v)
          label_list.append('NON-PERSON')
    
        for v in self.last_sent_word_transform_30000['what is the']:
          non_person_list.append(v)
          label_list.append('NON-PERSON')
    
        my_answer_type_list = person_list+non_person_list
        label_list = label_list[:len(my_answer_type_list)]
        # convert lists to dataframe
        zippedList =  list(zip(label_list, my_answer_type_list))
        classification_df = pd.DataFrame(zippedList, columns=['label','answer_type'])
    
        LE = LabelEncoder()
        classification_df['label'] = LE.fit_transform(classification_df['label'])
        classification_df.head()
    
        groups = classification_df['answer_type'].values.tolist()
        labels = classification_df['label'].tolist()
    
        training_sentences, validation_sentences, training_labels, validation_labels = train_test_split(groups, labels, test_size=.2)
        tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        tokenizer([training_sentences[0]], truncation=True,
                                    padding=True, max_length=128)
        train_encodings = tokenizer(training_sentences,
                                    truncation=True,
                                    padding=True)
        val_encodings = tokenizer(validation_sentences,
                                    truncation=True,
                                    padding=True)
        train_dataset = tf.data.Dataset.from_tensor_slices((
            dict(train_encodings),
            training_labels
        ))
    
        val_dataset = tf.data.Dataset.from_tensor_slices((
            dict(val_encodings),
            validation_labels
        ))
    
        model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased',num_labels=2)
        optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5, epsilon=1e-08)
        model.compile(optimizer=optimizer, loss=model.compute_loss, metrics=['accuracy'])
        model.fit(train_dataset.shuffle(100).batch(16),
                  epochs=3,
                  batch_size=16,
                  validation_data=val_dataset.shuffle(100).batch(16))
    
        #save the checkpoint
        model.save_pretrained("./TriviaQuestion2NQ_Transform_Dataset/BERT_Classification/answer_type_classification_model/")


# Main

class QuestionRewriter:

  def __init__(self, lat_frequency, min_length, to_trim, valid_verbs, remove_dict, non_last_sent_transform_dict,heuristicsTransformer, answerTypeClassifier):
    self.lat_frequency = lat_frequency

    # Minimum number of tokens in a chunk
    self.min_length = min_length

    # Words or punctuation we remove from the start/end of the sentence
    self.to_trim_start = re.compile("^(" + "|".join(
      "%s " % x for x in to_trim) + ")*")
    self.to_trim_end = re.compile("(" + "|".join(
      " %s" % x for x in to_trim) + ")*$")

    self.valid_verbs = valid_verbs
    self.remove_dict = remove_dict
    self.non_last_sent_transform_dict = non_last_sent_transform_dict
    self.heuristicsTransformer = heuristicsTransformer
    self.answer_type_dict = heuristicsTransformer.answer_type_dict
    self.last_sent_word_transform_30000 = heuristicsTransformer.last_sent_word_transform_30000
    
    
    self.answerTypeClassifier = answerTypeClassifier

  def trim_chunk(self, chunk):
    """
    Remove non-content conjunctions and punctuation from start or end of chunk
    """

    chunk = self.to_trim_start.sub("", chunk)
    chunk = self.to_trim_end.sub("", chunk)

    return chunk


      # Find coref clusters
      # clusters = doc._.coref_clusters
      # Breakdown sentences using Parse Trees

#   def generate_candidate_chunks_from_qb_question(self, question):
#     doc = nlp(question)
#     seen = set() # keep track of covered words
#     chunks = []

#     for sent in doc.sents:

#       # For all of the individual verbs, generate subsequences as candidates
#       for verb in [x for x in sent if x.pos_ == "VERB"]:
#         chunks.append((verb.left_edge.i,
#                       " ".join(x.text for x in verb.subtree)))

#       # If there's a conjunction, find its parent and give both the
#       # left and right form of it
#       for conj in [x for x in sent if x.pos_=="CCONJ"]:
#         if conj.has_head() and conj.head.has_head():
#           conj_parent = conj.head.head

#           left = " ".join(x.text for x in sent[conj_parent.left_edge.i:conj.i])

#           right = " ".join(x.text for x in sent[conj_parent.left_edge.i:
#                                                 conj.head.left_edge.i]) + " ~ "
#           right += " ".join(x.text for x in
#                             sent[conj.i + 1:conj_parent.right_edge.i + 1])

#           chunks.append((conj_parent.left_edge.i, left))
#           chunks.append((conj_parent.left_edge.i, right))


#       # I saw some Advcl code here that clearly wouldn't work, please
#       # rewrite using the dependency parse code as an example.
#       # unseen = [ww for ww in sent if ww not in seen]
#       # chunk = ' '.join([ww.text for ww in unseen])
#       # chunks.append( (sent.root.i, chunk) )

#       # Sort the chunks based on word index to ensure first sentences
#       # formed come first
#     chunks = sorted(chunks, key=lambda x: x[0])
#     return chunks

  def generate_candidate_chunks_from_qb_question(self, question):
      # input: single qb_question
      sample = question.strip()
      sample = sample.strip('.')
      doc = nlp(sample)
      chunks = []
      seen = set()
      clusters = doc._.coref_clusters
      for sent in doc.sents:
          conj_heads = [cc for cc in sent.root.children if cc.dep_ == 'conj']
          advcl_heads = [cc for cc in sent.root.children if cc.dep_ == 'advcl']
          heads = conj_heads + advcl_heads
          for head in heads:
              words = [ww for ww in head.subtree]
              for word in words:
                  seen.add(word)
              chunk = (' '.join([ww.text for ww in words]))
              chunks.append( (head.i, chunk) )
          unseen = [ww for ww in sent if ww not in seen]
          chunk = ' '.join([ww.text for ww in unseen])
          chunks.append( (sent.root.i, chunk) )
      
      # Sort the chunks based on word index to ensure first sentences formed come first
      chunks = sorted(chunks, key=lambda x: x[0])
      return chunks, clusters
      
  def substitute_pronouns(self, curr_chunk, clusters):
    chunk_doc = nlp(curr_chunk)
    for id, w in enumerate(chunk_doc):
      if w.tag_ in ['NN', 'NNP', 'NNS', 'NNPS']:
        continue
      rep = w.text
      for cluster in clusters:
        #print('Noun chunks: ', cluster[0], '->', [x for x in cluster[0].noun_chunks])
        if (len([x for x in cluster[0].noun_chunks]) > 0) and (str(cluster[0]).lower() not in self.heuristicsTransformer.pronouns):
          match_cluster = [str(cc) for cc in cluster]
          #print(match_cluster)
          if w.text in match_cluster:
            rep = match_cluster[0]
            if w.text.lower() in ['his', 'her', 'its', 'it\'s']:
              rep += '\'s'
              #print(f'Found {w} in cluster!!!')
              #print('Replaceing with ', match_cluster[0])
              break
            if not w.text == rep:
              replacement_list = [str(c) for c in chunk_doc]
              replacement_list[id] = rep
              curr_chunk = (' ').join(replacement_list)
            else:
              curr_chunk = '' + curr_chunk
    return curr_chunk
      
  def filter_chunks_by_size(self, candidates):
      chunks = candidates
      # Ensure no sentences aren't too small
      if len(chunks)>1:
        for idx in range(1, len(chunks)):
          try:
            curr_i, curr_chunk = chunks[idx]
          except:
            #print('idx=',idx)
            #print('chunk len = ', len(chunks))
            raise NotImplementedError

          # JBGCOMMENT: Shouldn't the length filter happen after
          # trimming?
          if len(curr_chunk.split()) < self.min_length or \
             (curr_chunk.split()[0] in ['after']):
            last_i, last_chunk = chunks[idx-1]
            last_chunk = last_chunk + ' ' + curr_chunk
            chunks[idx-1] = (last_i, last_chunk)
            del chunks[idx]
          if (idx+1)>=len(chunks):
            break
        curr_i, curr_chunk = chunks[0]
        if len(curr_chunk.split()) < self.min_length and len(chunks)>1:
          # Found a small pre-sent!
          last_i, next_chunk = chunks[1]
          curr_chunk = curr_chunk + ' ' + next_chunk
          chunks[0] = (last_i, curr_chunk)
          del chunks[1]
      return chunks

  def preprocess_last_sent(self,q):
    # to make the last sentence start from the content after 'FTP's (name this/what)
    # merge the content before 'FTP's into previous sentence
    q_chunks = ''
    for k,v in self.remove_dict.items():
      index = q.find(k)
      if index!=-1:
        q_chunks = q[:index] # should merge to previous setence
        q = q[index:]
        break
    for k,v in self.remove_dict.items():
      q = re.sub(k, v, q)
    return q, q_chunks

  def last_sent_transform(self,q_with_the_chunks):
    q, q_chunks = self.preprocess_last_sent(q_with_the_chunks)
    if q.split(' ')[:2] == ['name', 'this'] or q.split(' ')[:2] == ['identify', 'this'] or q.split(' ')[:2] == ['give', 'this'] or q.split(' ')[:2] == ['name', 'the'] \
    or q.split(' ')[:2] == ['Name', 'this'] or q.split(' ')[:2] == ['Identify', 'this'] or q.split(' ')[:2] == ['Give', 'this'] or q.split(' ')[:2] == ['Name', 'the'] \
    or q.split(' ')[:2] == ['Give', 'the'] or q.split(' ')[:2] == ['give', 'the']:
      doc = nlp(q)
      tok = []
      flag=0
      for i,token in enumerate(doc[2:6]):
        if token.pos_ == 'NOUN':
          #print('Noun Token = ', token)
          tok.append(str(token))
          flag=1
        else:
          if flag:
            break
      word  = (' ').join(tok)

      replacement = 'which is the'
      for k,v in self.last_sent_word_transform_30000.items():
        if k == 'unk':
          continue
        if word in v:
          replacement = k
          break

      transformed_q = q.split(' ')
      transformed_q = transformed_q[2:]
      transformed_q = (' ').join(transformed_q)
      transformed_q = replacement + ' ' + transformed_q
    elif q.split(' ')[:2] == ['name', 'these'] or q.split(' ')[:2] == ['identify', 'these'] or q.split(' ')[:2] == ['give', 'these'] \
    or q.split(' ')[:2] == ['Name', 'these'] or q.split(' ')[:2] == ['Identify', 'these'] or q.split(' ')[:2] == ['Give', 'these'] \
    or q.split(' ')[:2] == ['Give', 'the'] or q.split(' ')[:2] == ['give', 'the']:
      doc = nlp(q)
      tok = []
      flag=0
      for i,token in enumerate(doc[2:6]):
        if token.pos_ == 'NOUN':
          #print('Noun Token = ', token)
          tok.append(str(token))
          flag=1
        else:
          if flag:
            break
      word  = (' ').join(tok)

      replacement = 'which are the'
      for k,v in self.last_sent_word_transform_30000.items():
        if not k == 'unk':
          continue
        if word in v:
          replacement = k
          break
      transformed_q = q.split(' ')
      transformed_q = transformed_q[2:]
      transformed_q = (' ').join(transformed_q)
      transformed_q = replacement + ' ' + transformed_q
    else:
      transformed_q = q
    transformed_q = q_chunks+' '+transformed_q
    # remove adjancent duplicates
    q = self.remove_duplicates(q)
    q = q[0].lower()+q[1:]
    return transformed_q.strip()

  def intermediate_sent_transform(self, qb_id, q):
    qb_id = str(qb_id)
    # capitalize the sentences after the answer_type extraction [Aug23: and deal with no pronous cases]
    self.capitalization(q)

    qb_id = str(qb_id) # match the answer type from answer_type_dict
    q_orig = q
    FLAG = 0
    if qb_id in self.answer_type_dict.keys():
      answer_type = self.answer_type_dict[qb_id] # get the answer type from qb_id
      # detect if the answer_type (noun) is a person or a thing
      if answer_type in self.last_sent_word_transform_30000['who is the']:
        # answer_type is PERSON
        replacement_prefix = 'which'
        replacement = replacement_prefix+' '+answer_type
        # he/He/he's/He's/his/His/who/Who/whose/Whose

        for k in ['He ', 'Who ', 'She ']:
          q = re.sub(k, replacement+' ', q, 1)
          if not q_orig == q:
            FLAG = 1
            break
        if FLAG:
          return self.steps_before_return(q)
        for k in ['This ']:
          q = re.sub(k, 'Which ', q, 1)
          if not q_orig == q:
            FLAG = 1
            break
        if FLAG:
          return self.steps_before_return(q)
        for k in [' he ', ' who ', ' she ', ' him ']:
          q = re.sub(k, ' '+replacement+' ', q, 1)
          if not q_orig == q:
            FLAG = 1
            break
        if FLAG:
          return self.steps_before_return(q)
        for k in [' this ']:
          q = re.sub(k, ' '+' which ', q, 1)
          if not q_orig == q:
            FLAG = 1
            break
        if FLAG:
          return self.steps_before_return(q)
        for k in ['He\'s ', 'His ', 'Whose ', 'She\'s ', 'Her ']:
          q = re.sub(k, replacement+'\'s'+' ', q, 1)
          if not q_orig == q:
            FLAG = 1
            break
        if FLAG:
          return self.steps_before_return(q)
        for k in [' he\'s ', ' his ', ' whose ', ' she\'s ', ' her ']:
          q = re.sub(k, ' '+replacement+'\'s'+' ', q, 1)
          if not q_orig == q:
            FLAG = 1
            break
        if FLAG:
          return self.steps_before_return(q)
      # answer type is not in the last_sent_word_transform_30000 dictionary
      else:
        # classified as PERSON by BERT
        classification_output = self.answerTypeClassifier.get_answer_type_group(answer_type)
        if classification_output == 'PERSON':
          # answer_type is PERSON
          replacement_prefix = 'which'
          replacement = replacement_prefix+' '+answer_type
          # he/He/he's/He's/his/His/who/Who/whose/Whose
          for k in ['He ', 'Who ', 'She ']:
            q = re.sub(k, replacement+' ', q, 1)
            if not q_orig == q:
              FLAG = 1
              break
          if FLAG:
            return self.steps_before_return(q)
          for k in ['This ']:
            q = re.sub(k, 'Which ', q, 1)
            if not q_orig == q:
              FLAG = 1
              break
          if FLAG:
            return self.steps_before_return(q)
          for k in [' he ', ' who ', ' she ', ' him ']:
            q = re.sub(k, ' '+replacement+' ', q, 1)
            if not q_orig == q:
              FLAG = 1
              break
          if FLAG:
            return self.steps_before_return(q)
          for k in [' this ']:
            q = re.sub(k, ' which ', q, 1)
            if not q_orig == q:
              FLAG = 1
              break
          if FLAG:
            return self.steps_before_return(q)
          for k in ['He\'s ', 'His ', 'Whose ', 'She\'s ', 'Her ']:
            q = re.sub(k, replacement+'\'s'+' ', q, 1)
            if not q_orig == q:
              FLAG = 1
              break
          if FLAG:
            return self.steps_before_return(q)
          for k in [' he\'s ', ' his ', ' whose ', ' she\'s ', ' her ']:
            q = re.sub(k, ' '+replacement+'\'s'+' ', q, 1)
            if not q_orig == q:
              FLAG = 1
              break
          if FLAG:
            return self.steps_before_return(q)
        else:
          # answer_type is a thing
          replacement_prefix = 'which'
          replacement = replacement_prefix+' '+answer_type
          # swap in with the replacement
          # what/What/what's/What's/it/It/it's/It's/its/Its -> what/What+replacement
          for k in ['What ', 'It ']:
            q = re.sub(k, replacement+' ', q, 1)
            if not q_orig == q:
              FLAG = 1
              break
          if FLAG:
            return self.steps_before_return(q)
          for k in ['This ']:
            q = re.sub(k, 'Which ', q, 1)
            if not q_orig == q:
              FLAG = 1
              break
          if FLAG:
            return self.steps_before_return(q)
          for k in [' what ', ' it ']:
            q = re.sub(k, ' '+replacement+' ', q, 1)
            if not q_orig == q:
              FLAG = 1
              break
          if FLAG:
            return self.steps_before_return(q)
          for k in [' this ']:
            q = re.sub(k, ' which ', q, 1)
            if not q_orig == q:
              FLAG = 1
              break
          if FLAG:
            return self.steps_before_return(q)
          for k in ['What\'s ', 'Its ', 'It\'s ']:
            q = re.sub(k, replacement+'\'s'+' ', q, 1)
            if not q_orig == q:
              FLAG = 1
              break
          if FLAG:
            return self.steps_before_return(q)
          for k in [' what\'s ', ' its ', ' it\'s ']:
            q = re.sub(k, ' '+replacement+'\'s'+' ', q, 1)
            if not q_orig == q:
              FLAG = 1
              break
          if FLAG:
            return self.steps_before_return(q)
    else:
        for k,v in self.non_last_sent_transform_dict.items():
          q = re.sub(' '+k, ' '+v, q, 1)
          if q.startswith(k):
            q = v + q[len(k):]
    return self.steps_before_return(q)

  def steps_before_return(self,q):
    # remove adjancent duplicates
    q = self.remove_duplicates(q)
    q = q[0].lower()+q[1:]
    return q.strip()

  def deal_with_no_pronouns_cases(self,qb_id, q):
    qb_id = str(qb_id)
    # input: questions after the parse tree steps and before transformation
    q = q[0].lower()+q[1:]

    question_test = nlp(q)
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
    return

  def single_question_transform(self, qb_id, question):
      # parse tree
      qb_id = str(qb_id)
      nq_like_questions = []
      orig_output_before_transformation = []

      # generate candidates from qb_question
      chunks, clusters = self.generate_candidate_chunks_from_qb_question(question)
      #for variant, chunk in enumerate(chunks):
      for idx in range(len(chunks)):
        id, chunk = chunks[idx]
        # Clean each sentence of trailing and, comma etc
        trimmed_chunk = self.trim_chunk(chunk)
        # Coreference subsitution
        trimmed_chunk = self.substitute_pronouns(trimmed_chunk, clusters)
        chunks[idx] = (id, trimmed_chunk)
    
      # filter chunks by size
      filtered_chunks = self.filter_chunks_by_size(chunks)

      #print('\033[1m'+'Different nq like statements: (after 2nd breakdown):')
      for ii, chunk in filtered_chunks:
        # with the same qid
        nq_like_questions.append(chunk)
        orig_output_before_transformation.append(chunk)
      for i in range(len(nq_like_questions)):
        # check if no pronouns in the question
        try:
          self.deal_with_no_pronouns_cases(qb_id, nq_like_questions[i])
          if i == len(nq_like_questions)-1:
            # last sent transformation
            nq_like_questions[i] = self.last_sent_transform(nq_like_questions[i])
            nq_like_questions[i] = self.heuristicsTransformer.quality_check(qb_id, nq_like_questions[i])
          else:
            # intermediate sent transformation
            nq_like_questions[i] = self.intermediate_sent_transform(qb_id, nq_like_questions[i])
            nq_like_questions[i] = self.heuristicsTransformer.quality_check(qb_id, nq_like_questions[i])
        except:
          continue
      # return a NQlike list from one qb question
      nq_like_questions_with_its_orig_outputs = []
      for nq_like, orig_output in zip(nq_like_questions, orig_output_before_transformation):
          nq_like_questions_with_its_orig_output = {}
          nq_like_questions_with_its_orig_output['nq_like_questions'] = nq_like
          nq_like_questions_with_its_orig_output['orig_output_before_transformation'] = orig_output
          nq_like_questions_with_its_orig_outputs.append(nq_like_questions_with_its_orig_output)
      return nq_like_questions_with_its_orig_outputs

  def transform_questions(self, input_file, limit):
    if limit > 0:
      qb_df = pd.read_json(input_file, lines=True, orient='records',nrows=limit)
    else:
      qb_df = pd.read_json(input_file, lines=True, orient='records')


    qb_questions_input = qb_df['question'].values
    qb_id_input = qb_df['qanta_id'].values

    # transformation
    transformed = defaultdict(list)
    for qq, ii in zip(qb_questions_input, qb_id_input):
      # transform single QB
      transformed[ii] = self.single_question_transform(ii, qq)

      if ii % 1000 == 0:
        print(transformed[ii])

    return transformed, qb_id_input
    
  # helper functions
  def capitalization(self, q):
    q = q[0].upper()+q[1:]
    return

  def remove_duplicates(self, q):
    words = q.split()
    for i, w in enumerate(words):
      if i >= (len(words)-1):
        continue
      w2 = words[i+1]
      w2 = re.sub('\'s', '', w2)
      if w == w2:
        words = words[:i]+words[i+1:]
    q = " ".join(words)
    return q


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Apply heuristic functions")
  parser.add_argument('--limit', type=int,
                      default=20,help="Limit of number of QB questions input")
  parser.add_argument('--qb_path', type=str,
                      default='./TriviaQuestion2NQ_Transform_Dataset/qb_train_with_contexts_lower_nopunc_debug_Feb24.json',
                      help="path of the qb dataset")
  parser.add_argument('--save_result', type=bool, default=True,
                      help="Save NQlike questions")
  parser.add_argument('--lat_freq', type=str, default='lat_frequency.json',
                      help="JSON of frequency for each LAT")
  parser.add_argument('--answer_type_classifier', type=bool, default=False,
                      help="Retrain the answer type classifier from scratch")
  parser.add_argument('--raw_text_output', type=bool, default=True,
                      help="Save both the raw output before transformation and the transformed questions")
  parser.add_argument('--min_chunk_length', type=int, default=5,
                      help="How long must extracted segment of QB question be?")
  parser.add_argument('--config_file', type=str, default='config.json',
                      help="File with data that configures extraction")

  args = parser.parse_args()
  # Load dataset
  qb_path = args.qb_path
  limit = args.limit
  qb_df = None

  answer_type_dict = pd.read_json(args.lat_freq, lines=True, orient='records').to_dict()

  # read contents from config.json file
  with open(args.config_file) as json_file:
    config = json.load(json_file)

  tokenizer = DistilBertTokenizerFast.from_pretrained(config["tokenizer"])
  loaded_model = TFDistilBertForSequenceClassification.from_pretrained(config["loaded_model"])
  with open(config["last_sent_word_transform_30000"], 'r') as f:
    last_sent_word_transform_30000 = json.load(f)
  

  transformer = HeuristicsTransformer(config)

  answerTypeClassifier = AnswerTypeClassifier(last_sent_word_transform_30000=last_sent_word_transform_30000, tokenizer=tokenizer, loaded_model=loaded_model)

  rewriter = QuestionRewriter(answer_type_dict, args.min_chunk_length,
                              to_trim=config["to_trim"],
                              valid_verbs=config["valid_verbs"],
                              remove_dict=config["remove_dict"],
                              non_last_sent_transform_dict=config["non_last_sent_transform_dict"],
                              heuristicsTransformer=transformer,
                              answerTypeClassifier=answerTypeClassifier)

  # retraining the answer type classifier
  if args.answer_type_classifier:
      print("retraining the answer type classifier from scratch......")
      answerTypeClassifier.answer_type_classifier_training()

  # save NQlike questions
  if args.save_result:
    if args.raw_text_output:
        nq_like_df = {
          'qanta_id':[],
          'question':[],
          'orig_output_before_transformation':[],
        }
    else:
        nq_like_df = {
          'qanta_id':[],
          'question':[],
        }

    transformed, qb_id_input = rewriter.transform_questions(qb_path, limit)
    for key in transformed.keys():
        # nqlike questions for a single QB sample
        nqlist = transformed[key]
        for i in range(len(nqlist)):
            nq_like_df['qanta_id'].append(str(key))
            nq_like_df['question'].append(nqlist[i]['nq_like_questions'])
            if args.raw_text_output:
                nq_like_df['orig_output_before_transformation'].append(nqlist[i]['orig_output_before_transformation'])
    new_nqlike = pd.DataFrame(nq_like_df)
    new_nqlike.to_json('./nq_like_questions.json', lines=True, orient='records')

    # prepare NQlike and QB with contexts datasets for the classifier retraining QA
    qb_id_list = qb_id_input.tolist()
    qb_df = pd.read_json(qb_path, lines=True, orient='records')
    selected_qb_df = qb_df.loc[qb_df.apply(lambda x: x.qanta_id in qb_id_list, axis=1)]
    selected_qb_df = selected_qb_df.rename(columns={'score': 'quality_score'})
    # save QB_with_contexts
    selected_qb_df.to_json('./qb_with_contexts.json', lines=True, orient='records')
    # mapping nq_like with contexts
    context_list = []
    char_spans_list = []
    answer_list = []
    for idx in qb_id_list:
        context_list.append(selected_qb_df.loc[selected_qb_df['qanta_id'] == idx]['context'])
        char_spans_list.append(selected_qb_df.loc[selected_qb_df['qanta_id'] == idx]['char_spans'])
        answer_list.append(selected_qb_df.loc[selected_qb_df['qanta_id'] == idx]['answer'])
    # save nqlike_with_contexts
    #nqlike_list = new_nqlike.loc[new_nqlike.apply(lambda x: x.qanta_id in qb_id_list, axis=1)]['question'].tolist()
    nqlike_list = new_nqlike['question'].tolist()
    nqlike_with_contexts_df = pd.DataFrame(list(zip(qb_id_list, nqlike_list, context_list, char_spans_list, answer_list)), columns =['qanta_id', 'question', 'context', 'char_spans', 'answer'])
    nqlike_with_contexts_df['quality_score'] = 1
    nqlike_with_contexts_df.to_json('./nqlike_with_contexts.json', lines=True, orient='records')
