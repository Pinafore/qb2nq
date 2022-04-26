import numpy as np
import pandas as pd
import json
import string
import nltk
import time
import os
import re
import random
import spacy
# import neuralcoref
from collections import Counter

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

nlp = spacy.load('en_core_web_sm')
# neuralcoref.add_to_pipe(nlp)

class LatFrequencyComputer:
  def __init__(self,remove_dict):
    self.remove_dict = remove_dict

  def uniques(self, your_string ):
      words = your_string.split()

      seen = set()
      seen_add = seen.add

      def add(x):
          seen_add(x)
          return x

      output = ' '.join( add(i) for i in words if i not in seen )
      return output

  def junk_last_sentence(self,q):
    # to make the last sentence start from the content after 'FTP's (name this/what)
    for k,v in self.remove_dict.items():
      index = q.find(k)
      if index!=-1:
        q = q[index:]
        break
    for k,v in self.remove_dict.items():
      q = re.sub(k, v, q)
    return q

  def get_answer_type(self,q):
    q = self.junk_last_sentence(q)
    word = ""
    # last sentence
    # find the answer type
    # simple case: extract NOUNs following 'name this's
    if q.split(' ')[:2] == ['name', 'this'] or q.split(' ')[:2] == ['identify', 'this'] or q.split(' ')[:2] == ['give', 'this'] or q.split(' ')[:2] == ['name', 'the'] \
    or q.split(' ')[:2] == ['Name', 'this'] or q.split(' ')[:2] == ['Identify', 'this'] or q.split(' ')[:2] == ['Give', 'this'] or q.split(' ')[:2] == ['Name', 'the'] \
    or q.split(' ')[:2] == ['Give', 'the'] or q.split(' ')[:2] == ['give', 'the']:
        doc = nlp(q)
        tok = []
        flag=0
        for i,token in enumerate(doc[2:12]):
          if token.pos_ == 'NOUN':
            #print('Noun Token = ', token)
            tok.append(str(doc[2:2+i]))
            tok.append(str(token))
            flag=1
          else:
            if flag:
              break
        word  = (' ').join(tok) # answer type
        # remove duplicates
        word = self.uniques(word)
        word  = word.strip()
        for k in ['" ', ', ']:
          word = re.sub(k, '', word)
    elif q.split(' ')[:1] == ['what'] or q.split(' ')[:1] == ['What']:
        doc = nlp(q)
        tok = []
        flag=0
        for i,token in enumerate(doc[1:12]):
          if token.pos_ == 'NOUN':
            tok.append(str(doc[1:1+i]))
            tok.append(str(token))
            flag=1
          else:
            if flag:
              break
        word  = (' ').join(tok)
        for k in ['is this ', 'was this ', 'are these ', 'were these ']:
          word = re.sub(k, '', word)
        word = self.uniques(word)
        word  = word.strip()
        for k in ['" ', ', ']:
          word = re.sub(k, '', word)
    elif q.split(' ')[:2] == ['name', 'these'] or q.split(' ')[:2] == ['identify', 'these'] or q.split(' ')[:2] == ['give', 'these'] \
    or q.split(' ')[:2] == ['Name', 'these'] or q.split(' ')[:2] == ['Identify', 'these'] or q.split(' ')[:2] == ['Give', 'these']:
        doc = nlp(q)
        tok = []
        flag=0
        for i,token in enumerate(doc[2:12]):
          if token.pos_ == 'NOUN':
            #print('Noun Token = ', token)
            tok.append(str(doc[2:2+i]))
            tok.append(str(token))
            flag=1
          else:
            if flag:
              break
        word  = (' ').join(tok)
        word = self.uniques(word)
        word  = word.strip()
        for k in ['" ', ', ']:
          word = re.sub(k, '', word)
    else:
        word = 'None'
    return word

def retrieve_answer_type_for_each_QB(orig_qb_path, lat_freq_computer):
  if os.path.exists(orig_qb_path) == False:
    print('Please check if {} exists in the current folder'.format(orig_qb_path))
  f1 = open(orig_qb_path)
  qb_data = json.load(f1)['questions']

  qanta_id = []
  qanta_questions_last = []
  qanta_questions_full = []
  qanta_answers = []
  qanta_page = []
  qanta_answer_type = []
  qanta_difficulty = []
  qanta_category = []
  qanta_subcategory = []
  qanta_year = []

  for i in range(len(qb_data)):
    if i%5000 == 0:
      print("===> "+str(i)+"/"+str(len(qb_data))+"\n")
    qanta_id.append(qb_data[i]['qanta_id'])
    qanta_questions_last.append(nltk.tokenize.sent_tokenize(qb_data[i]['text'])[-1])
    qanta_questions_full.append(qb_data[i]['text'])
    qanta_answers.append(qb_data[i]['answer'])
    qanta_page.append(qb_data[i]['page'])
    qanta_answer_type.append(lat_freq_computer.get_answer_type(nltk.tokenize.sent_tokenize(qb_data[i]['text'])[-1]))
    qanta_difficulty.append(qb_data[i]['difficulty'])
    qanta_category.append(qb_data[i]['category'])
    qanta_subcategory.append(qb_data[i]['subcategory'])
    qanta_year.append(qb_data[i]['year'])

  # save
  dataset1_lst = []
  for i in range(len(qb_data)):
      dataset1 = {}
      dataset1['qanta_id'] = qanta_id[i]
      dataset1['qanta_questions_last'] = qanta_questions_last[i]
      dataset1['qanta_questions_full'] = qanta_questions_full[i]
      dataset1['qanta_answers'] = qanta_answers[i]
      dataset1['qanta_page'] = qanta_page[i]
      dataset1['qanta_answer_type'] = qanta_answer_type[i]
      dataset1['qanta_difficulty'] = qanta_difficulty[i]
      dataset1['qanta_category'] = qanta_category[i]
      dataset1['qanta_subcategory'] = qanta_subcategory[i]
      dataset1['qanta_year'] = qanta_year[i]
      dataset1_lst.append(dataset1)
  with open("./qanta_train_with_answer_type.json", 'w') as f:
      for item in dataset1_lst:
          f.write(json.dumps(item) + "\n")
  return

def retrieve_most_freq_answer_type_for_qid(qanta_train_with_answer_type_path):
  if os.path.exists(qanta_train_with_answer_type_path) == False:
    print('Please check if {} exists in the current folder'.format(qanta_train_with_answer_type_path))
  qb_df = pd.read_json(qanta_train_with_answer_type_path, lines=True, orient='records')

  page_to_answer_type_dict = {}
  for i in range(len(qb_df)):
    key = qb_df.iloc[i]['qanta_page']
    if key not in page_to_answer_type_dict:
        page_to_answer_type_dict[key] = list()
    page_to_answer_type_dict[key].append(qb_df.iloc[i]['qanta_answer_type'])

  page_to_most_freq_answer_type_dict = {}
  for key in page_to_answer_type_dict.keys():
    lst = page_to_answer_type_dict[key]
    data = Counter(lst)
    most_freq = data.most_common(1)[0][0]
    page_to_most_freq_answer_type_dict[key] = most_freq

  # qid to most freq answer type
  most_freq_answer_type_lst = []
  for i in range(len(qb_df)):
    most_freq_answer_type = page_to_most_freq_answer_type_dict[qb_df.iloc[i]['qanta_page']]
    most_freq_answer_type_lst.append(most_freq_answer_type)

  qb_df['most_freq_answer_type'] = most_freq_answer_type_lst

  qid_to_answer_type_dict = {}
  for i in range(len(qb_df)):
    qid_to_answer_type_dict[str(qb_df.iloc[i]['qanta_id'])] = qb_df.iloc[i]['most_freq_answer_type']

  #save the most freq answer type for each qid into dictionary
  with open('./lat_frequency.json', 'w') as fp:
      json.dump(qid_to_answer_type_dict, fp)
  return

if __name__ == "__main__":
  orig_qb_path = 'qanta.train.2018.04.18.json'
  # load configuration 
  with open('config.json', 'r') as f:
    config = json.load(f)
  lat_freq_calculator = LatFrequencyComputer(config['remove_dict'])
  retrieve_answer_type_for_each_QB(orig_qb_path,lat_freq_calculator)

  qanta_train_with_answer_type_path = './qanta_train_with_answer_type.json'
  retrieve_most_freq_answer_type_for_qid(qanta_train_with_answer_type_path)
