import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import re
import json
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
import sklearn.model_selection
import sklearn.preprocessing as preproc
from sklearn.feature_extraction import text
from sklearn import tree
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from collections import Counter
import pickle
import warnings
warnings.filterwarnings("ignore")
import nltk
import argparse


# word2vec library
import gensim.downloader
import gensim.models
from gensim.test.utils import common_texts
# Word2Vec initialization
# w2v_model = gensim.downloader.load('glove-wiki-gigaword-300')

# unigram initialization
MAXFEATURES = 5000
unigram_converter = CountVectorizer(max_features=MAXFEATURES)


nq_path = './NaturalQuestions_train_reformatted_Feb24.json'
qb_path = './qb_train_with_contexts_lower_nopunc_debug_Feb24.json'
nq_like_path = './qb_train_with_contexts_lower_nopunc_debug_Feb24.json'

# Step 6: Functions to extract features

# function to compute length feature
def get_length(df_train):
  return df_train['question'].str.split().apply(len).values

#function to compute kincaid readability score
def get_kincaid(df_train):
  return df_train['question'].apply(textstat.flesch_kincaid_grade).values

# function to compute duplicate feature
def count_duplicates(x):
    x_list = x.lower().split(' ')
    counter = Counter(x_list)
    i = 0
    for e in counter:
        if counter[e] > 1:
            i = i+1
    return i

def get_duplicates(df_train):
  return df_train['question'].apply(count_duplicates).values

# function to compute kincaid readability score
def get_kincaid(df_train):
  return df_train['question'].apply(textstat.flesch_kincaid_grade).values

# function to get number of nouns
def count_num_nouns(text):
    tokens = nltk.word_tokenize(text.lower())
    text = nltk.Text(tokens)
    tags = nltk.pos_tag(text)
    counts = Counter(tag for word,tag in tags)
    noun_num = counts['NN']
    return noun_num

def get_num_nouns(df_train):
  return df_train['question'].apply(count_num_nouns).values

# function to get number of verbs
def count_num_verbs(text):
    tokens = nltk.word_tokenize(text.lower())
    text = nltk.Text(tokens)
    tags = nltk.pos_tag(text)
    counts = Counter(tag for word,tag in tags)
    verb_num = counts['VB']
    return verb_num

def get_num_verbs(df_train):
  return df_train['question'].apply(count_num_verbs).values

# function compute maximum idf score feature
def extract_terms( document ):
   terms = document.lower().replace('?',' ').replace('.',' ').replace(',',' ').split()
   return terms

def calculate_idf( documents ):
   N = len(documents)
   from collections import Counter
   tD = Counter()
   for d in documents:
      for f in d:
          tD[f] += 1
   IDF = {}
   import math
   for (term,term_frequency) in tD.items():
       term_IDF = math.log(float(N) / term_frequency)
       IDF[term] = term_IDF
   return IDF

def get_max_idf(df_train):
  documents = df_train['question'].apply(extract_terms)
  IDF = calculate_idf(documents)
  max_idf = []
  for doc in documents:
    idf_lst = []
    for t in doc:
      idf_lst.append(IDF[t])
    if len(idf_lst) != 0:
      max_idf.append(max(idf_lst))
    else:
      max_idf.append(len(documents))
  return max_idf

def get_word2vec(df_train):
  return df_train['question'].apply(vectorize).values

def get_unigram(df_train):
  umatrix = []
  if len(df_train) < MAXFEATURES:
    umatrix = unigram_converter.transform(df_train['question'])
  else:
    umatrix = unigram_converter.fit_transform(df_train['question'])
  unigram_list = unigram_converter.inverse_transform(umatrix)
  return unigram_list

# function transform the training data to train
def transform_data(df):
    # dictionary to store the features
    feature_dictionary = {}
    y_train = df['score'].values
    feature_dictionary['label'] =  y_train
    feature_dictionary['length'] = get_length(df)
    feature_dictionary['duplicates'] = get_duplicates(df)
    #feature_dictionary['kincaid'] = get_kincaid(df)
    #feature_dictionary['num_nouns'] = get_num_nouns(df)
    #feature_dictionary['num_verbs'] = get_num_verbs(df)
    feature_dictionary['max_idf'] = get_max_idf(df)
    # feature_dictionary['unigram'] = get_unigram(df)
    #feature_dictionary['word2vec'] = get_word2vec(df_train)
    return feature_dictionary

# Step 7

# helper function to prepare data for classifier
def get_x_vals(dictionary):
  x_vals = np.zeros((len(dictionary['label']), 1))
  for key in dictionary.keys():
    if key != 'label' and key != 'word2vec':
      if key == 'unigram':
        x = [' '.join(it) for it in dictionary['unigram']]
        x = unigram_converter.transform(x)
        x_vals = np.concatenate((x_vals, x.toarray()), axis=1)
      else:
        x = np.reshape(dictionary[key], (-1, 1))
        x_vals = np.concatenate((x_vals, x), axis=1)
  
  if 'word2vec' in dictionary.keys():
      new_x_vals = []
      all_vectors = reshape_all(dictionary['word2vec'])
      for i in range(0,len(all_vectors)):
        new_x_vals.append(np.concatenate((x_vals[i],all_vectors[i])))
      x_vals = new_x_vals
  x_vals = np.delete(x_vals, 0, 1)
  return x_vals

# function to train classifier    
def train_classifier(train_dict):
    y_train = train_dict['label']
    x_train = get_x_vals(train_dict) 

    model = LogisticRegression().fit(x_train,y_train)
    return model

# function to evaluate classifier
def evaluate(model, df_test, is_nq_like=False):
    test_dict = transform_data(df_test)
    
    x = get_x_vals(test_dict)
    results = model.predict_proba(x)
    test_dict['prob_scores'] = results
    predictions = model.predict(x)
    test_dict['prediction'] = predictions
    # accuracy
    if not is_nq_like:
      acc = model.score(x, df_test.score)
      print('Accuracy: '+format(acc)+'\n')
    return test_dict

def generate_feature_weight(model, names):
  feature_weight = {}
  weights = model.coef_[0]
  for i in range(0,len(names)-1):
    feature_weight[names[i+1]] = weights[i]

  return feature_weight

def save_dictionary(questions, dict_data, file_path):
  columns = dict_data.keys()
  len_of_data = len(questions)
  dict_to_write = {}
  for col in columns:
    dict_to_write[col] = dict_data[col]
  df = pd.DataFrame(data=dict_to_write)
  df.to_csv(file_path)
  
  
# Step 8

# function average word2vec vector
def avg_feature_vector(words, model, num_features, ind2key_set):
    feature_vec = np.zeros((num_features, ), dtype='float32')
    n_words = 0
    for word in words:
        if word in ind2key_set:
            n_words += 1
            feature_vec = np.add(feature_vec, model[word])
    if (n_words > 0):
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec

# define vectorizer
def word2vec_vectorizer(data, model,num_features,ind2key_set):
    sentence = data.lower().split(' ')
    return avg_feature_vector(sentence,model,num_features,ind2key_set)

# vectorize function
def vectorize(question):
    return word2vec_vectorizer(question,w2v_model,300,set(w2v_model.index_to_key))

# reshape word2vec vectors
def reshape_all(w2v_vectors):
    a = len(w2v_vectors)
    b = 300
    all_vectors = []
    for v in w2v_vectors:
        for e in v:
            all_vectors.append(e)
    all_vectors = np.reshape(all_vectors, (a, b))
    return all_vectors
  


  
# Step 9

# flag = 0 (0 flag refers to wellformedness classifier with an accuracy score) 
# flag = 1 (1 flag refers to NQ-like classifier with 0 accuracy score as we do not have labeled qualityscore for NQ-like)
# qb_last = True: only consider the last sentence of the qb questions
# qb_last = False: consider the whole qb questions
if __name__=="__main__":

  parser = argparse.ArgumentParser(description="Create classifier to discriminate synthetic questions from real questions")
  parser.add_argument('--limit', type=int, default=-1)
  parser.add_argument('--test_predictions', type=str, default='test_feature_dict_QB_NQ.csv')
  parser.add_argument('--features', type=str, default='nqlike_feature_dict_QB_NQ.csv')
  args = parser.parse_args()
	# set flag and if_qb_last_sent here
	# 0 --wellformedness accuracy output
	# 1 --NQ-like output
  qb_last = True
  
  # transform dataset
  qb_train = pd.read_json(qb_path, lines=True, orient='records')
  if args.limit >= 0:
    qb_train = qb_train.head(args.limit)
  qb_len = len(qb_train)
  
  if qb_last:
    qb_last = []
    for i in range(qb_len):
      qb_last.append(nltk.tokenize.sent_tokenize(qb_train.iloc[i]['question'])[-1].lower())
    del qb_train['question']
    qb_train['question'] = qb_last
  nq_train = pd.read_json(nq_path, lines=True, orient='records')
  if args.limit >= 0:
    nq_train = nq_train.head(args.limit)

  df = qb_train.append(nq_train, ignore_index=True)
  training_data, test_data = sklearn.model_selection.train_test_split(df, train_size = 0.75, random_state=42)
  df_train = training_data
  df_test = test_data
  
  train_feature_dict = transform_data(df_train)
  train_feature_df = pd.DataFrame.from_dict(train_feature_dict, orient='index').transpose()
  
	# uncomment to save results
  save_dictionary(df_train['question'],train_feature_dict, './train_feature_dict_QB_NQ.csv')
	# train classifier
  model = train_classifier(train_feature_dict)

  print(train_feature_df.head())
  
	# save feature weight
  weight_dict = generate_feature_weight(model, list(train_feature_dict.keys()))
  with open('logistic_regression_weight_dict_Qb_NQ.txt', 'w') as f:
    f.write(json.dumps(weight_dict))
  
  if args.test_predictions:
		# test
    print('QB NQ Test ')
    test_dict = evaluate(model, df_test)
    test_feature_dict = transform_data(df_test)
    save_dictionary(df_test['question'],test_feature_dict, args.test_predictions)
    
  if args.features:
		# predict nq score for nq-like question
		# transform data
    df_nqlike = pd.read_json(nq_like_path)
    df_nqlike = df_nqlike[['qanta_id', 'question', 'quality_score']].copy()
    df_nqlike = df_nqlike.rename(columns={"quality_score": 'score'})
		# sample 5% of all questions as dataset too large
    df_nqlike = df_nqlike.sample(frac=0.05, replace=False).reset_index()
    nqlike_feature_dict = transform_data(df_nqlike)
		# predicting and store results
    print('NQ-Like ')
    eval_nqlike = evaluate(model, df_nqlike, True)
    save_dictionary(df_nqlike['question'],  eval_nqlike, args.features)
