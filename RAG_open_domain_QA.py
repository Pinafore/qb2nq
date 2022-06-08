import os
# os.environ['TRANSFORMERS_CACHE'] = '/fs/clip-quiz/saptab1/QA-MT-NLG/RAG-entireNQ/cache'
# os.environ['HF_DATASETS_CACHE'] = '/fs/clip-quiz/saptab1/QA-MT-NLG/RAG-entireNQ/cache'
import numpy as np
import pandas as pd
import json
import nltk
import time
import os
import re
import random
import argparse
import spacy
#import neuralcoref

import faiss
from functools import partial
from sklearn.model_selection import train_test_split

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset, RandomSampler

from datasets import Features, Value, load_dataset, Sequence, load_from_disk
from datasets import load_metric
import sacrebleu

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics.functional import bleu_score

#from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast
from transformers import StoppingCriteriaList, MaxLengthCriteria
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration, RagTokenForGeneration

print('Downloading NLTK packagaes for Bleu and Meteor Calculation...')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

RAGtokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
'''
RAGretriever = RagRetriever.from_pretrained("facebook/rag-token-nq", 
                                         index_name="custom", 
                                         passages_path=small_wiki_dataset_path,
                                         index_path=index_path)
'''
RAGretriever = RagRetriever.from_pretrained("facebook/rag-token-nq", index_name="exact", use_dummy_dataset=True)

RAGmodel = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=RAGretriever)
RAGmodel = RAGmodel.to(device)
RAGmodel.config.n_docs = 10
q_max = 22
a_max = 8

class RAGQA(pl.LightningModule):
  # Instantiate the model
  def __init__(self, learning_rate, tokenizer, model, hparams):
    super().__init__()
    self.tokenizer = tokenizer                  # For detokenizing generated sentences
    self.model = model                          # Actual RAG model
    self.learning_rate = learning_rate          # Learning rate 
    self.bleu = load_metric('bleu')             # Function for calculating the bleu scores
    self.meteor = load_metric('meteor')         # Function to calculate the meteor scores
    self.hyperparams = hparams                  # Hyperparamters
    self.train_gold = []
    self.train_pred = []
    self.val_gold = []
    self.val_pred = []

    # Freeze the Question Encoder to save time
    if self.hyperparams.freeze_question_encoder:  
      freeze_params(self.model.question_encoder)

    # Freeze embeddings of the Model
    if self.hyperparams.freeze_embeds:
      self.freeze_embeds()

  def train_exact_match(self, predictions, references):
    match = 0
    for rs,p in zip(references, predictions):
      for r in rs:
        r_tok = nltk.word_tokenize(r)
        p_tok = nltk.word_tokenize(p)
        if r_tok == p_tok:
          match += 1
          break
    #print('Match = ',match, ' Out of ', len(predictions))
    self.train_pred = []
    self.train_gold = []
    return match/len(predictions)
    
  def val_exact_match(self, predictions, references):
    match = 0
    for rs,p in zip(references, predictions):
      for r in rs:
        r_tok = nltk.word_tokenize(r)
        p_tok = nltk.word_tokenize(p)
        if r_tok == p_tok:
          match += 1
          break
    #print('Match = ',match, ' Out of ', len(predictions))
    self.val_pred = []
    self.val_gold = []
    return match/len(predictions)


  def freeze_embeds(self):
    ''' freeze the positional embedding parameters of the model; adapted from finetune.py '''
    freeze_params(self.model.model.shared)
    for d in [self.model.model.encoder, self.model.model.decoder]:
      freeze_params(d.embed_positions)
      freeze_params(d.embed_tokens)

  # Do a forward pass through the model
  def forward(self, input_ids, **kwargs):
    return self.model(input_ids, **kwargs)
  
  def configure_optimizers(self):
    optimizer = torch.optim.Adagrad(self.parameters(), lr = self.learning_rate)
    return optimizer

  # Training step for one batch
  def training_step(self, batch, batch_idx):
    # Load the data into variables
    src_ids, src_mask = batch['input_ids'], batch['attention_mask']
    tgt_ids = batch['labels']
    gold_answer = batch['gold_answer']
    
    # Run the model and get the logits
    outputs = self(src_ids, attention_mask=src_mask, labels = tgt_ids, use_cache=False)
    loss = outputs.loss.mean()

    # Calculate metrics
    generated_ids = self.model.generate(
        src_ids, 
        attention_mask = src_mask,
        use_cache=True,
        stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=40)]),
        early_stopping=True,
        num_beams=1)
    
    # Detokenize the generated tokens into text
    generated_answer = self.tokenizer.batch_decode(generated_ids.detach().cpu(), skip_special_tokens = True)   

    self.train_gold += gold_answer
    self.train_pred += generated_answer


    # Log the training loss and scores per batch
    #self.log('train_em', self.train_exact_match(self.train_pred, self.train_gold), on_step=False, on_epoch=True)
    self.log('train_loss', loss, on_step=False, on_epoch=True)

    return {'loss':loss}

  def training_epoch_end(self, X):
    self.log('train_em', self.train_exact_match(self.train_pred, self.train_gold), on_step=False, on_epoch=True)
    return 

  def validation_step(self, batch, batch_idx):
    src_ids, src_mask = batch['input_ids'], batch['attention_mask']
    tgt_ids = batch['labels']
    gold_answer = batch['gold_answer']
    
    # Run the model and get the logits
    outputs = self(src_ids, attention_mask=src_mask, labels=tgt_ids, use_cache=False)
    val_loss = outputs.loss.mean()
    sc = StoppingCriteriaList([MaxLengthCriteria(max_length=41)])
    generated_ids = self.model.generate(
        src_ids, 
        attention_mask = src_mask,
        use_cache=True,
        stopping_criteria = sc,
        num_beams=1)
    
    generated_answer = self.tokenizer.batch_decode(generated_ids.detach().cpu(), skip_special_tokens = True, clean_up_tokenization_spaces=True)   
    
    self.val_gold += gold_answer
    self.val_pred += generated_answer

    # Log the metrics
    #self.log('val_em', self.val_exact_match(self.val_pred, self.val_gold), on_step=False, on_epoch=True)
    self.log('val_loss', val_loss, on_step=False, on_epoch=True)
    
    return {'val_loss': val_loss}
  
  def validation_epoch_end(self, X):
    self.log('val_em', self.val_exact_match(self.val_pred, self.val_gold), on_step=False, on_epoch=True)
    return 

  # Method that generates text using the BartForConditionalGeneration's generate() method
  def generate_text(self, text, eval_beams, early_stopping = True, max_len = 40):
    ''' Function to generate text '''
    self.model.eval()
    generated_ids = self.model.generate(
        text['input_ids'], 
        attention_mask = text['attention_mask'],
        use_cache=False,
        stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=15)]),
        early_stopping=early_stopping,
        num_beams=eval_beams)
    
    generated_answer = self.tokenizer.batch_decode(generated_ids.detach().cpu(), skip_special_tokens = True)
    
    return generated_answer

def freeze_params(model):
  ''' Function that takes a model as input (or part of a model) and freezes the layers for faster training
      adapted from finetune.py '''
  for layer in model.parameters():
    layer.requires_grade = False

class QADataset(Dataset):
  def __init__(self, input_ids, att_mask, labels, gold, q):
    super().__init__()
    self.input_ids = input_ids
    self.attention_masks = att_mask
    self.labels = labels
    self.question_texts = q
    self.gold_answers = gold

  def __len__(self):
    return len(self.input_ids)
  
  def __getitem__(self, idx):
    input_id = self.input_ids[idx]
    attention_mask = self.attention_masks[idx]
    label = self.labels[idx]
    question_text = self.question_texts[idx]
    gold_answer = self.gold_answers[idx]

    return_dict = {'input_ids':input_id,
                   'attention_mask':attention_mask,
                   'labels':label,
                   'question_text':question_text,
                   'gold_answer': gold_answer}
    return return_dict
### ------ Creating Qanta Question-Answer Pair Dataset--------------###

class QADataModule(pl.LightningDataModule):
  def __init__(self, tokenizer, data_file, batch_size, q_transform=None, a_transform=None):
    super().__init__()
    self.tokenizer = tokenizer
    self.data_file = data_file
    self.batch_size = batch_size
    self.q_transforms = q_transform
    self.a_transforms = a_transform

  # Loads and splits the data into training, validation and test sets with a 60/20/20 split
  def prepare_data(self):
    self.train = self.data_file['train']#, lines=True, orient='records')[:self.num_examples]
    self.validate = self.data_file['dev']#, lines=True, orient='records')[:self.num_examples]
    self.test = self.data_file['test']#, lines=True, orient='records')[:self.num_examples]
  
  # encode the sentences using the tokenizer  
  def setup(self, stage):
    self.train = encode_sentence(self.train, self.tokenizer, self.q_transforms, self.a_transforms)
    self.validate = encode_sentence(self.validate, self.tokenizer, self.q_transforms, self.a_transforms)
    self.test = encode_sentence(self.test, self.tokenizer, self.q_transforms, self.a_transforms)

  # Load the training, validation and test sets in Pytorch Dataset objects
  def train_dataloader(self):
    dataset = QADataset(self.train['input_ids'], 
                        self.train['attention_mask'], self.train['labels'], 
                        self.train['gold_answer'], self.train['question_text'])                          
    train_data = DataLoader(dataset, sampler = RandomSampler(dataset), 
                            batch_size = self.batch_size, collate_fn=my_collate_fn)
    return train_data

  def val_dataloader(self):
    dataset = QADataset(self.validate['input_ids'], 
                        self.validate['attention_mask'], self.validate['labels'], 
                            self.validate['gold_answer'], self.validate['question_text']) 
    val_data = DataLoader(dataset, 
                          batch_size = self.batch_size, collate_fn=my_collate_fn)                       
    return val_data

  def test_dataloader(self):
    dataset = QADataset(self.test['input_ids'], 
                        self.test['attention_mask'], self.test['labels'], 
                        self.test['gold_answer'], self.test['question_text']) 
    test_data = DataLoader(dataset, 
                           batch_size = self.batch_size, collate_fn=my_collate_fn)                   
    return test_data

def encode_sentence(df, tokenizer, q_transforms, a_transforms):
  #tok_df = pd.DataFrame(columns=['input_ids', 'attention_mask', 'labels', 'question_text', 'gold_answer'])
  sample = {'input_ids':[], 'attention_mask':[], 'labels':[], 'question_text':[], 'gold_answer':[]}
  for idx,data in df.iterrows():
    question = data["question_text"]    # Load question
    answer = data["answer"]     # Load answer
    #print(answer)
    if q_transforms:
      if q_transforms.__name__ in ['random_deletion', 'random_insertion', 'random_swap']:
        question = q_transforms(question, self.p)               # Transform Question
      else:
        question = q_transforms(question)
    if a_transforms:
      answer = [a_transforms(ans) for ans in answer]                   # Transform Answer

    q = tokenizer.question_encoder(question,return_tensors = "pt", return_attention_mask=True,
                  padding='max_length', max_length=q_max+15, truncation=False)         # Tokenize Question

     
    with tokenizer.as_target_tokenizer():
      a = tokenizer(answer[0], return_tensors = "pt", padding='max_length', 
                    max_length=a_max+15, truncation=False)           # Tokenize Answer
    
    sample["input_ids"].append(q["input_ids"])
    sample["attention_mask"].append(q['attention_mask'])
    sample["labels"].append(a["input_ids"])
    sample["question_text"].append(question)
    sample["gold_answer"].append(answer)
  return sample

def my_collate_fn(batch):
  batch_size = len(batch)
  input_ids = batch[0]['input_ids']
  attention_masks = batch[0]['attention_mask']
  if 'labels' in batch[0].keys():
    labels = batch[0]['labels']
  questions = [batch[0]['question_text']]
  gold_answers = [batch[0]['gold_answer']]
  for i in range(1,batch_size):
    input_ids = torch.cat((input_ids, batch[i]['input_ids']), dim=0) 
    attention_masks = torch.cat((attention_masks, batch[i]['attention_mask']), dim=0)
    if 'labels' in batch[i].keys():
      labels = torch.cat((labels, batch[i]['labels']), dim=0)
    questions.append(batch[i]['question_text'])
    gold_answers.append(batch[i]['gold_answer'])
  return_dict = {'input_ids':input_ids,
                  'attention_mask' : attention_masks,
                 'labels':labels,
                 'question_text':questions,
                 'gold_answer':gold_answers}
  return return_dict

hparams = argparse.Namespace()

hparams.freeze_question_encoder = False
hparams.freeze_embeds = False
hparams.eval_beams = 1

def remove_punctuation(text):
    punct_list = ['{', '}']
    for punc in punct_list:
        if punc in text:
            text = text.replace(punc, '')
    partitioned_string = text.partition(' or ')
    before_first_period = partitioned_string[0]
    return before_first_period

class CheckpointEveryNSteps(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
        self,
        save_step_frequency,
        prefix="N-Step-Checkpoint",
        use_modelcheckpoint_filename=False,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

    def on_train_batch_end(self, trainer: pl.Trainer, *args):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 0:
            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = f"{self.prefix}_epoch={epoch}_global_step={global_step}.ckpt"
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Train RAG QA system")
    parser.add_argument('--q_train', type=str, required=True)
    parser.add_argument('--q_dev', type=str, required=True)
    parser.add_argument('--q_test', type=str, required=True)
    parser.add_argument('--checkpoint_PATH', type=str, required=True)
    args = parser.parse_args()

    train = pd.read_json(args.q_train, lines=True, orient='records')
    dev = pd.read_json(args.nq_dev, lines=True, orient='records')
    test = pd.read_json(args.nq_test, lines=True, orient='records')

    train = train.rename(columns={'question':'question_text'})
    dev = dev.rename(columns={'question':'question_text'})
    test = test.rename(columns={'question':'question_text'})

    data_file_dict = {'train':train, 'dev' : dev,'test' : test}

    QA_data = QADataModule(RAGtokenizer, data_file=data_file_dict, batch_size = 5, q_transform=None, a_transform=remove_punctuation)

    #model.load_from_checkpoint('/content/gdrive/Shareddrives/Improving-QA-MT/Colab/RAG QA/models/Quizdb/tmp/last.ckpt')
    QA_model = RAGQA(learning_rate = 8e-5, tokenizer = RAGtokenizer, model = RAGmodel, hparams = hparams)

    nstep_checkpoint = CheckpointEveryNSteps(1000)

    if not os.path.exists(checkpoint_PATH):
        os.makedirs(checkpoint_PATH)
        print('Creating New Directory...')

    modelcheckpoint = ModelCheckpoint(dirpath=checkpoint_PATH,monitor = 'val_em', save_last = True, save_top_k = -1, mode = 'max', verbose=1, every_n_epochs = 1)#every_n_train_steps=10)

    trainer = pl.Trainer(gpus = 1, precision=16,
                        max_epochs = 2,
                        min_epochs = 1,
                        auto_lr_find = False,
                        auto_scale_batch_size=False,
                        checkpoint_callback = True,
                        callbacks = [modelcheckpoint, nstep_checkpoint],
                        progress_bar_refresh_rate = 1)#, 
                        #resume_from_checkpoint=checkpoint_PATH+'epoch=2-step=1826.ckpt')

    trainer.fit(QA_model, QA_data)
    torch.cuda.empty_cache()

