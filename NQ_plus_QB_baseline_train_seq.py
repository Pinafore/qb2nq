#nq_train_path = sys.argv[2]
#nq_dev_path = sys.argv[3]

import sys
import os
# os.environ['TRANSFORMERS_CACHE'] = '/fs/clip-quiz/saptab1/QA-MT-NLG/Mar_2022/cache'
# os.environ['HF_DATASETS_CACHE'] = '/fs/clip-quiz/saptab1/QA-MT-NLG/Mar_2022/cache'
import pandas as pd
import numpy as np
import json
import random
import os
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
import datasets
from datasets import load_dataset,ClassLabel, Sequence
from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering, TrainingArguments, default_data_collator, Trainer, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device is: ",device)

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
max_length = 380 # The maximum length of a feature (question and context)
doc_stride = 128 # The authorized overlap between two part of the context when splitting it is needed.
pad_on_right = tokenizer.padding_side == "right"

print('Loading validation data...')
tiny_val_data = load_dataset('json', data_files='./TriviaQuestion2NQ_Transform_Dataset/NaturalQuestions_dev_reformatted.json', split='train[:]')
print('Loaded validation data!')

print('Loading Training data...')
# Data <Please change the dataset names appropriately>
#tiny_train_data = load_dataset('json', data_files='/fs/clip-quiz/saptab1/QA-MT-NLG/Datasets/nq_and_nqlike_processed.json', split='train[:]')

tiny_nq_train_data = load_dataset('json', data_files='./TriviaQuestion2NQ_Transform_Dataset/NaturalQuestions_train_reformatted.json', split='train[:]')

tiny_qb_train_data = load_dataset('json', data_files=qb_train_path, split='train[:]')

print('Loaded Training Data!')
# 2 different variables, so does not matter if train split is used

print('Dataset Loaded!')

def prepare_train_features(examples):
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation f the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    examples["question"] = [q.lstrip() for q in examples["question"]]

    # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length", 
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        #answers = examples["answers"][sample_index]
        answers = examples["answer"][sample_index]
        #print(examples['char_spans'])
        spans = examples['char_spans'][sample_index]
        #if i == 0:
            #print(spans)
        #detected_answers = examples['detected_answers'][sample_index]
        # If no answers are given, set the cls_index as answer.
        if spans[0][0] == 0 and spans[0][1] == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            # start_char = detected_answers["char_spans"][0]['start'][0]
            # end_char = detected_answers["char_spans"][0]['end'][0]
            start_char = spans[0][0]
            end_char = spans[0][1]
            #start_char = answers["answer_start"][0]
            #end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

        # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1
            #print(token_end_index)
            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while token_end_index > 0 and offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    #print(len(tokenized_examples['input_ids']))
    #print(len(tokenized_examples['attention_mask']))

    #print(tokenized_examples['attention_mask']])
    #return_dict = {'input_ids':tokenized_examples['input_ids'],
     #              'start_positions':tokenized_examples['start_positions'],
     #              'end_positions':tokenized_examples['end_positions']}
    return tokenized_examples	

print('Tokenizing Train Data...')

tiny_nq_train_tokenized_datasets = tiny_nq_train_data.map(prepare_train_features, batched=True, batch_size=64,
                                    remove_columns=['context', 'char_spans', 'answer',
                                                    'question', 'qanta_id', 'score'])

print('Tokenized NQ Train Data!')

tiny_qb_train_tokenized_datasets = tiny_qb_train_data.map(prepare_train_features, batched=True, batch_size=64,
                                    remove_columns=['context', 'char_spans', 'answer',
                                                    'question', 'qanta_id', 'quality_score'])

print('Tokenized QB Train Data!')

print('Tokenizing Val Data...')
tiny_val_tokenized_datasets = tiny_val_data.map(prepare_train_features, batched=True, batch_size=64,
                                    remove_columns=['context', 'char_spans', 'answer',
                                                    'question', 'qanta_id', 'quality_score'])

print('Tokenized Val Data!')

train_batch_size = 6
eval_batch_size = 8

nq_trainloader = torch.utils.data.DataLoader(tiny_nq_train_tokenized_datasets, batch_size=train_batch_size, collate_fn = default_data_collator, shuffle=False)

qb_trainloader = torch.utils.data.DataLoader(tiny_qb_train_tokenized_datasets, batch_size=train_batch_size, collate_fn = default_data_collator, shuffle=False)

valloader = torch.utils.data.DataLoader(tiny_val_tokenized_datasets, batch_size=eval_batch_size, collate_fn = default_data_collator, shuffle=False)

#Training Step Function to handle Loss Comp and Grad Calc
def training_step(model, opt, sch,  batch):
    #print('KEYS',batch.keys())
    #print(batch)
    for k,v in batch.items():
    #  print(type(v[0]))
       batch[k] = v.cuda()
    
    outputs = model(**batch)
    loss,logits = outputs[:2] 
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    opt.step()
    sch.step()
    opt.zero_grad()
    return loss.item()

# Evaluation Step Function to handle Loss Calc for Validation Dataset
def evaluation_step(model, valloader):
    pred = []
    gold =[]
    total_loss = 0
    model.eval()
    
    for batch in tqdm(valloader, total=len(valloader)):
      #gold += batch['labels'].numpy().flatten().tolist()
      for k,v in batch.items():
         batch[k] = v.cuda()
      #batch = {k:v.type(torch.long).to(device) for k,v in batch.items()}
      with torch.no_grad():
        outputs = model(**batch)
        loss, logits = outputs[:2]
        logits = logits.detach().cpu().numpy()
        total_loss += loss.item()
        predict_content = logits.argmax(axis=-1).flatten().tolist()
        #pred += predict_content

    avg_epoch_loss = total_loss / len(valloader)
  
    return pred, gold, avg_epoch_loss

# Save the Model State Dict and Optimizer State Dict at meaningful ckpt path
def save_checkpoint(model, opt, tokenizer, train_loss, val_loss, id, ep, data_type):
   ckpt_path = checkpoint_dir + '/epoch_'+str(ep)+'_step_'+str(id)+'_'+data_type
   #save_dict = {'model_state_dict':model.state_dict(), 'opt_state_dict':opt.state_dict(), 'train_loss':train_loss, 'val_loss':val_loss.item(), 'step':id, 'epoch':ep}
   #torch.save(save_dict, ckpt_path)
   
   os.makedirs(ckpt_path, exist_ok = True)
   print('Saving Model Checkpoint to ', ckpt_path)
   model.save_pretrained(ckpt_path)
   tokenizer.save_pretrained(ckpt_path)

   return 

def train(from_pretrained_NQ):
    # model
    model = AutoModelForQuestionAnswering.from_pretrained('bert-base-cased')
    model = model.to(device)
    model_name = 'bert-base-cased'
    checkpoint_dir = "./NQ_QB_train_seq_checkpoints"
    
    if not os.path.exists(checkpoint_dir):
      print('Creating Checkpoint Directory...')
      os.makedirs(checkpoint_dir)
    
    # optimizer
    nq_learning_rate=2e-5
    qb_learning_rate=2e-5
    weight_decay=0.01
    nq_checkpoint_periodicity = 2000
    qb_checkpoint_periodicity = 500
    EPOCHS = 1
    
    opt1 = AdamW(model.parameters(), lr=nq_learning_rate, weight_decay=0.01)
    opt2 = AdamW(model.parameters(), lr=qb_learning_rate, weight_decay=0.01)
    
    sch1 = get_linear_schedule_with_warmup(opt1, num_warmup_steps = 0, num_training_steps = len(nq_trainloader))
    sch2 = get_linear_schedule_with_warmup(opt2, num_warmup_steps = 0, num_training_steps = len(qb_trainloader))
    # train
    all_loss = {'train_loss':[], 'val_loss':[]}
    all_acc = {'train_acc':[], 'val_acc':[]}
    
    print('Initializing Training...')
    for epoch in range(1, EPOCHS+1):
       # Training Code
       # Step 1: NQ Dataloader
       if from_pretrained_NQ == False:
           for i,batch in enumerate(nq_trainloader):
              batch_loss = training_step(model, opt1, sch1, batch)
              if (i+1)%nq_checkpoint_periodicity == 0:
                 true_labels, prediction_labels, val_loss = evaluation_step(model, valloader)
                 print('NQorig Epoch ',epoch,' Batch ',i+1,' Train Loss = ',batch_loss,' Val Loss = ',val_loss)
                 save_checkpoint(model, opt1, tokenizer, batch_loss, val_loss, i+1, epoch, 'NQorig')
           true_labels, prediction_labels, val_epoch_loss = evaluation_step(model, valloader)
           save_checkpoint(model, opt1, tokenizer, batch_loss, val_epoch_loss, i+1, epoch, 'NQorig')
           print('NQ Training Finished for epoch = ',epoch)
       else:
            model = AutoModelForQuestionAnswering.from_pretrained('./TriviaQuestion2NQ_Transform_Dataset/epoch_1_step_26802_NQorig/')
            model = model.to(device)
       
       # Step 2: NQ-like Dataloader
       # for loop over NQ-like Dataloader with loss cacl and grad updates
       for i,batch in enumerate(qb_trainloader):
          batch_loss = training_step(model, opt2, sch2, batch)
          if (i+1)%qb_checkpoint_periodicity == 0:
             true_labels, prediction_labels, val_loss = evaluation_step(model, valloader)
             print('QB Epoch ',epoch,' Batch ',i+1,' Train Loss = ',batch_loss,' Val Loss = ',val_loss)
             save_checkpoint(model, opt2, tokenizer, batch_loss, val_loss, i+1, epoch, 'QB')
    
       true_labels, prediction_labels, val_epoch_loss = evaluation_step(model, valloader)
       save_checkpoint(model, opt2, tokenizer, batch_loss, val_epoch_loss, i+1, epoch, 'QB')
       print('NQ Like Training Finished for epoch = ', epoch)
    
    print('Training Finished Successfully!')

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Train QA system for NQ+QB")
    parser.add_argument('--qb_train_path', type=str, default='./qb_with_contexts_with_quality_scores.json')
    parser.add_argument('--from_pretrained_NQ', type=bool, default=False, help='Set it True if you want to use pretrained NQ checkpoints.')
    args = parser.parse_args()
    
    train(args.from_pretrained_NQ)
