from tqdm import tqdm
from nltk.tokenize import sent_tokenize
import openai
import nltk
import json
import os,sys
import time 
import spacy
nlp = spacy.load('en_core_web_sm')
#nltk.download('punkt')
#import backoff

openai.api_key = "APIKEY"

questions, answers, categorys = [],[],[]
with open('./qanta.advtest.2018.04.18.json') as json_file:
    questions_full = json.load(json_file)['questions']

example=[]
qid=[]
exs=[]
for i in questions_full:
    question=i['text']
    qid_i=i['qanta_id']
    answer=' '.join(i["answer"].split("_"))
    question_split=nltk.sent_tokenize(question)
    for j in range(len(question_split)):
        e1=tuple((qid_i,answer,question_split[j]))
        exs.append(e1)

print("number of questions to convert: ",len(exs))

for qid, q_answer,example in exs:
        nlp.max_length = len(example) + 100
        doc=nlp(example)
        delt=" "

        for i in doc.noun_chunks:
                if i.text.lower().find("this ")!=-1:
                        #delt=i.text.lower().replace("this ","which ")
                        delt=i.text.lower()
                        #print(delt)
                        break
                elif i.text.lower().find("these ")!=-1:
                        #delt=i.text.lower().replace("these ","which ")
                        delt=i.text.lower()
                        #print(delt)
                        break
        
        if delt==" ":
                continue
        else:
                prompt1 = example+ " Ask about "+delt+" in the question. Your question's correct answer should be "+q_answer+". Make sure the answer is not in the your question. Make the question as natural as a google search query."

        try:
                message = prompt1
                messages = [
                        {"role": "system", "content": "You are a helpful assistant."},]
                if message:
                        messages.append(
                                {"role": "user", "content": message},
                        )
                        chat_completion = openai.ChatCompletion.create(
                                model="gpt-3.5-turbo",
                                messages=messages
                        )
                
                answer = chat_completion.choices[0].message.content
                #print(f"prompt: {prompt1}")
                #print(f"ChatGPT: {answer}")
                
                dict_= {}
                dict_['chatgpt_prompt'] = prompt1
                dict_['question']= answer
                dict_['answer'] = q_answer

                with open('./evaluation_set_new.json','a') as f:
                        json.dump(dict_,f)
                time.sleep(20)

        except:
                time.sleep(20)
                pass
                
        time.sleep(10)
            #messages.append({"role": "assistant", "content": answer})







