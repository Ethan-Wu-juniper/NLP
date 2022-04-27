#Author: 吳原博
#Student ID: 0816004
#HW ID: hw2
#Due Date: 04/21/2022
import spacy
import pandas as pd
import numpy as np

pout = True
ans = []
data_path = "data_file/data.csv"
data = pd.read_csv(data_path)
nlp = spacy.load("en_core_web_sm")
s_pos = ['nsubj','nsubjpass']
v_pos = ['VERB','AUX']
o_pos = ['dobj','pobj','attr']

def sublist(sub,lst):
  for i in range(len(lst)):
    j = 0
    while sub[j] == lst[i+j]:
      j += 1
      if j >= len(sub):
        return True
      if i+j >= len(lst):
        break
  return False

def get_token_set(text,sent,doc):
  set_ = []
  idx = sent.find(text+' ')
  if idx < 0:
    return set_
  for token in doc:
    if token.idx >= idx and token.idx <= idx + len(text):
      set_.append(token)
  return set_

def CheckS(s_set,v_set):
  for s in s_set:
    if s.dep_ in s_pos and s.head in v_set:
      return True
  return False

def CheckO(o_set,v_set):
  for o in o_set:
    if o.dep_ in o_pos and o.head in v_set:
      return True
  return False

for i in range(len(data)):
# if True:
#   i = 2424
  # print(i)
  sent = data['sentence'][i]
  doc = nlp(sent)
  S = data['S'][i]
  V = data['V'][i]
  O = data['O'][i]
  s_set = []
  v_set = []
  o_set = []
  check_s = False
  check_o = False

  for token in doc:
    s_set = get_token_set(S,sent,doc)
    v_set = get_token_set(V,sent,doc)
    o_set = get_token_set(O,sent,doc)
  if s_set == [] or v_set == [] or o_set == []:
    ans.append([data['id'][i],0])
    # print("not in")
    continue
  for v in v_set:
    if v.pos_ in v_pos and sublist(v_set,list(v.subtree)):
      if sublist(o_set,list(v.subtree)) and CheckO(o_set,v_set):
        check_o = True
      if sublist(s_set,list(v.subtree)) and CheckS(s_set,v_set):
        check_s = True
      while v.dep_ == "conj":
        v = v.head
        v_set = [v]
      if sublist(s_set,list(v.subtree)) and CheckS(s_set,v_set):
        check_s = True
      # if v.dep_ == "relcl":
      #   sub = v.head
      #   if sublist(s_set,list(sub.subtree)):
      #     check_s = True
      break
  ans.append([data['id'][i],int(check_s & check_o)])

pd.DataFrame(ans,columns=['id','label'],index=None).to_csv('local_conj_subtree.csv',index=False)