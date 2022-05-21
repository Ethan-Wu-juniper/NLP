#Author: 吳原博
#Student ID: 0816004
#HW ID: hw3
#Due Date: 05/26/2022
from collections import Counter, namedtuple, defaultdict
from nltk.tokenize import RegexpTokenizer
import nltk
import json
import os
import re
import random
import pandas as pd
from nltk.corpus import gutenberg, brown, reuters, inaugural, cmudict
from gensim.corpora import WikiCorpus

FORTEST = True
corpus_file = ['gutenberg', 'brown', 'reuters', 'inaugural', 'cmudict']
Corpus = [gutenberg, brown, reuters, inaugural]
OptCode = {"A" : 0, "B" : 1, "C" : 2, "D" : 3}
AnsCode = {0 : "A", 1 : "B", 2 : "C", 3 : "D"}
TrainPath = "data/train"
TestPath = "data/test"
SOS = "<s> "
EOS = " </s>"
UNK = "<UNK>"

def wiki_corpus():
    file = 'enwiki-20220501-pages-articles-multistream1.xml-p1p41242.bz2'
    wiki = WikiCorpus(file, dictionary={})
    f = open('wiki_small.txt', encoding='utf8', mode='w')
    i = 0
    for text in wiki.get_texts():
        str_line = ' '.join(text)
        os.system('clear')
        i += 1
        print(i)
        f.write(str_line + '\n')

def load_wiki():
    wiki_article = []
    with open('wiki_small.txt') as f:
        sents = f.readlines()
        for sent in sents:
            wiki_article.append(sent)
    return wiki_article

def test_corpus():
    for f,c in zip(corpus_file,Corpus):
        try:
            c.fileids()
        except:
            nltk.download(f)

def load_corpus():
    test_corpus()
    content = []
    for corpus in Corpus:
        for f in corpus.fileids():
            text = corpus.raw(f).lower()
            text = re.sub('[^\w\s]',' ',text)
            content.append(text)
    return content

def load_train_data():
    train_text = []
    for filename in os.listdir(TrainPath):
        with open(os.path.join(TrainPath,filename),'r') as f:
            data = json.load(f)
            text = data['article'].lower()
            underline = re.findall(" _ ",text)
            numberline = re.findall("\s\d+ _ ",text)
            ''' clear redundant numbers that occur just right before the underline '''
            if  underline and len(numberline) / len(underline) > 0.7:
                text = re.sub("\s\d+ _ ","  _ ",text)
            ''' replace underline with the  nswer and remove punctuations '''
            for key, val in data['answers'].items():
                ans = data['options'][key][OptCode[val]]
                text = text.replace(" _ ",f"{ans}",1)
                text = re.sub('[^\w\s]',' ',text)
            train_text.append(text)
        # if FORTEST:
        #     break
    return train_text

def preprocess(text, n):
    ''' plain text without punctuations will be preprocessed and tokenized here '''
    sos = SOS * (n-1) if n > 1 else SOS
    text = [sos + t + EOS for t in text]
    text = ' '.join(text)
    tokenizer = RegexpTokenizer(r'\S+')
    return tokenizer.tokenize(text)

def load_test_data():
    questions = []
    for filename in os.listdir(TestPath):
        with open(os.path.join(TestPath,filename),'r') as f:
            m_gram = []
            data = json.load(f)
            text = data['article'].lower()
            underline = re.findall(" _ ",text)
            numberline = re.findall("\s\d+ _ ",text)
            ''' clear redundant numbers that occur just right before the underline '''
            if  underline and len(numberline) / len(underline) > 0.7:
                text = re.sub("\s\d+ _ ","  _ ",text)
            text = re.sub('[^\w\s]',' ',text)
            text = re.sub('\s+',' ',text)
            sos = SOS * 4
            eos = EOS * 4
            text = sos + text + eos
            m_gram = list(nltk.ngrams(text.split(" "),5))
            m_gram = [t for t in m_gram if t[2] == '_']
            questions.append({"m_gram" : m_gram, "options" : data['options'], "source" : data['source'] })
            if len(m_gram) != len(data['options']):
                print(filename)
                # print(text)
                # print(m_gram)
            # if FORTEST:
            #     break
    return questions


class LanguageModel():
    def __init__(self, train_text, laplace=1):
        self.tokens = preprocess(train_text, 3)
        self.laplace = laplace
        self.gram2 = self.model(2)
        self.gram3 = self.model(3)
        self.outage = []
        self.cnt = 0

    def model(self, n):
        vocab = nltk.FreqDist(self.tokens)
        if n == 1:
            return { (token,): count / len(self.tokens) for token, count in vocab.items() }
        
        n_grams = nltk.ngrams(self.tokens, n)
        n_vocab = nltk.FreqDist(n_grams)
        m_grams = nltk.ngrams(self.tokens, n-1)
        m_vocab = nltk.FreqDist(m_grams)

        def smooth(n_gram, n_count):
            m_gram = n_gram[:-1]
            m_count = m_vocab[m_gram]
            return (n_count + self.laplace) / (m_count + self.laplace * len(vocab))
        
        return { n_gram: smooth(n_gram, count) for n_gram, count in n_vocab.items() }

    def choose_opt(self, gram, option, qid):
        opt_prob = {}
        for idx, opt in enumerate(option):
            p31 = self.gram3[gram[:2] + tuple([opt])] if gram[:2] + tuple([opt]) in self.gram3 else 0
            p32 = self.gram3[tuple([gram[1], opt, gram[3]])] if tuple([gram[1], opt, gram[3]]) in self.gram3 else 0
            p21 = self.gram2[tuple([gram[1], opt])] if tuple([gram[1], opt]) in self.gram2 else 0
            p22 = self.gram2[tuple([opt, gram[3]])] if tuple([opt, gram[3]]) in self.gram2 else 0
            opt_prob[idx] = [p32*p31, p32, p31, p21*p22, p21, p22]

        find = False
        choose = ""
        for i in range(6):
            Max = 0
            for idx in opt_prob:
                if opt_prob[idx][i] > Max:
                    Max = opt_prob[idx][i]
                    choose = AnsCode[idx]
                    find = True
            if find:
                break
        if not find:
            self.outage.append(qid)
            choose = "C"
        return choose
            # print(tuple([gram[1], opt, gram[3]]), p32)


    def get_ans(self, questions):
        ans = {}
        for q in questions:
            for i, gram in enumerate(q['m_gram']):
                qid = f"{q['source']}_{i}"
                ans[qid] = self.choose_opt(gram, q['options'][qid], qid)
        return ans

train_text = load_train_data()
train_text += load_corpus()
train_text += load_wiki()
questions = load_test_data()
lm = LanguageModel(train_text)
ans = lm.get_ans(questions)
pd.DataFrame(list(ans.items()),columns=['id','label'],index=None).to_csv('out_3gram.csv',index=False)
''' Test if there is any word that is not in the LM '''
# with open("outage.txt",'w') as f:
#     for qid in lm.outage:
#         f.write(qid+'\n')
''' For Report 7. '''
# thisis = [0,0]
# hesaid = [0,0]
# shesaid = [0,0]
# for g in lm.gram3:
#     if g[:2] == ('this','is') and lm.gram3[g] > thisis[1]:
#         thisis = [g[2], lm.gram3[g]]
#     if g[:2] == ('he','said') and lm.gram3[g] > hesaid[1]:
#         hesaid = [g[2], lm.gram3[g]]
#     if g[:2] == ('she','said') and lm.gram3[g] > shesaid[1]:
#         shesaid = [g[2], lm.gram3[g]]

# print(thisis)
# print('-----------------')
# print(hesaid)
# print('-----------------')
# print(shesaid)