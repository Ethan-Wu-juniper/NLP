#Author: 吳原博
#Student ID: 0816004
#HW ID: hw1
#Due Date: 01/30/2020

from IPython.display import display
import pandas as pd
import numpy as np

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

class Edit:
    def __init__(self,input,target,cost_map):
        self.target = target
        self.input = input
        self.trace = [[[0,0] for j in range(len(self.target)+1)] for i in range(len(self.input)+1)]
        self.cost_map = cost_map
    def cost(self):
        dp = np.zeros((len(self.input)+1,len(self.target)+1), dtype=np.int8)
        for i in range(dp.shape[0]):
            dp[i][0] = i
            self.trace[i][0] = [-1,0] if i != 0 else [0,0]
        for j in range(dp.shape[1]):
            dp[0][j] = j
            self.trace[0][j] = [0,-1] if j != 0 else [0,0]
        for i in range(dp.shape[0]-1):
            for j in range(dp.shape[1]-1):
                prev_idx = [[-1,0],[0,-1],[-1,-1]]
                weight = self.cost_map[self.target[j]][ord(self.input[i])-97]
                sub = dp[i][j]+weight if self.input[i] != self.target[j] else dp[i][j]
                dp[i+1][j+1] = min(dp[i][j+1]+1,dp[i+1][j]+1,sub)
                temp_arr = np.array([dp[i][j+1]+1,dp[i+1][j]+1,sub])
                prev = np.random.choice(np.where(temp_arr==dp[i+1][j+1])[0])
                self.trace[i+1][j+1] = prev_idx[prev]
        # form = pd.DataFrame(self.trace,columns=[i for i in '#'+self.target],index=[i for i in '#'+self.input])
        # display(form)
        # print(dp)
        return dp[-1][-1]
    def backtrack(self):
        Itrace = []
        Ttrace = []
        next = self.trace[-1][-1]
        cur = [np.shape(self.trace)[0]-1,np.shape(self.trace)[1]-1]
        while next != [0,0]:
            Itrace.append(self.input[cur[0]-1] if next[0] != 0 else '*')
            Ttrace.append(self.target[cur[1]-1] if next[1] != 0 else '*')
            cur[0] += next[0]
            cur[1] += next[1]
            next = self.trace[cur[0]][cur[1]]
        Itrace.reverse()
        Ttrace.reverse()
        out = ['' for i in range(4)]
        for i in range(len(Itrace)):
            out[0] += Itrace[i]+' '
            out[1] += '| '
            out[2] += Ttrace[i]+' '
            if Itrace[i] == '*':
                out[3] += 'i '
            elif Ttrace[i] == '*':
                out[3] += 'd '
            elif Itrace[i] == Ttrace[i]:
                out[3] += 'n '
            else:
                out[3] += 's '
        for i in range(4):
            print(out[i])
        # return Itrace, Ttrace

weighted_map = pd.read_csv('costs2.csv')
two_map = pd.read_csv('costs1.csv')
# print(weighted_map['a'][ord('c')-97])

with open('input.txt','r') as file:
    for line in file:
        words = line.split()
        target = words[0]
        inputs = words[1:]
        for i in range(len(inputs)):
            edit = Edit(inputs[i],target,two_map)
            # edit = Edit('ab','c')
            cost = edit.cost()
            edit.backtrack()
            print('cost:',cost)
            print('')
            edit = Edit(inputs[i],target,weighted_map)
            # edit = Edit('ab','c')
            cost = edit.cost()
            edit.backtrack()
            print('cost:',cost)
            print('-------------------------------------------')
