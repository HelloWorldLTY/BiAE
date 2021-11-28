import scprep
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import phate
import graphtools as gt
import magic
import os
import datetime
import scanpy as sc
import sklearn.preprocessing as preprocessing
import loompy as lp
import umap.umap_ as umap
from sklearn.utils import shuffle
from scipy.stats.mstats import gmean

# Preprocessing protein data (ADT)
def clr_rate(protein):
      pro = protein+0.001
  g_mean = gmean(pro, axis=1)
  clr_protein = np.log(np.array([i/j for i,j in zip(pro,g_mean)]))
  return clr_protein

def KNN_Matching(data1, data2, label_list):
    celltype1 = label_list[0]
    celltype2 = label_list[1]

    id_list1 = [i for i in range(len(data1))]
    id_list2 = [i for i in range(len(data2))]

    result_pair = []

    while id_list2 != []:
        item = id_list2[0]
        temp = [i for i in range(len(id_list1)) if celltype1[i]==celltype2[item]]
        k = np.random.choice(temp)
        result_pair.append((k, item))
        id_list2.remove(item)

    return [result_pair,id_list1]

class Mish(nn.Module):
      def __init__(self):
    super().__init__()

  def forward(self,x):
    return x*torch.tanh(F.softplus(x))



# NN model
# use supervisied learning method
# target: transfer rna data into protein data

class generator_r2p(nn.Module):
    def __init__(self):
        super(generator_r2p, self).__init__()
        self.relu_l = nn.ReLU(True)
        self.gen = nn.Sequential(

            nn.Linear(2000, 1024),  
            nn.BatchNorm1d(1024),
            Mish(),

            nn.Linear(1024, 512),  
            nn.BatchNorm1d(512),
            Mish(),

            nn.Linear(512, 14)
           
        )

        self.lin = nn.Linear(2000, 14)


    def forward(self, x):
        ge = self.gen(x)
        
        return ge



class generator_p2r(nn.Module):
    def __init__(self):
        super(generator_p2r, self).__init__()
        self.relu_l = nn.ReLU(True)
        self.gen = nn.Sequential(
            nn.Linear(14,128),  
            nn.BatchNorm1d(128),
            Mish(),

            nn.Linear(128, 256),  
            nn.BatchNorm1d(256),
            Mish(),

            nn.Linear(256, 512),  
            nn.BatchNorm1d(512),
            Mish(),

            nn.Linear(512, 1024),  
            nn.BatchNorm1d(1024),
            Mish(),

            nn.Linear(1024, 2000),  
           
        )

        self.lin = nn.Linear(14,2000)

    def forward(self, x):
        x = self.relu_l(self.gen(x) + self.lin(x))
        return x



