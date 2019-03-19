# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 23:14:36 2019

@author: raphaelfeijao and erikmedeiros
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import json

def first_view(df):
    print(df.head())
    print(df.info())
    
def k_most_common(df, k):
    A = list(df['comment_text'])
    A = str(A)
    Count = Counter( A.split())
    most_occur = Count.most_common(k) 
    most_occur = pd.DataFrame.from_dict(most_occur)
    print("The most commom words are :")
    print(most_occur) 
    
    print("The number of diferent words is:")
    print(len(Count.keys()))
    
def Plot_quantity (df, total = True):
    D = {}
    if total:
        D['total'] = df['toxic'].count()
    #Quantity
    D.update({'toxic': df['toxic'].sum(),
        'severe_toxic': df['severe_toxic'].sum(),
        'obscene':df['obscene'].sum(),
        'threat':df['threat'].sum(),
        'insult':df['insult'].sum(),
        'identity_hate':df['identity_hate'].sum()})

    plt.bar(range(len(D)), list(D.values()), color=(0.2, 0.4, 0.6, 0.6))
    plt.xticks(range(len(D)), list(D.keys()))
    plt.show()
    
def plot_hist(size):
    size = [v for v in size if v <= 1000]
    plt.hist(size,bins=30)
    plt.xlabel('words length')
    plt.ylabel('quantity')
    plt.title('Data histogram')
    plt.show()
    
def Size(df, minimal_len = 100):
    Size = np.array(list(map(len, df['comment_text'])))
    plot_hist(Size)    
    print("The mean size of the sentences is %.1f" %( Size.mean()))
    print ("For a minimal lenght of %d, the quantity of sentences with less words is %.2f%%" % (minimal_len, sum(Size<minimal_len )/df['toxic'].count()*100))

def gen_visualisation(words_list, att_vec, file=None):
    '''
    words_list: list of words of length n.
    att_vec: array of shape (n, 1)

    Example:
    words = ['test', 'test', 'test']
    x = [[0.3], [0.1], [0.6]]
    _ = gen_visualization(words, x, file='attn_vis-master/attn_vis_data.json')
    '''
    obj = {
        'article_lst': [],
        'p_gens': att_vec,
        'decoded_lst': words_list,
        'abstract_str': '',
        'attn_dists': [[]]
    }
    
    if not file is None:
        with open(file, 'w') as outfile:
            json.dump(obj, outfile)
        
    return json.dumps(obj)

