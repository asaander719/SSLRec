'''
This code is utilized to created the kg.txt file.
'''

import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import csr_matrix

predir = ''

trn_pos = pickle.load(open(predir+'train_mat_pos.pkl', 'rb'))
trn_neutral = pickle.load(open(predir+'train_mat_neutral.pkl', 'rb'))
trn_neg = pickle.load(open(predir+'train_mat_neg.pkl', 'rb'))

# construct kg.txt
trn_neg = 1 * (trn_neg != 0)
trn_neutral = 1 * (trn_neutral != 0)
trn_pos = 1 * (trn_pos != 0)

ii_neg = trn_neg.T * trn_neg
ii_neutral = trn_neutral.T * trn_neutral
ii_pos = trn_pos.T * trn_pos

ii_neg = 1 * (ii_neg > 3)
ii_neutral = 1 * (ii_neutral > 3)
ii_pos = 1 * (ii_pos > 3)

neg_data = np.zeros(len(ii_neg.data))
neutral_data = np.ones(len(ii_neutral.data))
pos_data = np.full(len(ii_pos.data), 2)

neg_x, neg_y = ii_neg.nonzero()
neutral_x, neutral_y = ii_neutral.nonzero()
pos_x, pos_y = ii_pos.nonzero()

neg_kg = np.stack((neg_x, neg_data, neg_y))
neutral_kg = np.stack((neutral_x, neutral_data, neutral_y))
pos_kg = np.stack((pos_x, pos_data, pos_y))

neg_kg = neg_kg.T
neutral_kg = neutral_kg.T
pos_kg = pos_kg.T

print(neg_kg.shape)
print(neutral_kg.shape)
print(pos_kg.shape)

kg = np.vstack((neg_kg, neutral_kg, pos_kg)).astype(int) 
kg_df = pd.DataFrame(kg)
kg_df.to_csv(predir+'kg.txt', sep=' ', header=None, index=None)





