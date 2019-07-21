from __future__ import absolute_import
from __future__ import print_function

import pandas as pd



def csv_to_df(path, header=0,index_col=0):
    return pd.read_csv(path, header=header, index_col=index_col)

def read_csv(path, header=0, index_col=None):
    return pd.read_csv(path, header=header, index_col=index_col)

def IsValid(x):
    return str.isdigit(x)
