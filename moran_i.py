import numpy as np 
import pandas as pd 
import libpysal
from esda.moran import Moran


mentions = pd.read_csv("vladimir_data.csv", sep=';')
mentions = mentions[mentions['tag'].isin([ 'culture', 'third_place'])]
mentions_filtered = mentions[mentions['frequency']>1].copy().reset_index(drop=True)
coords_filtered = mentions_filtered[['coord1', 'coord2']].to_numpy(dtype=float)
freqs_filtered  = mentions_filtered['frequency'].to_numpy(dtype=float)

table=[]

for k_i in range(3,7):
    w = libpysal.weights.KNN.from_array(coords_filtered, k=k_i)
    w.transform = 'r'
    np.random.seed(12345)
    m = Moran(freqs_filtered, w, permutations=999)
    table.append({"k":k_i,"I":m.I})

pd.DataFrame(table).to_csv("moran_by_k.csv")