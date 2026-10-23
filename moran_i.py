import numpy as np 
import pandas as pd 
import libpysal
from esda.moran import Moran


mentions = pd.read_csv("vladimir_data.csv", sep=';')
mentions = mentions[mentions['tag'].isin([ 'culture', 'third_place'])]
coords = mentions[['coord1', 'coord2']].to_numpy(dtype=float)
freqs  = mentions['frequency'].to_numpy(dtype=float)

mentions_filtered = mentions[mentions['frequency']>1].copy()
coords_filtered = mentions_filtered[['coord1', 'coord2']].to_numpy(dtype=float)
freqs_filtered  = mentions_filtered['frequency'].to_numpy(dtype=float)


w = libpysal.weights.KNN.from_array(coords, k=5)
w.transform = 'r'
np.random.seed(12345)
m = Moran(freqs, w, permutations=999)


w = libpysal.weights.KNN.from_array(coords_filtered, k=5)
w.transform = 'B'
np.random.seed(12345)
m1 = Moran(freqs_filtered, w, permutations=999)

print(f"Для всех мест: I: {m.I}, p: {m.p_sim}")
print(f"Для мест с >1 упоминанием: I: {m1.I}, p: {m1.p_sim}")