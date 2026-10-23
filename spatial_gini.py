import numpy as np 
import pandas as pd 
import libpysal
from inequality.gini import Gini_Spatial
from esda.moran import Moran


mentions = pd.read_csv("vladimir_data.csv", sep=';')
coords = mentions[['coord1', 'coord2']].to_numpy(dtype=float)
freqs  = mentions['frequency'].to_numpy(dtype=float)
w = libpysal.weights.KNN.from_array(coords, k=5)
np.random.seed(12345)
gs = Gini_Spatial(freqs, w, permutations=999)
print(f"Пространственный коэфициент Джини: {gs.wcg_share}")