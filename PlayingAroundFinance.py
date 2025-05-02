import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

# dati = pd.DataFrame()
# for nome in elenco:
#     print(nome, end=" - ")
#     df = pd.read_csv(urlo + elenco[nome], index_col=0)
#     df.index = pd.to_datetime(df.index)
#     df.rename(columns={df.columns[0]:nome}, inplace=True)
# #    df.dropna(inplace=True)
#     dati = pd.concat([dati,df], axis=1)
# dati.dropna(inplace=True)
# print(dati.head())