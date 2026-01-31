import pandas as pd
import pygwalker as pyg

df = pd.read_csv("data/datamahasiswa_clean.csv")
pyg.walk(df)
