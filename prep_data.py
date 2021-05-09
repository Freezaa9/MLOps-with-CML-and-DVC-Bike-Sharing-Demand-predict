import pandas as pd 
import numpy as np
# Set random seed
seed = 42

################################
########## DATA PREP ###########
################################

# Load in the data
df = pd.read_csv("velo.csv")

# Prep

df["datetime"] = pd.to_datetime(df["datetime"])
df["season"] = pd.Categorical(df["season"], ordered=True)
df["holiday"] = pd.Categorical(df["holiday"], ordered=False)
df["workingday"] = pd.Categorical(df["workingday"], ordered=False)
df["weather"] = pd.Categorical(df["weather"], ordered=False)

df['dayofweek'] = df.datetime.dt.dayofweek
df['hour'] = df.datetime.dt.hour
df['month'] = df.datetime.dt.month
df['year'] = df.datetime.dt.year

df["dayofweek"] = pd.Categorical(df["dayofweek"], ordered=False)
df["hour"] = pd.Categorical(df["hour"], ordered=False)
df["month"] = pd.Categorical(df["month"], ordered=False)
df["year"] = pd.Categorical(df["year"], ordered=False)

df = df.drop(["casual", "registered", "datetime"], axis=1)

df.to_csv("velo_processed.csv")

