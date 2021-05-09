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


df['windspeed'] = np.where( ( (df['windspeed'] == 0) & (df['month'] == 1) ), 14.58, df['windspeed'])
df['windspeed'] = np.where( ( (df['windspeed'] == 0) & (df['month'] == 2) ), 13.96, df['windspeed'])
df['windspeed'] = np.where( ( (df['windspeed'] == 0) & (df['month'] == 3) ), 15.36, df['windspeed'])
df['windspeed'] = np.where( ( (df['windspeed'] == 0) & (df['month'] == 4) ), 15.58, df['windspeed'])
df['windspeed'] = np.where( ( (df['windspeed'] == 0) & (df['month'] == 5) ), 12.29, df['windspeed'])
df['windspeed'] = np.where( ( (df['windspeed'] == 0) & (df['month'] == 6) ), 12.34, df['windspeed'])
df['windspeed'] = np.where( ( (df['windspeed'] == 0) & (df['month'] == 7) ), 11.01, df['windspeed'])
df['windspeed'] = np.where( ( (df['windspeed'] == 0) & (df['month'] == 8) ), 11.93, df['windspeed'])
df['windspeed'] = np.where( ( (df['windspeed'] == 0) & (df['month'] == 9) ), 11.57, df['windspeed'])
df['windspeed'] = np.where( ( (df['windspeed'] == 0) & (df['month'] == 10) ), 11.22, df['windspeed'])
df['windspeed'] = np.where( ( (df['windspeed'] == 0) & (df['month'] == 11) ), 13.12, df['windspeed'])
df['windspeed'] = np.where( ( (df['windspeed'] == 0) & (df['month'] == 12) ), 10.68, df['windspeed'])


df = df.drop(5631)
df .weather = df.weather.cat.remove_unused_categories()

df = df[(z < 3)]


df["count"]= np.log1p(df["count"])

df.to_csv("velo_processed.csv")

