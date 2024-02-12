# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 09:24:48 2023

@author: sai
"""

import pandas as pd 
import numpy as np

df=pd.read_csv("C:/2-dataset/modified ethnic.csv")

#check none value
df.isna().sum()

from sklearn.impute import SimpleImputer

# Create a SimpleImputer object to replace missing values with the mean
mean_imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

# Impute missing values in the 'Salaries' column with the mean using SimpleImputer
df["Salaries"] = pd.DataFrame(mean_imputer.fit_transform(df[["Salaries"]]))

# Check again the number of missing values in the 'Salaries' column after imputation
df["Salaries"].isna().sum()
