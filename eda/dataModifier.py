
import pandas as pd 
import numpy as np 

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy import sparse

import re


# Define Regex Rules for Dates, etc
rx = re.compile(r"(DATE|TIME)", re.IGNORECASE)

class DataModifier:

    def __init__(self, df):
        self.df = df
        self.date_list = [c for c in df.columns if rx.search(c)]

        # Values That Are to be set by user
        self.to_drop = set()
        self.hot_one_encoing_ls = []

    # Main Function that would hot-one encode, drop_cols, etc 
    def clean(self):
        if self.to_drop:
            print("Dropped columns:")
            for col in self.to_drop:
                print(f" - {col}")
            self.df = self.df.drop(columns=self.to_drop, errors="ignore")
        return self.df


    def hot_one_encode(self):
        pass

    def setHotOneEncodingList(self):
        pass

    def set_data_cols(self):
        pass

    def set_dropped_names(self, names_to_drop):
        drop = []
        names_to_drop = set(names_to_drop)
        for c in self.df.columns:
            low = c.lower()
            if any(name in low for name in names_to_drop):
                drop.append(c)
            if "email" in low:
                drop.append(c)
            if low == "name":
                drop.append(c)
        self.to_drop = sorted(set(drop))