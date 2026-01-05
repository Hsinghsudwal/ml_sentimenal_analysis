import numpy as np
import pandas as pd
from utils.logger import LOG

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from src.helper_cleaner import data_cleaning


class DataPrep:
    def __init__(self):
        LOG.info("DataPrep initialized")

    def load_data(self, data):
        LOG.info("Load data initialized")

        column_names=['Tweet_ID','Entity','Sentiment','Tweet_content']
        data = pd.read_csv(data, sep=',',names=column_names)

        # Drop null and duplicate
        # data.dropna(inplace=True)
        # data.drop_duplicates(inplace=True)
        LOG.info(
            f"Data loaded | rows={data.shape[0]} cols={data.shape[1]}"
        )
        return data

    def data_preprocess(self, df):
        LOG.info("Data process initialized")
        df = df[df["Sentiment"] != "Irrelevant"].copy()
        LOG.info(
            f"Data after irrelevant drop | {df.shape}"
        )
        df = df.dropna(subset=["Tweet_content"])
        LOG.info(
            f"Data after dropna | {df.shape}"
        )


        df['processed_text'] = df['Tweet_content'].apply(data_cleaning)
        LOG.info(
            f"Data process | rows={df.shape[0]} cols={df.shape[1]}"
        )
        return df

        
    def split_encoder(self, df):
        LOG.info("Data split and encoder initialized")
        X = df['processed_text'].fillna("").astype(str) 
        y = df['Sentiment']

        LOG.info("Data train test split")
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
        # Ensure train/test are strings
        X_train = X_train.astype(str).reset_index(drop=True)
        X_test = X_test.astype(str).reset_index(drop=True)

        LOG.info("Data split for label encoder")
        label_encoder = LabelEncoder()
        y_train_enc = label_encoder.fit_transform(y_train)
        y_test_enc = label_encoder.transform(y_test)
        LOG.info("Complete data split and label encoder")

        return X_train, X_test, y_train, y_test, y_train_enc, y_test_enc, label_encoder