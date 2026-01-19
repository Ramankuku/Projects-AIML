import os
import pandas as pd
from sklearn.model_selection import train_test_split
from config.utils import DATA_DIR, ARTIFACT_DIR, TRAIN_PATH, TEST_PATH, TEST_RATIO_SPLIT

#
class DataIngestion:
    def __init__(self):
        self.data_path = DATA_DIR
        self.artiact_path = ARTIFACT_DIR
        self.train_path = TRAIN_PATH
        self.test_path = TEST_PATH
        self.train_split_ratio = TEST_RATIO_SPLIT

        # To ensure the artifacts dir present
        os.makedirs(self.artiact_path, exist_ok=True)

    def data_ingestion(self):
        try:
            df = pd.read_csv(self.data_path)
            print(f"Data loaded --- {df.shape}\n")

            train_df, test_df = train_test_split(df, test_size=self.train_split_ratio, random_state=42)
            print(f"Train_df shape: {train_df.shape}\n")
            print(f"Test_df shape: {test_df.shape}\n")

            train_df.to_csv(self.train_path, index=False)
            test_df.to_csv(self.test_path, index=False)

            print(f"train_df saved to {self.train_path}\n")
            print(f"test_df saved to {self.test_path}")

            return train_df, test_df

        except Exception as e:
            print("Getting error while perfoming Data Ingestion part: ",e ) 
        
