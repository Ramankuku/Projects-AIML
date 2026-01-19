import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_ingestion import DataIngestion
from src.data_transformation import DataTransformation
from src.model_building import ModelBuilding

data_fetch = DataIngestion()
data_transform = DataTransformation()
model_imp = ModelBuilding()
train_df, test_df = data_fetch.data_ingestion()
train_data, test_data = data_transform.drop_unwanted_columns(train_df, test_df)

train_data, test_data = data_transform.missing_values(train_data, test_data)
train_data, test_data = data_transform.encode_data(train_data, test_data)

classifier_model = model_imp.model_build(train_data, test_data)

