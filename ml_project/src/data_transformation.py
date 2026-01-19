from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from config.utils import DROP_COLUMNS, ORDINAL_ENCODE, NUM_MISSING_VALUES, CAT_MISSING_VALUES

class DataTransformation:
    def __init__(self):
        self.impute_num = SimpleImputer(strategy='median')
        self.impute_cat = SimpleImputer(strategy='most_frequent')
        self.ordinal_en = OrdinalEncoder()
        self.drop_columns = DROP_COLUMNS
        self.ordinal_encode = ORDINAL_ENCODE
        self.num_missing_values = NUM_MISSING_VALUES
        self.cat_missing_values = CAT_MISSING_VALUES

    # Drop unwanted columns
    def drop_unwanted_columns(self, train_data, test_data):
        try:
            train_data = train_data.drop(columns=self.drop_columns, errors='ignore')
            test_data = test_data.drop(columns=self.drop_columns, errors='ignore')

            print(f"Train Data: {train_data.shape}")
            print(f"Test Data: {test_data.shape}")
            return train_data, test_data
        except Exception as e:
            print('Error while dropping columns:', e)
            return train_data, test_data

    # Fill missing values
    def missing_values(self, train_data, test_data):
        try:
            # Numeric columns
            if self.num_missing_values:
                train_data[self.num_missing_values] = self.impute_num.fit_transform(train_data[self.num_missing_values])
                test_data[self.num_missing_values] = self.impute_num.transform(test_data[self.num_missing_values])
            
            # Categorical columns
            if self.cat_missing_values:
                train_data[self.cat_missing_values] = self.impute_cat.fit_transform(train_data[self.cat_missing_values])
                test_data[self.cat_missing_values] = self.impute_cat.transform(test_data[self.cat_missing_values])
            
            print(f"Train Data: {train_data.shape}")
            print(f"Test Data: {test_data.shape}")
            return train_data, test_data
        except Exception as e:
            print('Error while filling null values:', e)
            return train_data, test_data

    # Encode categorical columns
    def encode_data(self, train_data, test_data):
        try:
            if self.ordinal_encode:
                train_data[self.ordinal_encode] = self.ordinal_en.fit_transform(train_data[self.ordinal_encode])
                test_data[self.ordinal_encode] = self.ordinal_en.transform(test_data[self.ordinal_encode])
            
            print(f"Train Data: {train_data.shape}")
            print(f"Test Data: {test_data.shape}")
            return train_data, test_data
        except Exception as e:
            print('Error while encoding data:', e)
            return train_data, test_data
