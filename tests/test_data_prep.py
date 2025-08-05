
import unittest
import pandas as pd
from src import data_prep

class TestDataPrep(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            'customerID': ['0001', '0002'],
            'TotalCharges': ['100.5', ''],
            'Churn': ['Yes', 'No'],
            'gender': ['Male', 'Female']
        })

    def test_clean_data(self):
        cleaned = data_prep.clean_data(self.df)
        self.assertFalse('customerID' in cleaned.columns)
        self.assertFalse(cleaned['TotalCharges'].isnull().any())

    def test_encode_categoricals(self):
        cleaned = data_prep.clean_data(self.df)
        encoded = data_prep.encode_categoricals(cleaned)
        self.assertIsInstance(encoded, pd.DataFrame)
        self.assertFalse(encoded.select_dtypes(include='object').any().any())

if __name__ == '__main__':
    unittest.main()
