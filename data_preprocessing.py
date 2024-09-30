# data_preprocessing.py
import pandas as pd
import logzero
from logzero import logger

class DataPreprocessor:
    def __init__(self, file_path):
        logger.info("Initializing DataPreprocessor with file: %s", file_path)
        self.df = pd.read_csv(file_path)
        logger.info("Original DataFrame loaded")
        print("Original DataFrame:")
        print(self.df.head())
        print(self.df.info())

        # Handle missing values by dropping rows with NaN values
        self.df = self.df.dropna()
        logger.info("Missing values handled by dropping rows")

    def is_mixed_numeric(self, series):
        """
        Check if a column contains both integers and floats.
        """
        return series.apply(lambda x: isinstance(x, (int, float))).all() and series.apply(lambda x: isinstance(x, float)).any()

    def convert(self):
        """
        Process each column in the dataframe to ensure consistent data types.
        """
        logger.info("Starting conversion of data types")
        for col in self.df.columns:
            if self.df[col].dtype == 'int64' or self.df[col].dtype == 'float64':
                continue
            elif self.is_mixed_numeric(self.df[col]):
                self.df[col] = self.df[col].astype(float)
                logger.info("Converted mixed numeric column %s to float", col)
            elif self.df[col].dtype == 'object':
                self.df[col] = self.df[col].astype('category')
                self.df[col] = self.df[col].cat.codes
                logger.info("Converted object column %s to categorical", col)
            else:
                self.df[col] = self.df[col].astype(float)
                logger.info("Converted column %s to float", col)

        self.df.columns = self.df.columns.str.strip()
        self.df.columns = self.df.columns.str.lower()

        # Display the updated DataFrame to verify changes
        print("\nUpdated DataFrame:")
        print(self.df.head())
        print(self.df.info())
        logger.info("DataFrame updated with consistent data types")
