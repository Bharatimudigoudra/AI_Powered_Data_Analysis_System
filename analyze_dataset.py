import pandas as pd
from langchain_groq import ChatGroq
import logzero
from logzero import logger

class DatasetAnalyzer:
    def __init__(self, api_key, model_name="llama-3.1-70b-versatile"):
        self.api_key = api_key
        self.model_name = model_name
        self.model = ChatGroq(model_name=self.model_name, api_key=self.api_key)
        logger.info("Initialized DatasetAnalyzer with model %s", self.model_name)

    def analyze_dataset(self, file_path):
        logger.info("Starting dataset analysis for file: %s", file_path)
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            logger.info("CSV file loaded successfully")
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='ISO-8859-1')
            logger.info("CSV file loaded with ISO-8859-1 encoding")
        except Exception as e:
            logger.error("Error loading CSV file: %s", e)
            return "Error loading CSV file."

        summary = self.generate_summary(df)
        
        try:
            response = self.model.invoke(summary)
            logger.info("Model response received successfully")
            return response.content
        except Exception as e:
            logger.error("An error occurred during model invocation: %s", e)
            return "An error occurred during model invocation."

    def generate_summary(self, df):
        summary = []
        summary.append(f"Number of Rows: {df.shape[0]}")
        summary.append(f"Number of Columns: {df.shape[1]}")
        summary.append("\nData Types of Each Column:")
        summary.append(df.dtypes.to_string())
        summary.append("\nSummary Statistics of Numerical Columns:")
        summary.append(df.describe().to_string())
        summary.append("\nMissing Values in Each Column:")
        summary.append(df.isnull().sum().to_string())
        summary.append("\nUnique Values in Each Column:")
        for col in df.columns:
            summary.append(f"{col}: {df[col].nunique()} unique values")
        summary.append("\nDetailed Analysis of Each Feature:")
        for col in df.columns:
            summary.append(f"\nFeature: {col}")
            if df[col].dtype == 'object':
                summary.append("Feature Type: Categorical")
                summary.append(f"Number of Unique Categories: {df[col].nunique()}")
                summary.append(f"Most Frequent Category: {df[col].mode()[0]}")
                summary.append(f"Frequency of Most Frequent Category: {df[col].value_counts().iloc[0]}")
                summary.append(f"Top Categories: \n{df[col].value_counts().head()}")
            else:
                summary.append("Feature Type: Numerical")
                summary.append(f"Mean: {df[col].mean()}")
                summary.append(f"Median: {df[col].median()}")
                summary.append(f"Standard Deviation: {df[col].std()}")
                summary.append(f"Minimum Value: {df[col].min()}")
                summary.append(f"Maximum Value: {df[col].max()}")
                summary.append(f"25th Percentile: {df[col].quantile(0.25)}")
                summary.append(f"50th Percentile (Median): {df[col].quantile(0.5)}")
                summary.append(f"75th Percentile: {df[col].quantile(0.75)}")
            
            if df[col].dtype == 'object':
                unique_types = df[col].apply(type).nunique()
                if unique_types > 1:
                    summary.append("Note: This column contains mixed data types.")
        
        return "\n".join(summary)
