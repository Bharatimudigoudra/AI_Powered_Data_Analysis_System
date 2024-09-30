import pandas as pd
import json
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import logzero
from logzero import logger

class DataInsightGenerator:
    def __init__(self, model_name, api_key):
        self.model = ChatGroq(model_name=model_name, api_key=api_key)
        logger.info("Initialized DataInsightGenerator with model %s", model_name)

    def generate_insights(self, df):
        """
        Generate insights for each column of the dataset using ChatGroq.
        """
        logger.info("Generating insights for dataset")
        sample_data = df.head().to_dict(orient='records')
        sample_data_json = json.dumps(sample_data, indent=2)

        prompt_text = (
            f"Analyze the following dataset and provide insights for each column. "
            f"Describe the contents, what each column tells about the data, and what kind of features can be extracted from each column. "
            f"Dataset:\n\n{sample_data_json}"
        )
        
        messages = [HumanMessage(content=prompt_text)]

        try:
            response = self.model.invoke(messages)
            logger.info("Insights generated successfully")
            if response and hasattr(response, 'content'):
                return response.content
            else:
                logger.error("Response content is not accessible")
                return "An error occurred: Response content is not accessible."
        except Exception as e:
            logger.error("An error occurred while generating insights: %s", e)
            return f"An error occurred: {e}"

    def process_dataframe(self, df):
        """
        Process the DataFrame and generate insights.
        """
        logger.info("Processing DataFrame for insights")
        logger.info("Dataset Overview:")
        logger.info("\n%s", df.head())
        logger.info("%s", df.info())

        insights = self.generate_insights(df)
        return insights

if __name__ == "__main__":
    generator = DataInsightGenerator(
        model_name="llama-3.1-70b-versatile",
        api_key="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"
    )
    # Example of how to call this in app.py instead of this main section
