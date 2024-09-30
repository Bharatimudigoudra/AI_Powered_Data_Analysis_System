import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import os  # Import os to handle directory creation
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import logzero
from logzero import logger

# Use a non-GUI backend for Matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for rendering plots without a GUI

class DataAnalyzer:
    def __init__(self, api_key, model_name="llama-3.1-70b-versatile"):
        self.api_key = api_key
        self.model_name = model_name
        self.model = ChatGroq(model_name=self.model_name, api_key=self.api_key)
        logzero.logfile("llm_data_analyzer.log", maxBytes=1e6, backupCount=3)
        logger.info("Initialized DatasetAnalyzer with model %s", self.model_name)
        self.df = None  # Initialize df as None

        # Ensure the plots directory exists
        self.plots_dir = 'plots'
        if not os.path.exists(self.plots_dir):
            os.makedirs(self.plots_dir)
            logger.info("Created directory for plots: %s", self.plots_dir)

    def load_data(self, file_path):
        """Load the dataset from the specified CSV file."""
        try:
            self.df = pd.read_csv(file_path)
            logger.info("Dataset loaded successfully from %s", file_path)
        except Exception as e:
            logger.error("Failed to load dataset: %s", e)
            raise

    def analyze_data(self):
        if self.df is None or self.df.empty:
            logger.error("DataFrame is empty or not loaded")
            raise ValueError("DataFrame is empty or not loaded")

        numeric_features = self.df.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = self.df.select_dtypes(include=['object', 'category']).columns

        analysis_prompt = f"""
        I have a dataset with the following features:
        Numeric Features: {', '.join(numeric_features)}
        Categorical Features: {', '.join(categorical_features)}

        Please suggest the most appropriate plots to create for each feature and between pairs of features.
        """

        message = HumanMessage(content=analysis_prompt)

        try:
            response = self.model.invoke([message])  # Fixed this line to use self.model
            if hasattr(response, 'content'):
                logger.info("LLM plot suggestions received successfully")
                logger.info("LLM Plot Suggestions:\n%s", response.content)
                return numeric_features, categorical_features, response.content
            else:
                logger.error("No content in LLM response")
                return numeric_features, categorical_features, "No content available in the LLM response."
        except Exception as e:
            logger.error("An error occurred during LLM analysis: %s", e)
            return numeric_features, categorical_features, f"An error occurred: {e}"

    def generate_plots(self, numeric_features, categorical_features):
        for feature in numeric_features:
            plt.figure(figsize=(10, 6))
            sns.histplot(self.df[feature], kde=True)
            plt.title(f'Distribution of {feature}')
            plt.savefig(os.path.join(self.plots_dir, f'distribution_{feature}.png'))  # Save the plot as a file
            plt.close()  # Close the figure to free up memory
            logger.info("Generated distribution plot for %s", feature)

            plt.figure(figsize=(10, 6))
            sns.boxplot(x=self.df[feature])
            plt.title(f'Boxplot of {feature}')
            plt.savefig(os.path.join(self.plots_dir, f'boxplot_{feature}.png'))  # Save the plot as a file
            plt.close()  # Close the figure to free up memory
            logger.info("Generated boxplot for %s", feature)

        for feature in categorical_features:
            plt.figure(figsize=(10, 6))
            sns.countplot(x=self.df[feature])
            plt.title(f'Count Plot of {feature}')
            plt.xticks(rotation=45)
            plt.savefig(os.path.join(self.plots_dir, f'countplot_{feature}.png'))  # Save the plot as a file
            plt.close()  # Close the figure to free up memory
            logger.info("Generated count plot for %s", feature)

            for num_feature in numeric_features:
                plt.figure(figsize=(10, 6))
                sns.barplot(x=self.df[feature], y=self.df[num_feature])
                plt.title(f'{num_feature} vs {feature}')
                plt.xticks(rotation=45)
                plt.savefig(os.path.join(self.plots_dir, f'barplot_{num_feature}_vs_{feature}.png'))  # Save the plot as a file
                plt.close()  # Close the figure to free up memory
                logger.info("Generated bar plot for %s vs %s", num_feature, feature)

        if len(numeric_features) > 1:
            plt.figure(figsize=(10, 6))
            sns.pairplot(self.df[numeric_features])
            plt.title('Pairplot of Numeric Features')
            plt.savefig(os.path.join(self.plots_dir, 'pairplot_numeric_features.png'))  # Save the plot as a file
            plt.close()  # Close the figure to free up memory
            logger.info("Generated pairplot of numeric features")

            plt.figure(figsize=(10, 6))
            correlation_matrix = self.df[numeric_features].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
            plt.title('Correlation Heatmap')
            plt.savefig(os.path.join(self.plots_dir, 'correlation_heatmap.png'))  # Save the plot as a file
            plt.close()  # Close the figure to free up memory
            logger.info("Generated correlation heatmap")
