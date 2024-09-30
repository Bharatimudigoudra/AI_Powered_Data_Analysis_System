from flask import Flask, render_template, request, redirect, url_for
import os
from data_preprocessing import DataPreprocessor
from analyze_dataset import DatasetAnalyzer
from llm_DataAnalyzer import DataAnalyzer
from generate_insights import DataInsightGenerator
import logzero
from logzero import logger
import io

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Logging setup
logzero.logfile("app.log", maxBytes=1e6, backupCount=3)

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        logger.error("No file part in request")
        return redirect(request.url)

    file = request.files['file']
    
    if file.filename == '':
        logger.error("No selected file")
        return redirect(request.url)

    if file and file.filename.endswith('.csv'):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        logger.info("File %s uploaded successfully", file.filename)

        try:
            # Preprocess the data
            processor = DataPreprocessor(file_path)

            original_df_html = processor.df.to_html(classes='table table-striped')
            original_df_head_html = processor.df.head().to_html(classes='table table-striped')
            
            original_data_info = io.StringIO()
            processor.df.info(buf=original_data_info)
            original_df_info = original_data_info.getvalue()

            processor.convert()

            updated_df_html = processor.df.to_html(classes='table table-striped')
            updated_df_head_html = processor.df.head().to_html(classes='table table-striped')

            updated_data_info = io.StringIO()
            processor.df.info(buf=updated_data_info)
            updated_df_info = updated_data_info.getvalue()

            # Analyze the dataset with DatasetAnalyzer
            analyzer = DatasetAnalyzer(api_key='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')
            analysis_report = analyzer.analyze_dataset(file_path)
            
            # Initialize DataAnalyzer with API key and load data
            llm_analyzer = DataAnalyzer(api_key='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')
            llm_analyzer.load_data(file_path)  # Pass the file_path to load_data
            numeric_features, categorical_features, plot_suggestions = llm_analyzer.analyze_data()
            llm_analyzer.generate_plots(numeric_features, categorical_features)

            # Generate insights from the processed data
            insight_generator = DataInsightGenerator(model_name="llama-3.1-70b-versatile", api_key='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')
            insights = insight_generator.generate_insights(processor.df)

            return render_template('report1.html', 
                                   original_df_html=original_df_html, 
                                   original_df_head_html=original_df_head_html, 
                                   original_df_info=original_df_info,
                                   updated_df_html=updated_df_html, 
                                   updated_df_head_html=updated_df_head_html, 
                                   updated_df_info=updated_df_info,
                                   analysis_report=analysis_report,
                                   plot_suggestions=plot_suggestions,
                                   insights=insights)  # Pass insights to the template
        except Exception as e:
            logger.error("Error processing file: %s", str(e))
            return "Error processing file."

    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)
