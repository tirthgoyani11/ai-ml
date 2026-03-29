# CS7180 Final Project

## Features

- Select symptoms from a comprehensive dropdown list
- Predict potential diseases based on selected symptoms
- View confidence levels for multiple possible diseases
- Find recommended doctors based on the predicted condition
- Responsive design that works on desktop and mobile devices

## Quick Start

1. Clone this repository:

   ```bash
   git clone
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

   alternatively, you can use `conda` to create an environment:

   ```bash
   conda create -n disease_prediction
   conda activate disease_prediction
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:

   ```bash
   streamlit run app.py
   ```

5. Open your browser at `http://localhost:8501`


## Docker Setup
1. Build the Docker image:

   ```bash
   docker build -t disease_prediction_app .
   ```

2. Run the Docker container:

   ```bash
   docker run -d -p 8501:8501 --name disease_prediction_app disease_prediction_app:latest
   ```

3. Open your browser at `http://localhost:8501`

## How It Works

1. The app loads a pre-trained Multinomial Naive Bayes model
2. Users select symptoms from the dropdown menu
3. When the "Predict Disease" button is clicked, the model analyzes the symptoms
4. The system displays the most likely disease and alternative possibilities with confidence scores
5. Users can find recommended specialists for the predicted condition

## Project Structure

- `app.py`: Main Streamlit application
- `requirements.txt`: List of dependencies
- `disease_prediction_multinomial_naive_bayes.joblib`: Trained model file
- `model_features.csv`: List of symptoms the model was trained on

## Technical Details

- **Model**: Multinomial Naive Bayes
- **Features**: 300+ symptom indicators
- **Interface**: Streamlit web application
- **Data Preprocessing**: Binary encoding of symptoms

## For Developers

To modify the application:

1. Update the model: You can train and save a new model to replace the existing `.joblib` file
2. Add features: Modify the `app.py` file to add new functionality
3. Customize styling: Edit the CSS in the `st.markdown()` section at the top of `app.py`

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This application is for educational purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider with questions regarding medical conditions.
