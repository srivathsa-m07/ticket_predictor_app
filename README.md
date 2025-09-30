# Ticket Resolution Time Predictor

## Project Overview
This project is a Flask web application that predicts IT ticket resolution time based on various ticket features using a machine learning model. It also includes a clustering dashboard visualizing ticket groupings.

## Project Flow
- User accesses the web app at `/`
- User enters ticket details: issue type, priority, project name, contributors, comments, processing steps, day of week, month, and hour
- The app uses pre-trained encoders and an XGBoost model to predict ticket resolution time in hours
- The dashboard at `/dashboard` shows clusters of tickets based on type, priority, and project using KMeans and Plotly for visualization

## Dependencies
- Python 3.x
- Flask
- pandas
- scikit-learn
- joblib
- plotly
- xgboost (if model files require it)

Install dependencies using:

pip install flask pandas scikit-learn joblib plotly xgboost

text

## Running the Application
1. Clone the repository
2. Navigate to the project folder
3. Run the Flask app:

python app.py

text

4. Open the browser at http://127.0.0.1:5000 to use the predictor and http://127.0.0.1:5000/dashboard for the cluster dashboard

## Output Screenshots
![Prediction Page](screenshots/prediction_page.png)


---