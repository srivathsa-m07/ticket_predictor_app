# IT Support Ticket Resolution Time Predictor

## Project Overview
This project predicts the resolution time for IT support tickets using a machine learning regression model (XGBoost). The application provides a Flask-based web interface allowing users to enter ticket details and receive predicted resolution times in hours. Additionally, it offers a clustering dashboard that visualizes ticket groupings using KMeans and Plotly.

## Project Flow
1. Users open the web app at the home route `/`.
2. They fill out a form with ticket features including issue type, priority, project name, number of contributors, comments count, processing steps, day of the week, month, and hour.
3. The form inputs are encoded using saved `LabelEncoder` models.
4. Encoded data is passed to a pre-trained XGBoost regression model that predicts ticket resolution time.
5. The predicted resolution time is displayed on the page.
6. Users can visit `/dashboard` to view clustering and resolution distribution visualizations based on historical ticket data.

## Dependencies
Install these Python packages before running the app:

- Python 3.x
- Flask
- pandas
- scikit-learn
- joblib
- plotly
- xgboost

Install all dependencies with:

pip install flask pandas scikit-learn joblib plotly xgboost

text

## Running the Application

1. Clone or download this repository:

git clone https://github.com/srivathsa-m07/ticket_predictor_app.git

text

2. Navigate into the project directory:

cd ticket_predictor_app

text

3. Run the Flask app:

python app.py

text

4. Access the web app in your browser:
   - Prediction page: [http://127.0.0.1:5000](http://127.0.0.1:5000)
   - Dashboard: [http://127.0.0.1:5000/dashboard](http://127.0.0.1:5000/dashboard)

## Output Screenshots

### Prediction Form with Sample Input and Output

![Prediction Page](screenshots/prediction.png)

### Ticket Clustering and Resolution Dashboard

![Dashboard Page](screenshots/dashboard.png)

