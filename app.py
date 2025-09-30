from flask import Flask, request, jsonify, render_template
import joblib, numpy as np, pandas as pd
from scipy.sparse import hstack
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.io as pio

# Load trained model + vectorizer
model = joblib.load("model_xgb.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# 🔹 Load your real dataset
df = pd.read_csv("issues.csv")

# Keep only needed columns for clustering
cluster_features = ["contributors","comments","steps","resolution_hours"]
df_cluster = df[cluster_features].dropna()

# Train clustering model
kmeans = KMeans(n_clusters=3, random_state=42).fit(df_cluster)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    issue_type = data.get("issue_type", "")
    priority = data.get("priority", "")
    project = data.get("project", "")
    contributors = float(data.get("contributors", 0))
    comments = float(data.get("comments", 0))
    steps = float(data.get("steps", 0))
    day = float(data.get("day", 0))
    month = float(data.get("month", 1))
    hour = float(data.get("hour", 0))

    text_features = issue_type + " " + priority + " " + project
    text_vec = vectorizer.transform([text_features])

    num_features = np.array([[contributors, comments, steps, day, month, hour]])
    features = hstack([text_vec, num_features])

    prediction = model.predict(features)[0]
    prediction = float(prediction)

    days = int(prediction // 24)
    hours = round(prediction % 24, 2)

    return jsonify({
        "predicted_hours": round(prediction, 3),
        "formatted": f"{days} days {hours} hrs"
    })

# 🔹 Dashboard route
@app.route("/dashboard")
def dashboard():
    df_dash = df.copy()
    df_dash = df_dash.dropna(subset=cluster_features)
    df_dash["cluster"] = kmeans.predict(df_dash[cluster_features])

    # 1. Cluster distribution
    fig1 = px.histogram(df_dash, x="cluster", title="Ticket Cluster Distribution")
    graph1 = pio.to_html(fig1, full_html=False)

    # 2. Resolution time by issue type
    fig2 = px.box(df_dash, x="issue_type", y="resolution_hours", title="Resolution Time by Issue Type")
    graph2 = pio.to_html(fig2, full_html=False)

    # 3. Resolution by priority
    fig3 = px.box(df_dash, x="priority", y="resolution_hours", title="Resolution Time by Priority")
    graph3 = pio.to_html(fig3, full_html=False)

    return f"""
    <html>
    <head><title>Dashboard</title></head>
    <body style='background:#0d1117;color:white;font-family:Arial'>
        <h1>📊 IT Ticket Dashboard</h1>
        {graph1}
        {graph2}
        {graph3}
        <a href="/">⬅ Back to Predictor</a>
    </body>
    </html>
    """

if __name__ == "__main__":
    app.run(debug=True)
