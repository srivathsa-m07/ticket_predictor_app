# app.py — final version (uses joblib, robust column detection, real clustering + dashboard)
import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.io as pio

app = Flask(__name__)

# ---------------------------
# 1) Load model + encoders
# ---------------------------
MODEL_PATH = "model_xgb.pkl"
ENC_TYPE_PATH = "enc_issue_type.pkl"
ENC_PRIO_PATH = "enc_issue_priority.pkl"
ENC_PROJ_PATH = "enc_issue_proj.pkl"
META_PATH = "meta.json"
DATA_CSV = "issues.csv"   # put your issues.csv here (you already have it)

# load XGBoost model and encoders using joblib (not pickle)
model = joblib.load(MODEL_PATH)
enc_type = joblib.load(ENC_TYPE_PATH)
enc_prio = joblib.load(ENC_PRIO_PATH)
enc_proj = joblib.load(ENC_PROJ_PATH)

# optional meta.json (feature order / target log flag)
meta = {}
if os.path.exists(META_PATH):
    try:
        with open(META_PATH, "r") as f:
            meta = json.load(f)
    except Exception:
        meta = {}

TARGET_LOG = bool(meta.get("target_log_transform", meta.get("target_log", False)))
# fallback feature order (common from earlier steps)
DEFAULT_FEATURE_ORDER = [
    "issue_type_enc", "issue_priority_enc", "issue_proj_enc",
    "issue_contr_count", "issue_comments_count", "processing_steps",
    "day_of_week", "month", "hour_of_day"
]
FEATURE_ORDER = meta.get("features_order", meta.get("features", DEFAULT_FEATURE_ORDER))

# ---------------------------
# 2) Load dataset and prepare columns
# ---------------------------
if not os.path.exists(DATA_CSV):
    raise FileNotFoundError(f"Place {DATA_CSV} in the same folder as app.py before running the app.")

data = pd.read_csv(DATA_CSV, low_memory=False)

# helper to pick a column name among candidates
def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

# detect column names in your CSV (many copies used slightly different names)
issue_type_col = pick_col(data, ["issue_type", "type", "Issue Type"])
priority_col = pick_col(data, ["issue_priority", "priority", "Priority"])
proj_col = pick_col(data, ["issue_proj", "project", "proj", "issue_project"])
contr_col = pick_col(data, ["issue_contr_count", "contributors", "contr"])
comments_col = pick_col(data, ["issue_comments_count", "comments", "issue_comments"])
steps_col = pick_col(data, ["processing_steps", "steps", "processing_step"])

# create standardized columns (fill missing with defaults)
data["issue_type_std"] = data[issue_type_col].astype(str) if issue_type_col else "unknown"
data["priority_std"] = data[priority_col].astype(str) if priority_col else "unknown"
data["proj_std"] = data[proj_col].astype(str) if proj_col else "unknown"
data["contr_std"] = pd.to_numeric(data[contr_col], errors="coerce").fillna(0) if contr_col else 0
data["comments_std"] = pd.to_numeric(data[comments_col], errors="coerce").fillna(0) if comments_col else 0
data["steps_std"] = pd.to_numeric(data[steps_col], errors="coerce").fillna(0) if steps_col else 0

# compute resolution_hours if timestamps exist
if "started" in data.columns and "ended" in data.columns:
    data["started"] = pd.to_datetime(data["started"], errors="coerce")
    data["ended"] = pd.to_datetime(data["ended"], errors="coerce")
    data["resolution_hours"] = (data["ended"] - data["started"]).dt.total_seconds() / 3600.0
else:
    data["resolution_hours"] = pd.NA

# map encoders -> mapping dicts for fast transform (safe fallback to 0)
type_map = {v: i for i, v in enumerate(enc_type.classes_)} if hasattr(enc_type, "classes_") else {}
prio_map = {v: i for i, v in enumerate(enc_prio.classes_)} if hasattr(enc_prio, "classes_") else {}
proj_map = {v: i for i, v in enumerate(enc_proj.classes_)} if hasattr(enc_proj, "classes_") else {}

data["issue_type_enc"] = data["issue_type_std"].map(type_map).fillna(0).astype(int)
data["issue_priority_enc"] = data["priority_std"].map(prio_map).fillna(0).astype(int)
data["issue_proj_enc"] = data["proj_std"].map(proj_map).fillna(0).astype(int)

# ---------------------------
# 3) Train clustering (on full dataset) and keep model in memory
# ---------------------------
# Choose clustering features (encoded categories + numeric ticket stats)
cluster_cols = ["issue_type_enc", "issue_priority_enc", "issue_proj_enc", "contr_std", "comments_std", "steps_std"]
cluster_df = data[cluster_cols].fillna(0)

# train KMeans (k=3 is reasonable by elbow earlier)
K = 3
cluster_model = KMeans(n_clusters=K, random_state=42, n_init=10)
data["cluster"] = cluster_model.fit_predict(cluster_df)

# precompute cluster averages for dashboard / prediction hint
cluster_avg = data.groupby("cluster")["resolution_hours"].mean().to_dict()

# ---------------------------
# Utility functions
# ---------------------------
def safe_encode_value(value, map_dict):
    # return mapped int if present, else 0
    return int(map_dict.get(value, 0))

def build_model_input_from_form(form):
    # gather values from form (names match index.html below)
    itype = form.get("issue_type", "")
    iprio = form.get("priority", "")
    iproj = form.get("project", "")
    try:
        contr = float(form.get("contributors", 0))
    except:
        contr = 0.0
    try:
        comments = float(form.get("comments", 0))
    except:
        comments = 0.0
    try:
        steps = float(form.get("steps", 0))
    except:
        steps = 0.0
    try:
        day = int(form.get("day_of_week", 0))
    except:
        day = 0
    try:
        month = int(form.get("month", 0))
    except:
        month = 0
    try:
        hour = int(form.get("hour", 0))
    except:
        hour = 0

    # encode using maps (fallback 0)
    type_enc = safe_encode_value(itype, type_map)
    prio_enc = safe_encode_value(iprio, prio_map)
    proj_enc = safe_encode_value(iproj, proj_map)

    # Build vector following FEATURE_ORDER if possible
    # map our standardized names to order names
    name_to_value = {
        "issue_type_enc": type_enc,
        "issue_priority_enc": prio_enc,
        "issue_proj_enc": proj_enc,
        "issue_contr_count": contr,
        "issue_comments_count": comments,
        "processing_steps": steps,
        "day_of_week": day,
        "month": month,
        "hour_of_day": hour
    }

    # create the feature vector in order
    vec = [float(name_to_value.get(name, 0.0)) for name in FEATURE_ORDER]
    X_input = np.array(vec).reshape(1, -1)
    return X_input, (type_enc, prio_enc, proj_enc, contr, comments, steps)

# ---------------------------
# Routes
# ---------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    cluster_info = None
    # prepare dropdown options from encoder classes passed to template
    types = list(enc_type.classes_) if hasattr(enc_type, "classes_") else []
    priorities = list(enc_prio.classes_) if hasattr(enc_prio, "classes_") else []
    projects = list(enc_proj.classes_) if hasattr(enc_proj, "classes_") else []

    if request.method == "POST":
        try:
            X_input, cluster_input_components = build_model_input_from_form(request.form)
            raw_pred = model.predict(X_input)[0]
            if TARGET_LOG:
                pred_hours = float(np.expm1(raw_pred))
            else:
                pred_hours = float(raw_pred)

            # cluster prediction for this new ticket: use same cluster feature ordering
            # we trained cluster_model on [issue_type_enc, issue_priority_enc, issue_proj_enc, contr_std, comments_std, steps_std]
            t_enc, p_enc, pr_enc, contr, comments, steps = cluster_input_components
            cluster_row = np.array([t_enc, p_enc, pr_enc, contr, comments, steps]).reshape(1, -1)
            cluster_label = int(cluster_model.predict(cluster_row)[0])
            avg_for_cluster = cluster_avg.get(cluster_label, None)

            prediction = f"{round(pred_hours, 2)} hours (~ {int(pred_hours // 24)} days + {round(pred_hours % 24,2)} hrs)"
            cluster_info = {
                "label": int(cluster_label),
                "avg_resolution_hours": None if pd.isna(avg_for_cluster) else round(float(avg_for_cluster), 2)
            }
        except Exception as e:
            prediction = f"Error: {e}"

    return render_template(
        "index.html",
        prediction=prediction,
        cluster_info=cluster_info,
        types=types,
        priorities=priorities,
        projects=projects
    )

@app.route("/dashboard")
def dashboard():
    # Build list of Plotly HTML fragments and render them together
    fragments = []

    # scatter: comments vs steps by cluster (works if those columns exist)
    if "comments_std" in data.columns and "steps_std" in data.columns:
        fig1 = px.scatter(
            data_frame=data,
            x="comments_std", y="steps_std",
            color=data["cluster"].astype(str),
            hover_data=["issue_type_std", "priority_std", "proj_std"],
            title="Ticket Clusters (comments vs processing steps)",
            labels={"comments_std": "Comments", "steps_std": "Processing Steps", "cluster": "Cluster"}
        )
        fragments.append(pio.to_html(fig1, full_html=False))

    # histogram of resolution times
    if "resolution_hours" in data.columns and data["resolution_hours"].notna().any():
        fig2 = px.histogram(data_frame=data, x="resolution_hours", nbins=50,
                            title="Distribution of Resolution Time (hours)",
                            labels={"resolution_hours": "Resolution hours"})
        fragments.append(pio.to_html(fig2, full_html=False))

        # average resolution per cluster
        cluster_avg_df = data.dropna(subset=["resolution_hours"]).groupby("cluster")["resolution_hours"].mean().reset_index()
        fig3 = px.bar(cluster_avg_df, x="cluster", y="resolution_hours",
                      title="Average Resolution Time per Cluster",
                      labels={"resolution_hours": "Avg resolution (hrs)"})
        fragments.append(pio.to_html(fig3, full_html=False))

        # boxplot resolution by issue type (if available)
        if "issue_type_std" in data.columns:
            tmp = data.dropna(subset=["resolution_hours"])
            fig4 = px.box(tmp, x="issue_type_std", y="resolution_hours",
                          title="Resolution Time by Issue Type",
                          labels={"issue_type_std": "Issue Type", "resolution_hours": "Resolution (hrs)"})
            fragments.append(pio.to_html(fig4, full_html=False))

    # combine fragments and render in dashboard.html
    graph_html = "\n".join(fragments)
    return render_template("dashboard.html", graph=graph_html)


if __name__ == "__main__":
    # run with debug for local testing
    app.run(debug=True)
