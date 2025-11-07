from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import numpy as np
import io
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import sklearn
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Ensure NLTK resources available (download only if missing)
def ensure_nltk():
    try:
        _ = stopwords.words("english")
        _ = word_tokenize("test")
    except LookupError:
        nltk.download("punkt")
        nltk.download("stopwords")

ensure_nltk()

STOPWORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()

# Models available
MODEL_DICT = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Naive Bayes": MultinomialNB(),
    "SVM (Linear)": SVC(kernel="linear", probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=200),
    "Gradient Boosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier()
}

# Simple NLTK preprocessing
def preprocess_texts(texts, use_stemming=False):
    processed = []
    for t in texts:
        if pd.isna(t):
            processed.append("")
            continue
        # tokenize words, keep alphabetic tokens
        tokens = [w.lower() for w in word_tokenize(str(t)) if w.isalpha()]
        tokens = [w for w in tokens if w not in STOPWORDS]
        if use_stemming:
            tokens = [STEMMER.stem(w) for w in tokens]
        processed.append(" ".join(tokens))
    return processed

# Compute metrics (weighted)
def compute_metrics(y_true, y_pred):
    return {
        "Accuracy": metrics.accuracy_score(y_true, y_pred),
        "Precision": metrics.precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": metrics.recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1-Score": metrics.f1_score(y_true, y_pred, average="weighted", zero_division=0)
    }

@app.route("/", methods=["GET"])
def index():
    algorithms = list(MODEL_DICT.keys())
    return render_template("index.html", algorithms=algorithms)

@app.route("/train", methods=["POST"])
def train():
    # File check
    if "dataset" not in request.files:
        flash("No file part. Select a TSV file to upload.")
        return redirect(url_for("index"))

    file = request.files["dataset"]
    if file.filename == "":
        flash("No file selected.")
        return redirect(url_for("index"))

    # Get form options
    selected_algo = request.form.get("algorithm")
    use_stem = True if request.form.get("use_stem") == "on" else False
    test_size = float(request.form.get("test_size") or 0.2)

    # Read TSV (tab separated)
    try:
        content = file.stream.read().decode("utf-8")
        df = pd.read_csv(io.StringIO(content), sep="\t")
    except Exception as e:
        flash(f"Could not read TSV. Make sure it is tab-separated (.tsv). Error: {e}")
        return redirect(url_for("index"))

    # Validate required columns
    if "text" not in df.columns or "label" not in df.columns:
        flash("TSV must contain columns named exactly: 'text' and 'label' (tab-separated).")
        return redirect(url_for("index"))

    # Preprocess
    df["text_proc"] = preprocess_texts(df["text"].astype(str).tolist(), use_stemming=use_stem)

    # Labels (keep as-is; sklearn handles string labels)
    y = df["label"].values
    X = df["text_proc"].values

    # Vectorize
    vectorizer = TfidfVectorizer(max_features=20000)
    X_vec = vectorizer.fit_transform(X)

    # Train-test split with stratify if possible
    stratify = y if len(np.unique(y)) > 1 and len(y) >= 2 * len(np.unique(y)) else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=test_size, random_state=42, stratify=stratify)
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=test_size, random_state=42)

    # Train all models and collect metrics
    results = []
    trained_models = {}
    for name, model in MODEL_DICT.items():
        try:
            clf = sklearn.base.clone(model)
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            m = compute_metrics(y_test, preds)
            results.append({
                "Algorithm": name,
                "Accuracy": round(m["Accuracy"], 4),
                "Precision": round(m["Precision"], 4),
                "Recall": round(m["Recall"], 4),
                "F1-Score": round(m["F1-Score"], 4)
            })
            trained_models[name] = clf
        except Exception as e:
            # Something failed for this model
            results.append({
                "Algorithm": name,
                "Accuracy": None,
                "Precision": None,
                "Recall": None,
                "F1-Score": None,
                "Error": str(e)
            })

    results_df = pd.DataFrame(results).sort_values(by="F1-Score", ascending=False, na_position="last")

    # Also get single selected model metrics (if requested)
    single_result = None
    if selected_algo in trained_models:
        clf = trained_models[selected_algo]
        preds = clf.predict(X_test)
        single_res = compute_metrics(y_test, preds)
        single_result = {
            "Algorithm": selected_algo,
            "Accuracy": round(single_res["Accuracy"], 4),
            "Precision": round(single_res["Precision"], 4),
            "Recall": round(single_res["Recall"], 4),
            "F1-Score": round(single_res["F1-Score"], 4)
        }

    # Prepare HTML tables
    comparison_table = results_df.to_html(classes="table table-striped table-bordered", index=False, float_format="%.4f", na_rep="N/A")
    single_table = None
    if single_result:
        single_table = pd.DataFrame([single_result]).to_html(classes="table table-sm table-bordered", index=False, float_format="%.4f")

    # Return results page; also keep vectorizer & best model in session? (Not persisted here)
    return render_template(
        "results.html",
        comparison_table=comparison_table,
        single_table=single_table,
        selected_algo=selected_algo,
        use_stem=use_stem,
        test_size=test_size
    )

if __name__ == "__main__":
    # Ensure templates folder exists (the files are provided below)
    app.run(debug=True, port=5000)
