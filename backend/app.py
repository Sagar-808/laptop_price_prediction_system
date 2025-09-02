from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
import joblib
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
from functools import wraps

from model.laptop_preprocess import build_input_dataframe, parse_laptop_dataframe
try:
    # Optional MongoDB client (graceful if unavailable)
    from db.mongo_client import mongo_client  # type: ignore
except Exception:
    mongo_client = None  # type: ignore

# Add relevant directories to Python path
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR = os.path.join(_BACKEND_DIR, "model")
if _BACKEND_DIR not in sys.path:
    sys.path.append(_BACKEND_DIR)
if _MODEL_DIR not in sys.path:
    sys.path.append(_MODEL_DIR)

app = Flask(
    __name__, template_folder="../frontend/templates", static_folder="../frontend/static"
)

# Basic config
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "adminadmin")

# Global variables for model and preprocessing
model_data = None
pipeline = None
feature_columns = []
categorical_columns = []
numerical_columns = []
options_cache = None  # holds unique option values from dataset


def admin_required(view_func):
    """Decorator to protect admin routes with simple session auth."""
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if not session.get("is_admin"):
            return redirect(url_for("admin_login", next=request.path))
        return view_func(*args, **kwargs)
    return wrapper

def load_model():
    """Load the trained laptop price model and metadata."""
    global model_data, pipeline, feature_columns, categorical_columns, numerical_columns
    try:
        model_path = os.path.join(os.path.dirname(__file__), "model", "laptop_price_model.pkl")
        if not os.path.exists(model_path):
            print("Model file not found. Please run the training script first.")
            return False
        model_data = joblib.load(model_path)
        pipeline = model_data["pipeline"]
        feature_columns = model_data.get("feature_columns", [])
        categorical_columns = model_data.get("categorical_columns", [])
        numerical_columns = model_data.get("numerical_columns", [])
        print("Laptop price model loaded.")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False


def load_options():
    """Load unique dropdown options from laptop_data.csv once and cache them."""
    global options_cache
    if options_cache is not None:
        return options_cache
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(project_root, "laptop_data.csv")
        if not os.path.exists(csv_path):
            print("Dataset not found for options:", csv_path)
            options_cache = {}
            return options_cache
        df = pd.read_csv(csv_path)
        def uniq(col):
            return sorted(df[col].dropna().astype(str).unique().tolist()) if col in df.columns else []

        options_cache = {
            "Company": uniq("Company"),
            "TypeName": uniq("TypeName"),
            "Inches": uniq("Inches"),
            "ScreenResolution": uniq("ScreenResolution"),
            "Cpu": uniq("Cpu"),
            "Ram": uniq("Ram"),
            "Memory": uniq("Memory"),
            "Gpu": uniq("Gpu"),
            "OpSys": uniq("OpSys"),
            "Weight": uniq("Weight"),
        }
        print("Loaded dropdown options from dataset.")
        return options_cache
    except Exception as e:
        print("Error loading options:", e)
        options_cache = {}
        return options_cache

def preprocess_input(laptop_form: dict) -> pd.DataFrame:
    """Convert form JSON into engineered feature DataFrame expected by the pipeline."""
    try:
        raw_df = build_input_dataframe(laptop_form)
        X_engineered, _, _, _ = parse_laptop_dataframe(raw_df.assign(Price=0.0), drop_target=False)
        # Ensure column order
        missing = [c for c in feature_columns if c not in X_engineered.columns]
        for c in missing:
            X_engineered[c] = 0
        X_engineered = X_engineered[feature_columns]
        return X_engineered
    except Exception as e:
        print("Preprocess error:", e)
        return None

def make_prediction(laptop_form: dict):
    """Return predicted price (converted to NPR) and simple metadata.
    Assumes model was trained on INR; converts to NPR using a configurable rate.
    """
    try:
        X = preprocess_input(laptop_form)
        if X is None:
            return {"error": "Invalid input"}
        # Base prediction in INR (model label scale)
        price_in_inr = float(pipeline.predict(X)[0])
        # Conversion: INR -> NPR (roughly pegged ~1.6); allow override via env
        try:
            rate = float(os.getenv("INR_TO_NPR_RATE", "1.6"))
        except Exception:
            rate = 1.6
        price_in_npr = price_in_inr * rate
        return {
            "predicted_price": price_in_npr,
            "currency": "NPR",
            "conversion": {
                "from": "INR",
                "to": "NPR",
                "rate": rate,
            },
        }
    except Exception as e:
        print("Prediction error:", e)
        return {"error": str(e)}


def _append_jsonl(path: str, record: dict):
    """Append a record to a JSONL file (creates dirs/files if needed)."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        import json
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception as e:
        print("Failed to append JSONL:", e)


def log_prediction(specs: dict, predicted_price: float):
    """Persist a prediction to MongoDB if available; fallback to local file."""
    record = {
        "timestamp": datetime.now().isoformat(),
        "specs": specs,
        "predicted_price": float(predicted_price),
    }
    # Try MongoDB first
    try:
        if mongo_client is not None:
            ok = mongo_client.save_prediction(record)
            if ok:
                return True
    except Exception as e:
        print("Mongo save failed:", e)
    # Fallback to local file
    fallback_path = os.path.join(_BACKEND_DIR, "db", "predictions.jsonl")
    _append_jsonl(fallback_path, record)
    return True


def get_recent_predictions(limit: int = 50):
    """Get recent predictions from Mongo or local file."""
    # Prefer Mongo
    try:
        if mongo_client is not None and hasattr(mongo_client, "get_predictions"):
            preds = mongo_client.get_predictions(limit=limit)
            # Normalize records for template (ensure string timestamp, dict specs)
            for p in preds:
                ts = p.get("timestamp")
                if isinstance(ts, datetime):
                    p["timestamp"] = ts.isoformat()
                elif ts is not None:
                    p["timestamp"] = str(ts)
                if not isinstance(p.get("specs"), dict):
                    try:
                        # Sometimes specs might come as JSON string
                        import json as _json
                        p["specs"] = _json.loads(p.get("specs") or "{}")
                    except Exception:
                        p["specs"] = {}
            return preds
    except Exception as e:
        print("Mongo fetch failed:", e)
    # Fallback to local file (JSONL)
    try:
        import json
        path = os.path.join(_BACKEND_DIR, "db", "predictions.jsonl")
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()[-limit:]
        preds = []
        for ln in lines:
            try:
                rec = json.loads(ln)
            except Exception:
                continue
            # Normalize timestamp
            ts = rec.get("timestamp")
            if isinstance(ts, datetime):
                rec["timestamp"] = ts.isoformat()
            elif ts is not None:
                rec["timestamp"] = str(ts)
            # Normalize specs
            if not isinstance(rec.get("specs"), dict):
                try:
                    rec["specs"] = json.loads(rec.get("specs") or "{}")
                except Exception:
                    rec["specs"] = {}
            preds.append(rec)
        preds.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return preds
    except Exception as e:
        print("Read JSONL failed:", e)
        return []


# Contact messages storage

def _append_jsonl_record(name: str, record: dict):
    path = os.path.join(_BACKEND_DIR, "db", f"{name}.jsonl")
    _append_jsonl(path, record)


def save_contact_message(msg: dict) -> bool:
    record = {
        **msg,
        "timestamp": datetime.now().isoformat(),
    }
    # Try Mongo directly if available
    try:
        if mongo_client is not None and getattr(mongo_client, "db", None) is not None:
            coll = mongo_client.db["messages"]
            coll.insert_one(record)
            return True
    except Exception as e:
        print("Mongo save message failed:", e)
    # Fallback: file
    _append_jsonl_record("messages", record)
    return True


def get_recent_messages(limit: int = 50) -> list:
    # Mongo preferred
    try:
        if mongo_client is not None and getattr(mongo_client, "db", None) is not None:
            coll = mongo_client.db["messages"]
            cur = coll.find().sort("timestamp", -1).limit(limit)
            msgs = []
            for m in cur:
                m["_id"] = str(m.get("_id"))
                ts = m.get("timestamp")
                if isinstance(ts, datetime):
                    m["timestamp"] = ts.isoformat()
                msgs.append(m)
            return msgs
    except Exception as e:
        print("Mongo fetch messages failed:", e)
    # Fallback file
    import json
    path = os.path.join(_BACKEND_DIR, "db", "messages.jsonl")
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()[-limit:]
        msgs = []
        for ln in lines:
            try:
                msgs.append(json.loads(ln))
            except Exception:
                continue
        msgs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return msgs
    except Exception as e:
        print("Read messages JSONL failed:", e)
        return []


def total_predictions_count() -> int:
    try:
        if mongo_client is not None and getattr(mongo_client, "collection", None) is not None:
            # Use collection count directly
            return mongo_client.collection.count_documents({})
        if mongo_client is not None and hasattr(mongo_client, "get_prediction_stats"):
            stats = mongo_client.get_prediction_stats()
            return int(stats.get("total_predictions", 0) or 0)
    except Exception:
        pass
    # Fallback: count lines in predictions.jsonl
    path = os.path.join(_BACKEND_DIR, "db", "predictions.jsonl")
    try:
        if not os.path.exists(path):
            return 0
        with open(path, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


def total_messages_count() -> int:
    try:
        if mongo_client is not None and getattr(mongo_client, "db", None) is not None:
            return mongo_client.db["messages"].count_documents({})
    except Exception:
        pass
    path = os.path.join(_BACKEND_DIR, "db", "messages.jsonl")
    try:
        if not os.path.exists(path):
            return 0
        with open(path, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


def format_count_short(n: int) -> str:
    try:
        n = int(n)
    except Exception:
        return str(n)
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M+".replace(".0", "")
    if n >= 1_000:
        return f"{n/1_000:.1f}K+".replace(".0", "")
    return f"{n}+"

@app.route('/')
def home():
    """Home page"""
    return render_template('index.html')

@app.route('/about')
def about():
    """About page"""
    total_preds = total_predictions_count()
    return render_template('about.html', total_predictions=total_preds, total_predictions_pretty=format_count_short(total_preds))

@app.route('/services')
def services():
    """Services page"""
    return render_template('services.html')

@app.route('/contact')
def contact():
    """Contact page"""
    return render_template('contact.html')


@app.route('/api/contact', methods=['POST'])
def api_contact():
    try:
        data = request.get_json(silent=True) or request.form.to_dict()
        required = ["firstName", "lastName", "email", "company", "inquiryType", "message"]
        if not all((data.get(k) or "").strip() for k in required):
            return jsonify({"error": "Missing required fields"}), 400
        save_contact_message(data)
        return jsonify({"ok": True}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict")
def predict_page():
    return render_template("predict.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if pipeline is None:
            return render_template("result.html", error="Model not loaded", laptop_data={})
        data = request.form.to_dict()
        result = make_prediction(data)
        if "error" in result:
            return render_template("result.html", error=result["error"], laptop_data=data)
        # Persist prediction for admin analytics (best-effort)
        try:
            log_prediction(data, result.get("predicted_price"))
        except Exception as _:
            pass
        return render_template(
            "result.html", predicted_price=result["predicted_price"], laptop_data=data
        )
    except Exception as e:
        return render_template("result.html", error=str(e), laptop_data={})

@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        if pipeline is None:
            return jsonify({"error": "Model not loaded"}), 500
        data = request.get_json() or {}
        result = make_prediction(data)
        # Persist on success
        if "error" not in result:
            try:
                log_prediction(data, result.get("predicted_price"))
            except Exception:
                pass
        return jsonify(result), (200 if "error" not in result else 400)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/options", methods=["GET"])
def api_options():
    try:
        opts = load_options()
        return jsonify(opts)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Removed MongoDB-related endpoints for this regression app

# Admin Panel

def _dataset_info():
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(project_root, "laptop_data.csv")
        rows = 0
        if os.path.exists(csv_path):
            try:
                import csv as _csv
                with open(csv_path, "r", encoding="utf-8") as f:
                    rows = sum(1 for _ in f) - 1  # rough, excluding header
            except Exception:
                rows = 0
            mtime = datetime.fromtimestamp(os.path.getmtime(csv_path)).isoformat()
        else:
            mtime = None
        return {"path": csv_path, "rows": max(rows, 0), "last_modified": mtime}
    except Exception:
        return {"path": None, "rows": 0, "last_modified": None}


@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session["is_admin"] = True
            flash("Welcome back, admin!", "success")
            nxt = request.args.get("next") or url_for("admin_dashboard")
            return redirect(nxt)
        flash("Invalid credentials", "danger")
    return render_template("admin/login.html")


@app.route("/admin/logout")
def admin_logout():
    session.pop("is_admin", None)
    return redirect(url_for("home"))


@app.route("/admin")
@admin_required
def admin_dashboard():
    metrics = None
    trained_at = None
    if isinstance(model_data, dict):
        metrics = model_data.get("metrics")
        trained_at = model_data.get("trained_at")
    dataset = _dataset_info()
    total_preds = total_predictions_count()
    total_msgs = total_messages_count()
    recent_msgs = get_recent_messages(limit=5)
    recents = get_recent_predictions(limit=10)
    return render_template(
        "admin/dashboard.html",
        metrics=metrics,
        trained_at=trained_at,
        dataset=dataset,
        recent_predictions=recents,
        recent_messages=recent_msgs,
        total_predictions=total_preds,
        total_messages=total_msgs,
        mongo_enabled=bool(mongo_client),
    )


@app.route("/admin/predictions")
@admin_required
def admin_predictions():
    limit = 200
    recents = get_recent_predictions(limit=limit)
    total_preds = total_predictions_count()
    return render_template(
        "admin/predictions.html",
        predictions=recents,
        total_predictions=total_preds,
        shown_count=len(recents),
        limit=limit,
    )


@app.route("/api/admin/predictions")
@admin_required
def api_admin_predictions():
    limit = int(request.args.get("limit", 100))
    return jsonify(get_recent_predictions(limit=limit))


@app.route("/admin/dataset/preview")
@admin_required
def admin_dataset_preview():
    info = _dataset_info()
    path = info.get("path")
    if not path or not os.path.exists(path):
        flash("Dataset file not found", "danger")
        return redirect(url_for("admin_dashboard"))
    # Build preview HTML (first 100 rows)
    try:
        import pandas as _pd
        df = _pd.read_csv(path)
        preview_html = df.head(100).to_html(classes="admin-table", index=False)
    except Exception as e:
        preview_html = f"<div class='error-card'><p>Failed to render preview: {e}</p></div>"
    return render_template(
        "admin/dataset_preview.html",
        dataset=info,
        preview_html=preview_html,
    )


@app.route("/admin/dataset/download")
@admin_required
def admin_dataset_download():
    info = _dataset_info()
    path = info.get("path")
    if not path or not os.path.exists(path):
        flash("Dataset file not found", "danger")
        return redirect(url_for("admin_dashboard"))
    from flask import send_file
    fname = os.path.basename(path) or "laptop_data.csv"
    return send_file(path, as_attachment=True, download_name=fname)


@app.route("/admin/messages")
@admin_required
def admin_messages():
    msgs = get_recent_messages(limit=200)
    return render_template("admin/messages.html", messages=msgs)


@app.route("/api/admin/messages")
@admin_required
def api_admin_messages():
    limit = int(request.args.get("limit", 100))
    return jsonify(get_recent_messages(limit=limit))

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return render_template('index.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return render_template('index.html'), 500

def main():
    print("Starting Laptop Price Predictor...")
    if not load_model():
        print("Failed to load model. Exiting...")
        return
    # Preload options for dropdowns
    load_options()
    port = int(os.getenv("PORT", "5000"))
    print(f"Server running at http://localhost:{port}")
    app.run(debug=True, host='0.0.0.0', port=port)

if __name__ == '__main__':
    main() 