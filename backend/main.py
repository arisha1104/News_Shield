from flask import Flask, request, redirect, url_for, render_template, session, jsonify     
from flask_cors import CORS
import pickle
import pandas as pd
import requests
import os
import json
import logging
from dotenv import load_dotenv

# --- Path setup ---
# Get the absolute path of the directory where the script is located
basedir = os.path.abspath(os.path.dirname(__file__))

# Define paths relative to the script's location
template_dir = os.path.join(basedir, '../frontend/templates')
model_path = os.path.join(basedir, "logistic_model.pkl")
vectorizer_path = os.path.join(basedir, "tfidf_vectorizer.pkl")
csv_path = os.path.join(basedir, "cleaned_news.csv")
dotenv_path = os.path.join(basedir, '.env')

# Load environment variables from .env file
load_dotenv(dotenv_path=dotenv_path)

app = Flask(__name__, template_folder=template_dir)
CORS(app)  # Enable CORS for all routes
app.secret_key = os.getenv("SERPAPI_KEY")  # Used for sessions

# Hardcoded user credentials (you can modify these)
USER_CREDENTIALS = {
    "username": "admin",
    "password": "password123"
}

# Load the trained model and vectorizer
model = None
vectorizer = None
df = None

try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Warning: Could not load logistic_model.pkl: {e}")

try:
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
except Exception as e:
    print(f"Warning: Could not load tfidf_vectorizer.pkl: {e}")

try:
    df = pd.read_csv(csv_path)
except Exception as e:
    print(f"Warning: Could not load cleaned_news.csv: {e}")

# Your News API Key and URL
NEWS_API_KEY = "4b8cf0c10896486eb505cf15948d9df1"
NEWS_API_URL = "https://newsapi.org/v2/everything"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# SerpAPI key from .env
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

# Load local model and vectorizer
try:
    with open(model_path, "rb") as f:
        local_model = pickle.load(f)
except Exception as e:
    logger.error(f"Could not load logistic_model.pkl: {e}")
    local_model = None

try:
    with open(vectorizer_path, "rb") as f:
        local_vectorizer = pickle.load(f)
except Exception as e:
    logger.error(f"Could not load tfidf_vectorizer.pkl: {e}")
    local_vectorizer = None

try:
    local_df = pd.read_csv(csv_path)
except Exception as e:
    logger.error(f"Could not load cleaned_news.csv: {e}")
    local_df = None

@app.route("/")
def index():
    return redirect(url_for("home"))

@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/real_time_search", methods=["GET"])
def real_time_search():
    return render_template("real_time_search.html")

@app.route('/dataset_analysis')
def dataset_analysis():
    return render_template("dataset_analysis.html")

@app.route('/health')
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/.well-known/appspecific/com.chrome.devtools.json")
def ignore_devtools_request():
    return "", 204  # 204 = No Content

def search_news(query, num_results=5):
    """Search news using SerpAPI."""
    try:
        params = {
            "engine": "google",
            "q": query,
            "api_key": SERPAPI_KEY,
            "num": num_results,
            "tbm": "nws"
        }
        response = requests.get("https://serpapi.com/search", params=params)
        response.raise_for_status()
        data = response.json()
        articles = []
        if "news_results" in data:
            for item in data["news_results"][:num_results]:
                articles.append({
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "source": item.get("source", "")
                })
        return articles
    except Exception as e:
        logger.error(f"SerpAPI error: {str(e)}")
        return []

def analyze_with_serpapi(news_text, related_articles):
    """
    Analyze the news based on presence of similar news articles.
    """
    if not related_articles:
        return {
            "result": "Fake",
            "confidence": "Low",
            "justification": "No similar news articles found from reputable sources."
        }

    reputable_sources = ["bbc", "cnn", "reuters", "the guardian", "ndtv", "times of india", "hindustan times", "the hindu"]
    credible_count = sum(1 for article in related_articles if any(source.lower() in article['source'].lower() for source in reputable_sources))

    if credible_count >= 2:
        return {
            "result": "Real",
            "confidence": "High",
            "justification": f"{credible_count} articles found from credible sources."
        }
    else:
        return {
            "result": "Possibly Fake",
            "confidence": "Medium",
            "justification": f"Only {credible_count} credible article(s) found. Needs manual verification."
        }

@app.route('/predict_dataset', methods=['POST'])
def predict_dataset():
    try:
        data = request.get_json()
        if not data or 'news' not in data:
            return jsonify({"error": "Missing 'news' field in request"}), 400
        news_text = data['news'].strip()
        if not news_text:
            return jsonify({"error": "Empty news text provided"}), 400

        if local_df is None:
            return jsonify({"error": "Dataset not available for analysis."}), 500

        # Search for similar articles in the dataset (partial match)
        similar_articles_df = local_df[
            local_df['text'].str.contains(news_text, case=False, na=False) |
            local_df['title'].str.contains(news_text, case=False, na=False)
        ]

        if not similar_articles_df.empty:
            # If matches are found, use the first one for the main prediction
            # and return all matches as related articles.
            first_match = similar_articles_df.iloc[0]
            label = first_match['label']
            result = "Real" if label == 1 else "Fake"
            confidence = "High (Dataset Match)"
            justification = "Similar article(s) found in the dataset. The prediction is based on the most relevant match."
            
            # Prepare articles for the response, creating a snippet from the text
            related_articles = []
            for _, row in similar_articles_df.iterrows():
                text_content = row.get("text")
                snippet = ""
                # Ensure text_content is a valid string before creating a snippet
                if isinstance(text_content, str):
                    snippet = text_content[:200] + "..." if len(text_content) > 200 else text_content
                
                related_articles.append({
                    "title": row.get("title", "No Title"),
                    "snippet": snippet,
                    "source": "Local Dataset", # Identify the source
                    "url": "#" # No URL for local articles
                })
        elif local_model is not None and local_vectorizer is not None:
            # If no dataset matches, fall back to the ML model
            X = local_vectorizer.transform([news_text])
            pred = local_model.predict(X)[0]
            proba = local_model.predict_proba(X)[0]
            result = "Real" if pred == 1 else "Fake"
            confidence = f"{max(proba)*100:.1f}%"
            justification = "Prediction based on trained model, as no direct matches were found in the dataset."
            related_articles = []
            # Add a source flag to indicate this was a model prediction
            prediction_source = "model"
        else:
            return jsonify({"error": "Model or vectorizer not loaded."}), 500

        response = {
            "result": result,
            "confidence": confidence,
            "justification": justification,
            "related_articles": related_articles,
            "source": prediction_source if 'prediction_source' in locals() else "dataset"
        }
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in /predict_dataset: {str(e)}")
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500

@app.route('/predict_serpapi', methods=['POST'])
def predict_serpapi():
    try:
        if not SERPAPI_KEY:
            logger.error("SERPAPI_KEY not configured.")
            return jsonify({"error": "The real-time analysis feature is not configured on the server. Please set the SERPAPI_KEY."}), 500
            
        data = request.get_json()
        if not data or 'news' not in data:
            return jsonify({"error": "Missing 'news' field in request"}), 400

        news_text = data['news']
        search_query = ' '.join(news_text.split('.')[:2])  # Take first 2 sentences as query
        related_articles = search_news(search_query)

        analysis = analyze_with_serpapi(news_text, related_articles)

        response = {
            "result": analysis["result"],
            "confidence": analysis["confidence"],
            "justification": analysis["justification"],
            "related_articles": related_articles
        }
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in /predict_serpapi: {str(e)}")
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
