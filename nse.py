from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from alpha_vantage.timeseries import TimeSeries
import matplotlib.pyplot as plt
import base64
from datetime import datetime, timedelta
from io import BytesIO
import pandas as pd
import matplotlib
import secrets
from peewee import *
matplotlib.use('Agg')
from statsmodels.tsa.arima.model import ARIMA
app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(32)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error


# Flask-Login Manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# SQLite Database Connection
DATABASE = SqliteDatabase('stock_app.db')


# User Model
class User(UserMixin, Model):
    id = AutoField()
    username = CharField(unique=True)
    email = CharField(unique=True)
    password = CharField()

    def verify_password(self, password):
        return check_password_hash(self.password, password)

    class Meta:
        database = DATABASE


# Watchlist Model
class Watchlist(Model):
    id = AutoField()
    user = ForeignKeyField(User, backref='watchlists')
    stock = CharField()

    class Meta:
        database = DATABASE


# Flask-Login: Load user function
@login_manager.user_loader
def load_user(user_id):
    return User.get_or_none(User.id == int(user_id))


# Initialize the Database
def initialize():
    DATABASE.connect()
    DATABASE.create_tables([User, Watchlist], safe=True)
    DATABASE.close()


# Home Route
@app.route('/')
def home():
    return render_template('index.html')


api_key = "FPRNBLEBOJ8KY7BC"  # Replace with your Alpha Vantage API key

from datetime import datetime
import pandas as pd
from alpha_vantage.timeseries import TimeSeries

def get_stock_data(symbol, start_date=None, end_date=None):
    ts = TimeSeries(key=api_key, output_format="pandas")
    data, meta_data = ts.get_daily(symbol=symbol, outputsize="full")  # Fetch full data

    # Convert index to datetime for filtering
    data.index = pd.to_datetime(data.index)

    # Set default start_date and end_date
    if start_date is None:
        start_date = pd.to_datetime("2020-03-13")
    else:
        start_date = pd.to_datetime(start_date)
    
    if end_date is None:
        end_date = pd.to_datetime(datetime.today().strftime("%Y-%m-%d"))
    else:
        end_date = pd.to_datetime(end_date)

    # Debugging: Check available date range
    print(f"Available Date Range: {data.index.min()} to {data.index.max()}")
    print(f"Filtering Data Between: {start_date} and {end_date}")

    # Ensure the requested dates exist in the dataset
    if start_date not in data.index or end_date not in data.index:
        print("Error: Some requested dates are missing from the dataset.")

    data = data[(data.index >= start_date) & (data.index <= end_date)] 
    data = data.reset_index()  
    data.rename(columns={'date': 'Date'}, inplace=True)
    data.rename(columns={
        '1. open': 'Open',
        '2. high': 'High',
        '3. low': 'Low',
        '4. close': 'Close',
        '5. volume': 'Volume'
    }, inplace=True)

    return data
from datetime import date
import time
import tweepy
from textblob import TextBlob
from tensorflow.keras.callbacks import EarlyStopping
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from textblob import TextBlob

def get_twitter_sentiment(stock_symbol, num_tweets=10):
    try:
        url = f"https://twitter.com/search?q={stock_symbol}&f=live"
        
        options = Options()
        options.add_argument("--headless")  # Run in background
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")

        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        driver.get(url)
        time.sleep(5)  # Wait for page to load

        for _ in range(3):  # Scroll to load more tweets
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(3)

        soup = BeautifulSoup(driver.page_source, "html.parser")
        driver.quit()  # Close the browser

        tweets = []
        tweet_elements = soup.find_all("div", {"data-testid": "tweetText"})

        for elem in tweet_elements[:num_tweets]:
            tweets.append({"text": elem.get_text(), "created_at": "Live"})

        if not tweets:
            return [], "Neutral"  # No tweets found

        # Calculate sentiment score
        sentiment_score = sum(TextBlob(tweet["text"]).sentiment.polarity for tweet in tweets) / len(tweets)

        # Determine sentiment
        sentiment = "Positive" if sentiment_score > 0 else "Negative" if sentiment_score < 0 else "Neutral"

        return tweets[:3], sentiment  # Return top 3 tweets and sentiment

    except Exception as e:
        return [], f"Error: {str(e)}"
import requests
def get_news_articles(stock_symbol):
    try:
        # News API setup
        news_api_key = "7383e85835cf462bae4a8d1eac21d536"  # Replace with your valid API key
        url = f"https://newsapi.org/v2/everything?q={stock_symbol}&apiKey={news_api_key}&pageSize=3"
        response = requests.get(url)
        articles = response.json()

        if articles.get('status') != 'ok' or 'articles' not in articles:
            return [], 0.0  # Ensure the function returns a numeric sentiment

        # Extract top articles
        top_articles = [{
            "title": article.get('title', ''),
            "description": article.get('description', ''),
            "url": article.get('url', ''),
            "publishedAt": article.get('publishedAt', '')
        } for article in articles.get('articles', [])[:3]]

        # Compute sentiment for the article texts
        article_texts = [f"{article['title']} {article['description']}" for article in top_articles if article['title'] and article['description']]
        
        # Default to 0.0 if no valid articles
        if not article_texts:
            return top_articles, 0.0

        # Compute average sentiment
        news_sentiment = sum(TextBlob(text).sentiment.polarity for text in article_texts) / len(article_texts)

        return top_articles, float(news_sentiment)  # Explicitly return float

    except Exception as e:
        return [], 0.0  # Ensure the function always returns a float
@app.route('/recommend', methods=['GET'])
def recommend():
    stock_symbol = request.args.get('stock_symbol')
    
    if not stock_symbol:
        return render_template('recommend.html', error="Stock symbol is required")
    
    # Get stock forecast
    forecast_data = predict_stock_prices(stock_symbol)

    if "error" in forecast_data:
        return render_template('recommend.html', error=forecast_data["error"])

    forecast = forecast_data["forecast"]  # Extracting forecast list

    # Get sentiment
    top_tweets, sentiment = get_twitter_sentiment(stock_symbol)
    
    if not sentiment or sentiment == "Error":
        return render_template('recommend.html', error="Unable to fetch sentiment data")

    # Make recommendation based on forecast and sentiment
    if sentiment == "Positive" and forecast[-1] > forecast[0]:
        recommendation = "Buy"
    elif sentiment == "Negative" or forecast[-1] < forecast[0]:
        recommendation = "Sell"
    else:
        recommendation = "Hold"

    return render_template('recommend.html', 
                           stock_symbol=stock_symbol, 
                           forecast=forecast, 
                           sentiment=sentiment, 
                           recommendation=recommendation)

    
def prepare_data(data, lookback=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
    
    X, Y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i - lookback:i, 0])
        Y.append(scaled_data[i, 0])
    
    X, Y = np.array(X), np.array(Y)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    return X, Y, scaler

# Function to build LSTM model
def build_lstm_model():
    model = Sequential([
        LSTM(units=100, return_sequences=True, input_shape=(60, 1)),
        Dropout(0.3),
        LSTM(units=100, return_sequences=False),
        Dropout(0.3),
        Dense(units=50, activation="relu"),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# Function to predict stock prices
def predict_stock_prices(stock_symbol):
    try:
        data = get_stock_data(stock_symbol)
        close_prices = data['Close']
        
        if len(close_prices) < 100:
            return {"error": "Not enough data for prediction"}
        
        X, Y, scaler = prepare_data(close_prices)
        X_train, X_test = X[:-200], X[-200:]
        Y_train, Y_test = Y[:-200], Y[-200:]
        
        model = build_lstm_model()
        model.fit(X_train, Y_train, epochs=30, batch_size=32, verbose=0, validation_split=0.2)
        
        predictions_scaled = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions_scaled)
        actual_prices = scaler.inverse_transform(Y_test.reshape(-1, 1))
        
        mape = mean_absolute_error(actual_prices, predictions)
        accuracy = max(0, 100 - mape)
        
        last_60_days = close_prices[-60:].values.reshape(-1, 1)
        last_60_scaled = scaler.transform(last_60_days)
        last_60_scaled = last_60_scaled.reshape(1, last_60_scaled.shape[0], 1)
        
        future_prediction = []
        for _ in range(7):
            next_price_scaled = model.predict(last_60_scaled)
            future_prediction.append(next_price_scaled[0, 0])
            last_60_scaled = np.append(last_60_scaled[:, 1:, :], np.array(next_price_scaled).reshape(1, 1, 1), axis=1)
        
        future_predictions = scaler.inverse_transform(np.array(future_prediction).reshape(-1, 1))
        
        return {
            "forecast": [float(val) for val in future_predictions.flatten()],
            "accuracy": round(accuracy, 2),
            "actual_prices": actual_prices.flatten().tolist(),
            "predictions": predictions.flatten().tolist()
        }
    except Exception as e:
        login.error(f"Prediction error: {str(e)}")
        return {"error": str(e)}

@app.route('/predict', methods=['GET'])
def predict():
    # Get stock symbol from query parameters
    stock_symbol = request.args.get('stock_symbol')
    
    if not stock_symbol:
        return jsonify({"error": "Stock symbol is required"}), 400

    print(f"Received stock symbol: {stock_symbol}")  # Debugging print statement

    # Get prediction results
    result = predict_stock_prices(stock_symbol)

    if "error" in result:
        return jsonify(result), 400

    df_predictions = pd.DataFrame({
        'Date': pd.date_range(start=date.today(), periods=len(result["forecast"])),  # FIXED
        'Predicted Price': result["forecast"]
    })

    return render_template(
        'predict.html',
        stock_symbol=stock_symbol,
        accuracy=result["accuracy"],
        df_predictions=df_predictions.to_html(classes='table table-striped'),
        actual_prices=result["actual_prices"],
        forecast=result["forecast"]
    )
@app.route('/sentiment', methods=['GET'])
def sentiment():
    stock_symbol = request.args.get('stock_symbol')

    if not stock_symbol:
        return jsonify({'error': 'Missing stock_symbol parameter'}), 400

    # Remove exchange suffixes like ".bse", ".nse", ".ns", etc.
    cleaned_symbol = stock_symbol.split('.')[0]

    # Get tweets and sentiment score
    top_tweets, tweet_sentiment = get_twitter_sentiment(cleaned_symbol)

    # Get news articles and sentiment score
    top_articles, news_sentiment = get_news_articles(cleaned_symbol)

    # Ensure tweet_sentiment and news_sentiment are floats
    try:
        tweet_sentiment = float(tweet_sentiment) if isinstance(tweet_sentiment, (int, float)) else 0.0
        news_sentiment = float(news_sentiment) if isinstance(news_sentiment, (int, float)) else 0.0
    except ValueError:
        tweet_sentiment = 0.0
        news_sentiment = 0.0

    # Calculate final sentiment score as the weighted average
    total_weight = len(top_tweets) + len(top_articles)

    if total_weight == 0:
        final_sentiment = "Neutral"
    else:
        overall_score = (tweet_sentiment * len(top_tweets) + news_sentiment * len(top_articles)) / total_weight
        final_sentiment = "Positive" if overall_score > 0 else "Negative" if overall_score < 0 else "Neutral"

    return render_template('sentiment.html', 
                           stock_symbol=stock_symbol, 
                           sentiment=final_sentiment, 
                           top_tweets=top_tweets,
                           top_articles=top_articles)



@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = generate_password_hash(request.form['password'])
        
        if User.get_or_none(User.username == username):
            return "User already exists!", 400
        
        User.create(username=username, email=email, password=password)
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.get_or_none(User.username == username)
        
        if user and check_password_hash(user.password, password):
            login_user(user)
            session['user_id'] = user.id
            return redirect(url_for('dashboard'))
        
        return "Invalid Credentials", 401
    return render_template('login.html')

@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    historical_data = None
    if request.method == 'POST':
        stock_symbol = request.form['stock_symbol']
        
        if stock_symbol:
            try:
                data = get_stock_data(stock_symbol)
                historical_data = data[['4. close']].reset_index().to_dict(orient='records')
            except Exception as e:
                flash(f"Error fetching historical data: {str(e)}", 'danger')
    
    return render_template("dashboard.html", historical_data=historical_data)


    
@app.route("/history", methods=["GET"])
def get_stock_history():
    stock_symbol = request.args.get("stock_symbol")
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")

    if not stock_symbol or not start_date or not end_date:
        return render_template("dashboard.html", stock_data=None, stock_symbol=None, graph_image=None)

    try:
        # Download stock data
        stock = get_stock_data(stock_symbol, start_date, end_date)
        stock.reset_index(inplace=True)

        # Flatten multi-index column names
        stock.columns = [col[0] if isinstance(col, tuple) else col for col in stock.columns]

        # Convert DataFrame to list of dictionaries for Jinja
        stock_data = stock.tail(10).to_dict(orient="records")
        

        print("Stock Data:", stock_data) 
        plt.figure(figsize=(8, 4))
        plt.plot(stock["Date"], stock["Close"], label="Close Price", color="blue")
        plt.xlabel("Date")
        plt.ylabel("Closing Price")
        plt.title(f"Stock Price Trend for {stock_symbol}")
        plt.legend()

        # Save graph to base64
        img = BytesIO()
        plt.savefig(img, format="png")
        img.seek(0)
        graph_image = base64.b64encode(img.getvalue()).decode()
        plt.close()

        return render_template("dashboard.html", stock_data=stock_data, stock_symbol=stock_symbol,graph_image=graph_image)

    
    except Exception as e:
        return f"Error fetching stock data: {e}"


@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.clear()
    flash('Logged out successfully!', 'success')
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)



