"""from flask import Flask, request, jsonify
from flask_login import UserMixin
from peewee import *
import datetime
import jwt
import os
import yfinance as yf
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from werkzeug.security import generate_password_hash, check_password_hash
from textblob import TextBlob
import tweepy
import secrets

app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(32)

# SQLite Database Connection
DATABASE = SqliteDatabase('stock_app.db')

# User Model
class User(UserMixin, Model):
    username = CharField(unique=True)
    email = CharField(unique=True)
    password = CharField()

    def hash_password(self, password):
        self.password = generate_password_hash(password)

    def verify_password(self, password):
        return check_password_hash(self.password, password)

    def generate_auth_token(self, expires_in=600):
        return jwt.encode({'id': self.id, 'exp': datetime.datetime.utcnow() + datetime.timedelta(seconds=expires_in)},
                          app.config['SECRET_KEY'], algorithm='HS256')

    class Meta:
        database = DATABASE

# Watchlist Model
class Watchlist(Model):
    user = ForeignKeyField(User, backref='watchlists')
    stock = CharField()

    class Meta:
        database = DATABASE

# Initialize the Database
def initialize():
    DATABASE.connect()
    DATABASE.create_tables([User, Watchlist], safe=True)
    DATABASE.close()

# Stock Price Prediction (ARIMA)
def predict_stock_prices(stock_symbol):
    try:
        data = yf.download(stock_symbol, period='1y')
        close_prices = data['Close'].dropna()

        if len(close_prices) < 10:  # Ensuring enough data points
            return "Not enough data for prediction"

        model = ARIMA(close_prices, order=(5, 1, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=7)
        return forecast.tolist()
    except Exception as e:
        return str(e)

# Twitter Sentiment Analysis
# Twitter Sentiment Analysis
def get_twitter_sentiment(stock_symbol):
    try:
        # Replace with your actual credentials
        bearer_token = "AAAAAAAAAAAAAAAAAAAAAIINzQEAAAAAc9Z2v0qN3Jy1B%2FsmJlDrdrQfqSU%3D53dDa5hDThDSqo5aGYEC9EhPT8158Z3vPLXYZWMo9jTeGSPM7j"

        client = tweepy.Client(bearer_token=bearer_token)

        query = f"{stock_symbol} -is:retweet lang:en"
        tweets = client.search_recent_tweets(query=query, max_results=50, tweet_fields=["text"])

        if not tweets or not tweets.data:
            return "No tweets found"

        sentiment_score = sum(TextBlob(tweet.text).sentiment.polarity for tweet in tweets.data) / len(tweets.data)

        return "Positive" if sentiment_score > 0 else "Negative"
    except tweepy.TweepyException as e:
        return f"Twitter API error: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"


@app.route('/predict/<string:stock_symbol>', methods=['GET'])
def predict(stock_symbol):
    forecast = predict_stock_prices(stock_symbol)
    return jsonify({'stock_symbol': stock_symbol, 'forecast': forecast})

@app.route('/sentiment/<string:stock_symbol>', methods=['GET'])
def sentiment(stock_symbol):
    sentiment = get_twitter_sentiment(stock_symbol)
    return jsonify({'stock_symbol': stock_symbol, 'sentiment': sentiment})

@app.route('/recommend/<string:stock_symbol>', methods=['GET'])
def recommend(stock_symbol):
    forecast = predict_stock_prices(stock_symbol)
    sentiment = get_twitter_sentiment(stock_symbol)

    if isinstance(forecast, list) and sentiment == "Positive" and forecast[-1] > forecast[0]:
        decision = "Buy"
    else:
        decision = "Sell"

    return jsonify({'stock_symbol': stock_symbol, 'decision': decision})

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    if not data.get('username') or not data.get('password') or not data.get('email'):
        return jsonify({'message': 'Missing credentials'}), 400

    existing_user = User.get_or_none(User.username == data['username'])
    if existing_user:
        return jsonify({'message': 'User already exists'}), 400

    user = User.create(username=data['username'], email=data['email'], password=generate_password_hash(data['password']))
    return jsonify({'message': 'User registered successfully'}), 201

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    user = User.get_or_none(User.username == data['username'])

    if user and check_password_hash(user.password, data['password']):
        token = user.generate_auth_token()
        return jsonify({'token': token})

    return jsonify({'message': 'Invalid credentials'}), 401

if __name__ == '__main__':
    initialize()
    app.run(debug=True)

from flask import Flask, render_template, request, redirect, url_for, jsonify, session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from peewee import *
import datetime
import jwt
import os
import yfinance as yf
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from werkzeug.security import generate_password_hash, check_password_hash
from textblob import TextBlob
import tweepy
import secrets

app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(32)

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# SQLite Database Connection
DATABASE = SqliteDatabase('stock_app.db')

# User Model
class User(UserMixin, Model):
    username = CharField(unique=True)
    email = CharField(unique=True)
    password = CharField()

    def hash_password(self, password):
        self.password = generate_password_hash(password)

    def verify_password(self, password):
        return check_password_hash(self.password, password)

    def generate_auth_token(self, expires_in=600):
        return jwt.encode({'id': self.id, 'exp': datetime.datetime.utcnow() + datetime.timedelta(seconds=expires_in)},
                          app.config['SECRET_KEY'], algorithm='HS256')

    class Meta:
        database = DATABASE

# Watchlist Model
class Watchlist(Model):
    user = ForeignKeyField(User, backref='watchlists')
    stock = CharField()

    class Meta:
        database = DATABASE

# Initialize the Database
def initialize():
    DATABASE.connect()
    DATABASE.create_tables([User, Watchlist], safe=True)
    DATABASE.close()

# Flask-Login User Loader
@login_manager.user_loader
def load_user(user_id):
    return User.get_or_none(User.id == user_id)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', user=current_user)

@app.route('/predict/<string:stock_symbol>', methods=['GET'])
def predict(stock_symbol):
    forecast = predict_stock_prices(stock_symbol)
    return jsonify({'stock_symbol': stock_symbol, 'forecast': forecast})

@app.route('/sentiment/<string:stock_symbol>', methods=['GET'])
def sentiment(stock_symbol):
    sentiment = get_twitter_sentiment(stock_symbol)
    return jsonify({'stock_symbol': stock_symbol, 'sentiment': sentiment})

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        data = request.form
        if not data.get('username') or not data.get('password') or not data.get('email'):
            return jsonify({'message': 'Missing credentials'}), 400

        existing_user = User.get_or_none(User.username == data['username'])
        if existing_user:
            return jsonify({'message': 'User already exists'}), 400

        user = User.create(username=data['username'], email=data['email'], password=generate_password_hash(data['password']))
        login_user(user)
        return redirect(url_for('dashboard'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.form
        user = User.get_or_none(User.username == data['username'])

        if user and check_password_hash(user.password, data['password']):
            login_user(user)
            return redirect(url_for('dashboard'))

        return jsonify({'message': 'Invalid credentials'}), 401
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

if __name__ == '__main__':
    initialize()
    app.run(debug=True)
def predict_stock_prices(stock_symbol):
    try:
        # Download stock data for the last year
        stock_data = yf.download(stock_symbol, period="1y")

        # Check if data is empty
        if stock_data.empty:
            return {"error": "No stock data available"}

        # Prepare time series data
        stock_data['Returns'] = stock_data['Close'].pct_change()
        stock_data.dropna(inplace=True)

        # Fit ARIMA model
        model = ARIMA(stock_data['Close'], order=(5,1,0))
        model_fit = model.fit()

        # Forecast the next 5 days
        forecast = model_fit.forecast(steps=5)
        return forecast.tolist()

    except Exception as e:
        return {"error": str(e)}

from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_login import UserMixin, LoginManager, login_user, logout_user, login_required, current_user
from peewee import *
import datetime
import jwt
import secrets
import yfinance as yf
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from werkzeug.security import generate_password_hash, check_password_hash
from textblob import TextBlob
import tweepy

app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(32)

# Database Configuration
DATABASE = SqliteDatabase('database.db')

# Flask-Login Configuration
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User Model
class User(UserMixin, Model):
    username = CharField(unique=True)
    email = CharField(unique=True)
    password = CharField()

    class Meta:
        database = DATABASE

# Watchlist Model
class Watchlist(Model):
    user = ForeignKeyField(User, backref='watchlist')
    stock = CharField()

    class Meta:
        database = DATABASE

# Initialize Database
def initialize():
    DATABASE.connect()
    DATABASE.create_tables([User, Watchlist], safe=True)
    DATABASE.close()

@login_manager.user_loader
def load_user(user_id):
    return User.get_or_none(User.id == user_id)

# Home Page
@app.route('/')
def home():
    return render_template('index.html')

# User Registration
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = generate_password_hash(request.form['password'])

        if User.get_or_none(User.username == username):
            flash('Username already exists!', 'danger')
            return redirect(url_for('register'))

        User.create(username=username, email=email, password=password)
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

# User Login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.get_or_none(User.username == username)

        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('dashboard'))

        flash('Invalid credentials!', 'danger')
        return redirect(url_for('login'))

    return render_template('login.html')

# Dashboard (Authenticated Users)
@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', username=current_user.username)

# Logout
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully!', 'success')
    return redirect(url_for('login'))
@app.route('/predict', methods=['POST'])
def predict_stock_post():
    stock_symbol = request.form.get("stock_symbol")
    if not stock_symbol:
        return jsonify({"error": "Stock symbol is required"}), 400
    return redirect(f"/predict/{stock_symbol}")
@app.route('/predict/<string:stock_symbol>', methods=['GET'])
def predict_stock(stock_symbol):
    try:
        if not stock_symbol:
            return jsonify({"error": "Stock symbol is missing"}), 400

        data = yf.download(stock_symbol, period='1y')
        close_prices = data['Close'].dropna()

        if len(close_prices) < 10:
            return jsonify({"error": "Not enough data for prediction"})

        model = ARIMA(close_prices, order=(5,1,0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=7).tolist()

        return render_template('predict.html', stock=stock_symbol, forecast=forecast)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/sentiment', methods=['GET'])
def sentiment():
    try:
        stock_symbol = request.args.get("stock_symbol")  # Get from query params
        
        if not stock_symbol:
            return jsonify({"error": "Stock symbol is required"}), 400

        bearer_token = "AAAAAAAAAAAAAAAAAAAAAIINzQEAAAAAc9Z2v0qN3Jy1B%2FsmJlDrdrQfqSU%3D53dDa5hDThDSqo5aGYEC9EhPT8158Z3vPLXYZWMo9jTeGSPM7j"
        client = tweepy.Client(bearer_token=bearer_token)

        query = f"{stock_symbol} -is:retweet lang:en"
        tweets = client.search_recent_tweets(query=query, max_results=50, tweet_fields=["text"])

        if not tweets or not tweets.data:
            return render_template('sentiment.html', stock=stock_symbol, sentiment="No tweets found")

        sentiment_score = sum(TextBlob(tweet.text).sentiment.polarity for tweet in tweets.data) / len(tweets.data)

        sentiment_result = "Positive" if sentiment_score > 0 else "Negative"
        return render_template('sentiment.html', stock=stock_symbol, sentiment=sentiment_result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    initialize()
    app.run(debug=True)

from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from peewee import *
import datetime
import jwt
import os
import yfinance as yf
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from werkzeug.security import generate_password_hash, check_password_hash
from textblob import TextBlob
import tweepy
import secrets
from flask import Flask, request, render_template, jsonify
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import matplotlib
matplotlib.use('Agg')
app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(32)

# Initialize Flask-Login
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

# Stock Prediction (ARIMA)
def predict_stock_prices(stock_symbol):
    try:
        data = yf.download(stock_symbol, period='1y')
        close_prices = data['Close'].dropna()

        if len(close_prices) < 10:
            return "Not enough data for prediction"

        model = ARIMA(close_prices, order=(5, 1, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=7)
        return [float(val) for val in forecast]
    except Exception as e:
        return str(e)

# Twitter Sentiment Analysis
def get_twitter_sentiment(stock_symbol):
    try:
        bearer_token = "YOUR_TWITTER_BEARER_TOKEN"
        client = tweepy.Client(bearer_token=bearer_token)

        query = f"{stock_symbol} -is:retweet lang:en"
        tweets = client.search_recent_tweets(query=query, max_results=50, tweet_fields=["text"])

        if not tweets or not tweets.data:
            return "Neutral"

        sentiment_score = sum(TextBlob(tweet.text).sentiment.polarity for tweet in tweets.data) / len(tweets.data)

        if sentiment_score > 0:
            return "Positive"
        elif sentiment_score < 0:
            return "Negative"
        else:
            return "Neutral"

    except tweepy.TweepyException as e:
        return f"Twitter API error: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/predict', methods=['GET', 'POST'])  
def predict_stock_post():
    stock_symbol = request.args.get("stock_symbol")  # Use `args` for GET requests
    if not stock_symbol:
        return jsonify({"error": "Stock symbol is required"}), 400
    return redirect(f"/predict/{stock_symbol}")


# Stock Prediction Route
@app.route('/predict/<string:stock_symbol>', methods=['GET'])
def predict_stock(stock_symbol):
    try:
        if not stock_symbol:
            return jsonify({"error": "Stock symbol is missing"}), 400

        forecast = predict_stock_prices(stock_symbol)
        if isinstance(forecast, str):
            return jsonify({"error": forecast})

        return render_template('predict.html', stock=stock_symbol, forecast=forecast)
    except Exception as e:
        return jsonify({"error": str(e)})

# Sentiment Analysis Route
@app.route('/sentiment/<string:stock_symbol>', methods=['GET'])
def sentiment(stock_symbol):
    sentiment = get_twitter_sentiment(stock_symbol)
    return jsonify({'stock_symbol': stock_symbol, 'sentiment': sentiment})

# Stock Recommendation Route
@app.route('/recommend/<string:stock_symbol>', methods=['GET'])
def recommend(stock_symbol):
    forecast = predict_stock_prices(stock_symbol)
    sentiment = get_twitter_sentiment(stock_symbol)

    if isinstance(forecast, list) and sentiment == "Positive" and forecast[-1] > forecast[0]:
        decision = "Buy"
    else:
        decision = "Sell"

    return jsonify({'stock_symbol': stock_symbol, 'decision': decision})

# Register Route
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

# Login Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.get_or_none(User.username == username)

        if user and check_password_hash(user.password, password):
            login_user(user)  # ✅ Correctly logs in the user
            session['user_id'] = user.id  # Store user ID in session
            return redirect(url_for('dashboard'))

        return "Invalid Credentials", 401
    return render_template('login.html')

# Dashboard Route (Protected)
@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
@app.route('/dashboard')
def dashboard():
# Ensure a response is returned
    return render_template('dashboard.html')


@app.route('/history', methods=['GET'])
def get_stock_history():
    stock_symbol = request.args.get("stock_symbol")
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")

    if not stock_symbol or not start_date or not end_date:
        return jsonify({"error": "Stock symbol, start date, and end date are required"}), 400

    try:
        stock = yf.download(stock_symbol, start=start_date, end=end_date)
        stock.reset_index(inplace=True)

        # Flatten multi-index column names
        stock.columns = [col[0] if isinstance(col, tuple) else col for col in stock.columns]

        # Convert DataFrame to list of dictionaries for Jinja
        stock_data = stock.to_dict(orient="records")        

        # Generate a plot for the stock's historical data
        img = generate_plot(stock_data, stock_symbol)

        return render_template("dashboard.html", stock_data=stock_data, stock_symbol=stock_symbol, graph_image=img)

    except Exception as e:
        return jsonify({"error": str(e)})

def generate_plot(df, symbol):
    Generates a line graph for Close prices 
    plt.figure(figsize=(10, 5))
    plt.plot(df['Date'], df['Close'], marker='o', linestyle='-', label="Closing Price", color='blue')
    plt.xlabel("Date")
    plt.ylabel("Close Price (USD)")
    plt.title(f"Price Trend for {symbol}")
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid()

    # Save plot to memory
    img = io.BytesIO()
    plt.savefig(img, format="png", bbox_inches="tight")
    img.seek(0)
    plt.close()

    # Convert to base64 string for HTML embedding
    return base64.b64encode(img.getvalue()).decode()


# Logout Route
@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.clear()  # Clear session to fully log out
    flash('Logged out successfully!', 'success')
    return redirect(url_for('login'))

# Run the App
if __name__ == '__main__':
    initialize()
    app.run(debug=True)
    
    """
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from peewee import *
import datetime
import jwt
import os
import yfinance as yf
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from werkzeug.security import generate_password_hash, check_password_hash
from textblob import TextBlob
import tweepy
import secrets
import matplotlib.pyplot as plt
import io
import base64
import matplotlib
from io import BytesIO


matplotlib.use('Agg')  # Use Agg backend for non-GUI environments

# Initialize Flask App
app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(32)

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


# Stock Prediction (ARIMA)
def predict_stock_prices(stock_symbol):
    try:
        data = yf.download(stock_symbol, period='1y')
        close_prices = data['Close'].dropna()

        if len(close_prices) < 10:
            return "Not enough data for prediction"

        model = ARIMA(close_prices, order=(5, 1, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=7)
        return [float(val) for val in forecast]
    except Exception as e:
        return str(e)


import tweepy
import time
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def get_twitter_sentiment(stock_symbol):
    try:
        bearer_token = "AAAAAAAAAAAAAAAAAAAAAIINzQEAAAAAc9Z2v0qN3Jy1B%2FsmJlDrdrQfqSU%3D53dDa5hDThDSqo5aGYEC9EhPT8158Z3vPLXYZWMo9jTeGSPM7j"  # Replace with your actual token
        client = tweepy.Client(bearer_token=bearer_token)

        query = f"{stock_symbol} OR #{stock_symbol} OR ${stock_symbol} -is:retweet lang:en"
        tweets = None  # Initialize to avoid errors

        for _ in range(3):  # Retry mechanism (up to 3 attempts)
            try:
                tweets = client.search_recent_tweets(query=query, max_results=50, tweet_fields=["text", "created_at"])
                break  # Exit loop if request succeeds
            except tweepy.TooManyRequests:
                print("Rate limit exceeded. Retrying after 15 seconds...")
                time.sleep(15)  # Wait before retrying
            except Exception as e:
                print(f"Twitter API error: {str(e)}")
                return [], f"Twitter API error: {str(e)}"  # ✅ Returns TWO values

        if tweets is None or tweets.data is None:
            print(f"No tweets found for: {stock_symbol}")
            return [], "Neutral"  # ✅ Returns TWO values

        # Use VADER for sentiment analysis
        analyzer = SentimentIntensityAnalyzer()
        sentiment_score = sum(analyzer.polarity_scores(tweet.text)['compound'] for tweet in tweets.data) / len(tweets.data)

        # Get top 3 tweets
        top_tweets = [{"text": tweet.text, "created_at": tweet.created_at} for tweet in tweets.data[:3]]

        # Determine sentiment
        if sentiment_score > 0.05:
            sentiment = "Positive"
        elif sentiment_score < -0.05:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        return top_tweets, sentiment  # ✅ Always returning TWO values

    except tweepy.TooManyRequests:
        return [], "Error: Twitter API rate limit exceeded. Try again later."  # ✅ Fixed return format
    except Exception as e:
        return [], f"Error: {str(e)}"  # ✅ Fixed return format


import requests    

    
def get_news_articles(stock_symbol):
    try:
        # News API setup
        news_api_key = "7383e85835cf462bae4a8d1eac21d536"  # Replace with your NewsAPI key
        url = f"https://newsapi.org/v2/everything?q={stock_symbol}&apiKey={news_api_key}&pageSize=3"
        response = requests.get(url)
        articles = response.json()

        if articles['status'] != 'ok' or 'articles' not in articles:
            return [], "Error: Unable to fetch news articles."

        top_articles = [{
            "title": article['title'],
            "description": article['description'],
            "url": article['url'],
            "publishedAt": article['publishedAt']
        } for article in articles['articles'][:3]]

        return top_articles, None  # Ensure two values are returned

    except Exception as e:
        return [], f"Error: {str(e)}"



@app.route('/predict', methods=['GET', 'POST'])  
def predict_stock_post():
    stock_symbol = request.args.get("stock_symbol")  # Use `args` for GET requests
    if not stock_symbol:
        return jsonify({"error": "Stock symbol is required"}), 400
    return redirect(f"/predict/{stock_symbol}")

# Stock Prediction Route
@app.route('/predict/<string:stock_symbol>', methods=['GET'])
def predict_stock(stock_symbol):
    try:
        forecast = predict_stock_prices(stock_symbol)
        if isinstance(forecast, str):
            return jsonify({"error": forecast})

        return render_template('predict.html', stock=stock_symbol, forecast=forecast)
    except Exception as e:
        print("Error fetching data:", e)
        time.sleep(20)  # Wait before retrying
        return predict_stock(stock_symbol)
    

@app.route('/sentiment', methods=['GET'])
def sentiment():
    stock_symbol = request.args.get('stock_symbol')

    if not stock_symbol:
        return jsonify({'error': 'Missing stock_symbol parameter'}), 400

    # Get tweets and sentiment
    top_tweets, sentiment_result = get_twitter_sentiment(stock_symbol)

    # Get news articles
    top_articles, news_error = get_news_articles(stock_symbol)

    if (sentiment_result is not None and "Error" in sentiment_result) or (news_error is not None and "Error" in news_error):
        # Handle the error case
        return "An error occurred with sentiment or news data."

    return render_template('sentiment.html', 
                           stock_symbol=stock_symbol, 
                           sentiment=sentiment_result, 
                           top_tweets=top_tweets,
                           top_articles=top_articles)



@app.route('/recommend', methods=['GET'])
def recommend():
    stock_symbol = request.args.get('stock_symbol')
    
    if not stock_symbol:
        return render_template('recommend.html', error="stock_symbol is required")
    
    # Get stock forecast
    forecast = predict_stock_prices(stock_symbol)
    
    if not isinstance(forecast, list):
        return render_template('recommend.html', error=forecast)
    
    # Get sentiment
    top_tweets, sentiment = get_twitter_sentiment(stock_symbol)
    
    if sentiment == "Error":
        return render_template('recommend.html', error="Unable to fetch sentiment data")
    
    # Make recommendation based on forecast and sentiment
    recommendation = "Buy" if sentiment == "Positive" and forecast[-1] > forecast[0] else "Sell"

    return render_template('recommend.html', 
                           stock_symbol=stock_symbol, 
                           forecast=forecast, 
                           sentiment=sentiment, 
                           recommendation=recommendation)




# Register Route
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


# Login Route
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


# Dashboard Route (Protected)
@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    historical_data = None
    if request.method == 'POST':
        stock_symbol = request.form['stock_symbol']
        start_date = request.form['start_date']
        end_date = request.form['end_date']

        if stock_symbol and start_date and end_date:
            try:
                data = yf.download(stock_symbol, start=start_date, end=end_date)
                if data.empty:
                    flash(f"No data found for {stock_symbol} in the given range.", 'danger')
                historical_data = data[['Close']].reset_index().to_dict(orient='records')
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
        stock = yf.download(stock_symbol, start=start_date, end=end_date)
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

# Logout Route
@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.clear()
    flash('Logged out successfully!', 'success')
    return redirect(url_for('login'))


# Run the App
if __name__ == '__main__':
    initialize()
    app.run(debug=True)
