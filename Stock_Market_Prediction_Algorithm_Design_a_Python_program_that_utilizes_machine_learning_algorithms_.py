import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from textblob import TextBlob
import tweepy
import datetime

# Define your API keys for Twitter API
consumer_key = 'YOUR_CONSUMER_KEY'
consumer_secret = 'YOUR_CONSUMER_SECRET'
access_token = 'YOUR_ACCESS_TOKEN'
access_token_secret = 'YOUR_ACCESS_TOKEN_SECRET'

# Authenticate with Twitter API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# Define the company stock symbol
stock_symbol = 'AAPL'

# Define the start and end date for historical data
start_date = datetime.datetime(2010, 1, 1)
end_date = datetime.datetime(2021, 1, 1)

# Fetch historical stock price data using pandas_datareader
df = pd.read_csv(f'https://query1.finance.yahoo.com/v7/finance/download/{stock_symbol}?period1={int(start_date.timestamp())}&period2={int(end_date.timestamp())}&interval=1d&events=history')

# Fetch news sentiment analysis using TextBlob
news_sentiment = []
for tweet in tweepy.Cursor(api.search, q=stock_symbol, lang='en', result_type='recent', tweet_mode='extended').items(100):
    analysis = TextBlob(tweet.full_text)
    news_sentiment.append(analysis.sentiment.polarity)

# Fetch economic indicators data from a CSV file
economic_data = pd.read_csv('economic_indicators.csv')

# Fetch social media trends data from a CSV file
social_media_trends = pd.read_csv('social_media_trends.csv')

# Merge all the data sources into a single DataFrame
df['Date'] = pd.to_datetime(df['Date'])
economic_data['Date'] = pd.to_datetime(economic_data['Date'])
social_media_trends['Date'] = pd.to_datetime(social_media_trends['Date'])

df = pd.merge(df, economic_data, on='Date', how='left')
df = pd.merge(df, social_media_trends, on='Date', how='left')

# Preprocess the data
df['Volume'] = df['Volume'].astype(float)
df['News Sentiment'] = news_sentiment
df.fillna(0, inplace=True)

# Split the data into training and testing sets
X = df.drop(['Date', 'Close'], axis=1)
y = df['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scale the features using MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train and evaluate the Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_predictions = lr_model.predict(X_test_scaled)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_predictions))

# Train and evaluate the Random Forest Regression model
rf_model = RandomForestRegressor(n_estimators=100, random_state=0)
rf_model.fit(X_train_scaled, y_train)
rf_predictions = rf_model.predict(X_test_scaled)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))

# Train and evaluate the Support Vector Regression model
svr_model = SVR(kernel='linear')
svr_model.fit(X_train_scaled, y_train)
svr_predictions = svr_model.predict(X_test_scaled)
svr_rmse = np.sqrt(mean_squared_error(y_test, svr_predictions))

# Compare the performance of the models
print(f"Linear Regression RMSE: {lr_rmse}")
print(f"Random Forest RMSE: {rf_rmse}")
print(f"Support Vector Regression RMSE: {svr_rmse}")