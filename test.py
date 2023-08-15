import yfinance as yf
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from tqdm import tqdm
from flask import Flask, request, jsonify
import re


# Fetching stock data
def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.Ticker(ticker)
    df = stock_data.history(period='1d', start=start_date, end=end_date)
    return df


def fetch_news_data(query, from_date, to_date, api_key):
    url = f"https://newsapi.org/v2/everything?q={query}&from={from_date}&to={to_date}&apiKey={api_key}"
    response = requests.get(url)
    return response.json()


def fetch_earnings_data(ticker):
    stock_data = yf.Ticker(ticker)
    earnings_data = stock_data.earnings
    return earnings_data


def fetch_stock_price_data(ticker, start_date, end_date):
    stock_data = yf.Ticker(ticker)
    price_data = stock_data.history(period="1d", start=start_date, end=end_date)
    return price_data


def combine_data(news_data, earnings_data, stock_price_data):
    # Merge news data with stock price data on date
    combined_data = pd.merge(news_data, stock_price_data, left_on='date', right_on='Date', how='inner')

    # Merge earnings data with the combined data on date
    combined_data = pd.merge(combined_data, earnings_data, left_on='date', right_on='Date', how='inner')

    # Handle missing data, duplicate columns, etc.
    # ...

    return combined_data


def feature_engineering(combined_data):
    vectorizer = TfidfVectorizer(max_features=1000)  # Limit to 1000 features for simplicity
    tfidf_features = vectorizer.fit_transform(combined_data['news_text'])

    # Add TF-IDF features to combined_data
    combined_data = combined_data.join(pd.DataFrame(tfidf_features.toarray()))

    # Standardize or normalize other numerical features
    # ...

    return combined_data


class StockPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(StockPredictor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions


# Data fetching and preprocessing
ticker = "AAPL"
start_date = "2022-01-01"
end_date = "2022-12-31"
stock_data = fetch_stock_data(ticker, start_date, end_date)

query = "Apple Stock"
from_date = "2022-01-01"
to_date = "2022-12-31"
api_key = "762cf6f371814e9da77f69bab355802e"
news_data = fetch_news_data(query, from_date, to_date, api_key)

earnings_data = fetch_earnings_data(ticker)
stock_price_data = fetch_stock_price_data(ticker, start_date, end_date)

# Assuming news_data is a DataFrame containing columns 'date' and 'news_text'
news_df = pd.DataFrame(news_data['articles'])
news_df['date'] = pd.to_datetime([article['publishedAt'] for article in news_data['articles']])
news_df['news_text'] = [article['content'] for article in news_data['articles']]

# Combine and preprocess data
combined_data = combine_data(news_df, earnings_data, stock_price_data)
final_data = feature_engineering(combined_data)
X = final_data.drop(columns=['target'])  # Adjust according to your data
y = final_data['target']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)


# Model Training
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train.values, dtype=torch.float32))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val.values, dtype=torch.float32))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model = StockPredictor(input_dim=X_train.shape[1], hidden_dim=50)  # Adjust hidden_dim as needed

loss_func = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=0.01)

n_epochs = 5
for epoch in range(n_epochs):
    loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
    for batch_idx, (X_batch, y_batch) in loop:
        y_pred = model(X_batch)
        loss = loss_func(y_pred, y_batch.unsqueeze(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_description(f"Epoch [{epoch+1}/{n_epochs}]")
        loop.set_postfix(loss=loss.item())

print("Training complete!")


def answer_question(question, data, model):
    if "What is the predicted stock price" in question:
        # Use the predictive model to forecast stock price
        price_prediction = model.predict(data)
        return f"The predicted stock price is {price_prediction}."

    elif "What was the news on" in question:
        # Extract date and provide relevant news
        date = extract_date_from_question(question)
        news = get_news_on_date(data, date)
        return f"The news on {date} was: {news}."

    elif "Tell me about the earnings report on" in question:
        # Extract date and provide earnings report
        date = extract_date_from_question(question)
        earnings = get_earnings_on_date(data, date)
        return f"The earnings report on {date} was: {earnings}."

    elif "Show the financial report for" in question:
        # Extract period and provide financial report
        period = extract_period_from_question(question)
        report = get_financial_report(data, period)
        return f"The financial report for {period} is: {report}."

    elif "What's the trend for the past" in question:
        # Extract time frame and provide trend analysis
        time_frame = extract_time_frame_from_question(question)
        trend = analyze_trend(data, time_frame)
        return f"The trend for the past {time_frame} is: {trend}."

    else:
        return "Sorry, I can't answer that question."
    return "Answer to your question"

# Helper to extract date from the question
def extract_date_from_question(question):
    # Assuming date is in YYYY-MM-DD format
    import re
    date_match = re.search(r'\d{4}-\d{2}-\d{2}', question)
    return date_match.group(0) if date_match else None

# Helper to retrieve news for a specific date
def get_news_on_date(data, date):
    # Assuming 'data' is a DataFrame with a 'date' column containing news
    news = data[data['date'] == date]['news'].iloc[0]
    return news

# Helper to retrieve earnings report for a specific date
def get_earnings_on_date(data, date):
    # Assuming 'data' is a DataFrame with a 'date' column containing earnings report
    earnings = data[data['date'] == date]['earnings'].iloc[0]
    return earnings

# Helper to extract reporting period from the question
def extract_period_from_question(question):
    # Assuming period is mentioned like "Q1 2022" or "FY 2021"
    import re
    period_match = re.search(r'(Q\d \d{4})|(FY \d{4})', question)
    return period_match.group(0) if period_match else None

# Helper to retrieve financial report for a specific period
def get_financial_report(data, period):
    # Assuming 'data' is a DataFrame with a 'period' column containing financial report
    report = data[data['period'] == period]['financial_report'].iloc[0]
    return report

# Helper to extract time frame from the question
def extract_time_frame_from_question(question):
    # Assuming time frame is mentioned like "3 months" or "1 year"
    import re
    time_frame_match = re.search(r'\d+ (months?|years?)', question)
    return time_frame_match.group(0) if time_frame_match else None

# Helper to analyze and describe the trend over the specified time frame
def analyze_trend(data, time_frame):
    # Assuming 'data' is a DataFrame with a 'date' and 'price' column
    # You could implement more complex trend analysis here
    import pandas as pd
    duration = int(time_frame.split()[0])
    unit = time_frame.split()[1]
    end_date = pd.to_datetime(data['date'].max())
    start_date = end_date - pd.Timedelta(weeks=4*duration) if "months" in unit else pd.Timedelta(weeks=52*duration)
    trend_data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]
    trend = trend_data['price'].mean() # Example: mean price in the time frame
    return f"Mean price in the past {time_frame} was {trend}."
# Flask API

app = Flask(__name__)

@app.route('/ask', methods=['POST'])
def ask():
    question = request.json['question']
    answer = answer_question(question, final_data, model)
    return jsonify({'answer': answer})


if __name__ == '__main__':
    app.run(debug=True)

