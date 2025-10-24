# streamlit_app.py
# Streamlit version of your Finbot app with full custom-module integration
# Run: streamlit run streamlit_app.py

import os
import io
import re
import json
import math
import numpy as np
import pandas as pd
import datetime as dt
from datetime import datetime, timedelta

import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt

# === Your custom modules (now wired in) ===
from QA import answer_Q, last_matched_question, last_matched_answer  # tracks Q/A for feedback
from name_management import name_change, check_name_change, name_response
from small_talk import talk_response, time_response
from spell_check import correct

# ML/metrics (optional heavy)
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D, Flatten

# Sentiment/news
from textblob import TextBlob
import requests

# --- CONFIG / THEME ---
st.set_page_config(page_title="Finbot (Streamlit)", page_icon="💸", layout="wide")

COLORS = {
    'primary': '#4CAF50',
    'secondary': '#2196F3',
    'accent': '#FF9800',
    'success': '#4CAF50',
    'warning': '#FFC107',
    'error': '#F44336',
    'text_primary': 'black',
    'text_secondary': '#757575',
    'background_light': '#FAFAFA',
    'background_medium': '#F5F5F5',
    'background_dark': '#E0E0E0',
    'card_background': '#FFFFFF',
    'colour_black': '#000000'
}

# --- STATE ---
def init_state():
    defaults = dict(
        user_name="(User)",
        user_wallet=0.0,
        user_portfolio={},  # {ticker: {'shares': float, 'avg_price': float}}
        chat_history=[],     # list of dicts: {'role': 'user'|'assistant', 'content': str}
        last_user_query="",
        last_bot_response="",
        literacy="Beginner",
        api_key_news=os.getenv("NEWS_API_KEY", '9491467042934eeb9a7fa58400031b8a'),
        enable_shap=False,
        model_choice="CNN",
        has_switched_level=False
    )
    for k,v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# --- UTIL STORAGE ---
def portfolio_path():
    os.makedirs("portfolio", exist_ok=True)
    return "portfolio/user_portfolio.json"

def feedback_dir():
    os.makedirs("feedback", exist_ok=True)
    return "feedback"

def poor_xlsx_path():
    return os.path.join(feedback_dir(), "poor_table.xlsx")

def feedback_json_path():
    return os.path.join(feedback_dir(), "finbot_feedback.json")

# --- PORTFOLIO LOAD/SAVE ---
def load_portfolio_data():
    path = portfolio_path()
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                data = json.load(f)
            st.session_state.user_wallet = data.get("wallet_balance", 0.0)
            st.session_state.user_portfolio = data.get("stocks", {})
            # legacy normalization
            for t,v in list(st.session_state.user_portfolio.items()):
                if not isinstance(v, dict):
                    price = get_current_stock_price(t)
                    st.session_state.user_portfolio[t] = {
                        "shares": float(v),
                        "avg_price": float(price) if isinstance(price, (int,float)) else 0.0
                    }
        except Exception as e:
            st.toast(f"Error loading portfolio: {e}", icon="⚠️")

def save_portfolio_data():
    data = {
        "wallet_balance": st.session_state.user_wallet,
        "stocks": st.session_state.user_portfolio
    }
    with open(portfolio_path(), "w") as f:
        json.dump(data, f, indent=4)

# Load portfolio once per session
if "portfolio_loaded_once" not in st.session_state:
    load_portfolio_data()
    st.session_state["portfolio_loaded_once"] = True

# --- FINANCE HELPERS ---
@st.cache_data(ttl=300)
def get_current_stock_price(ticker: str):
    try:
        stock = yf.Ticker(ticker)
        todays = stock.history(period="1d")
        price = todays["Close"].iloc[-1]
        return float(price)
    except Exception:
        return None

def add_to_wallet(amount: float):
    try:
        amount = float(amount)
        if amount <= 0:
            return "- Finbot: Please enter a positive amount to add to your wallet."
        st.session_state.user_wallet += amount
        save_portfolio_data()
        return f"- Finbot: ${amount:.2f} has been added to your wallet."
    except Exception:
        return "- Finbot: That doesn't seem to be a valid amount."

def buy_stock(ticker: str, amount: float):
    ticker = ticker.upper().strip()
    price = get_current_stock_price(ticker)
    if not isinstance(price, (int,float)):
        return "- Finbot: Sorry, I couldn't retrieve the current price for that symbol."

    num_shares = float(amount)/float(price)
    if num_shares < 0.01:
        return f"- Finbot: Amount too small. {ticker} price: ${price:.2f}"

    if st.session_state.user_wallet < float(amount):
        return f"- Finbot: Not enough funds. Wallet: ${st.session_state.user_wallet:.2f}"

    st.session_state.user_wallet -= float(amount)
    if ticker in st.session_state.user_portfolio:
        cur = st.session_state.user_portfolio[ticker]
        cur_shares = cur['shares']
        cur_avg = cur['avg_price']
        new_total_value = cur_shares*cur_avg + float(amount)
        new_total_shares = cur_shares + num_shares
        new_avg = new_total_value / new_total_shares
        st.session_state.user_portfolio[ticker] = {'shares': new_total_shares, 'avg_price': new_avg}
    else:
        st.session_state.user_portfolio[ticker] = {'shares': num_shares, 'avg_price': price}

    save_portfolio_data()
    return (f"- Finbot: Purchased {num_shares:.4f} shares of {ticker} at ${price:.2f}. "
            f"Wallet now ${st.session_state.user_wallet:.2f}")

def sell_stock(ticker: str, shares: float|None=None, amount: float|None=None):
    ticker = ticker.upper().strip()
    if ticker not in st.session_state.user_portfolio:
        return f"- Finbot: You don't own any {ticker}."

    price = get_current_stock_price(ticker)
    if not isinstance(price, (int,float)):
        return f"- Finbot: Could not fetch current price for {ticker}."

    owned = st.session_state.user_portfolio[ticker]['shares']
    avg_price = st.session_state.user_portfolio[ticker]['avg_price']

    if shares is not None:
        shares_to_sell = float(shares)
        if shares_to_sell <= 0 or shares_to_sell > owned:
            return f"- Finbot: Invalid share amount. You own {owned:.4f}."
    elif amount is not None:
        amt = float(amount)
        if amt <= 0:
            return "- Finbot: Enter a positive dollar amount."
        shares_to_sell = min(amt/price, owned)
        if shares_to_sell < 0.0001:
            return "- Finbot: Dollar amount too small to sell meaningful shares."
    else:
        shares_to_sell = owned

    sale_value = shares_to_sell*price
    purchase_value = shares_to_sell*avg_price
    realized = sale_value - purchase_value

    if math.isclose(shares_to_sell, owned, rel_tol=1e-9):
        del st.session_state.user_portfolio[ticker]
    else:
        st.session_state.user_portfolio[ticker]['shares'] -= shares_to_sell

    st.session_state.user_wallet += sale_value
    save_portfolio_data()

    res = [
        f"- Finbot: Sold {shares_to_sell:.4f} {ticker} @ ${price:.2f}.",
        f"- Finbot: ${sale_value:.2f} added to wallet (now ${st.session_state.user_wallet:.2f}).",
    ]
    if realized > 0:
        res.append(f"- Finbot: Profit ${realized:.2f}.")
    elif realized < 0:
        res.append(f"- Finbot: Loss ${-realized:.2f}.")
    else:
        res.append(f"- Finbot: Broke even.")
    return "\n".join(res)

def get_portfolio_table():
    rows = []
    for t, d in st.session_state.user_portfolio.items():
        shares = d['shares']
        avg = d['avg_price']
        price = get_current_stock_price(t)
        if isinstance(price, (int,float)):
            cur_val = shares*price
            buy_val = shares*avg
            gain = cur_val - buy_val
            pct = (gain/buy_val*100) if buy_val>0 else 0.0
            rows.append([t, f"{shares:.2f}", f"${avg:.2f}", f"${price:.2f}", f"${cur_val:.2f}", f"${gain:.2f} ({pct:.1f}%)"])
    return pd.DataFrame(rows, columns=["Ticker","Shares","Avg Price","Current Price","Current Value","Gain/Loss"])

# --- FEEDBACK (tracks matched Q/A via QA globals) ---
def save_feedback(user_query, bot_response, rating, comments=None):
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_query": user_query,
        "bot_response": bot_response,
        "rating": rating,
        "comments": comments,
        "financial_literacy_level": st.session_state.literacy
    }
    path = feedback_json_path()
    data = []
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = []
    data.append(entry)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def save_poor_rating_to_excel(user_query, bot_response, comments=None):
    df_new = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_question": user_query,
        "bot_response": bot_response,
        "file_question": last_matched_question if last_matched_question else "No matched question",
        "file_answer": last_matched_answer if last_matched_answer else "No matched answer",
        "source": "question_answering.xlsx" if last_matched_answer else "Unknown",
        "explanation": comments or "",
        "financial_literacy_level": st.session_state.literacy
    }])
    path = poor_xlsx_path()
    if os.path.exists(path):
        try:
            df_old = pd.read_excel(path)
            df = pd.concat([df_old, df_new], ignore_index=True)
        except Exception:
            df = df_new
    else:
        df = df_new
    df.to_excel(path, index=False)

def feedback_stats():
    path = feedback_json_path()
    if not os.path.exists(path):
        return "No feedback data available yet."
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if not data: return "No feedback data available yet."
        total = len(data)
        pos = sum(1 for x in data if x["rating"] in ["Good","Excellent"])
        neu = sum(1 for x in data if x["rating"]=="Neutral")
        neg = sum(1 for x in data if x["rating"] in ["Poor","Very Poor"])
        avg = pos/total*100
        return (f"Feedback Statistics:\n"
                f"Total responses rated: {total}\n"
                f"Positive ratings: {pos} ({pos/total*100:.1f}%)\n"
                f"Neutral ratings: {neu} ({neu/total*100:.1f}%)\n"
                f"Negative ratings: {neg} ({neg/total*100:.1f}%)\n"
                f"Overall satisfaction: {avg:.1f}%")
    except Exception as e:
        return f"Error retrieving feedback statistics: {e}"

# --- NEWS SENTIMENT ---
def fetch_news(api_key, ticker_symbol):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    url = 'https://newsapi.org/v2/everything'
    params = {
        'q': f"{ticker_symbol} AND (stock OR market OR finance)",
        'from': start_date.strftime('%Y-%m-%d'),
        'to': end_date.strftime('%Y-%m-%d'),
        'sortBy': 'relevance',
        'apiKey': api_key
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        articles = r.json().get('articles', [])
        return [{'headline': a['title'], 'date': a['publishedAt'][:10]} for a in articles]
    except Exception:
        return []

def analyze_sentiment(headlines):
    out = []
    for h in headlines:
        s = TextBlob(h['headline']).sentiment.polarity
        out.append({'date': h['date'], 'headline': h['headline'], 'sentiment': s})
    return out

def overall_sentiment(scores):
    if not scores: return 0
    return sum(x['sentiment'] for x in scores)/len(scores)

def advice_from_sentiment(avg):
    if avg > 0.1: return "BUY - Positive sentiment detected."
    if avg < -0.1: return "SELL - Negative sentiment detected."
    return "HOLD - Neutral sentiment."

# --- PREDICTION  ---
def predict_stock_price_with_xai(ticker, model_types=("CNN",), enable_shap=False, progress_cb=None):
    def step(msg):
        if progress_cb:
            progress_cb(msg)

    # Data
    step("Downloading historical data…")
    start = dt.datetime(2012,1,1)
    end = dt.datetime.now()
    data = yf.download(ticker, start=start, end=end)
    if data is None or data.empty:
        return {}, "No data for ticker.", []

    scaler = MinMaxScaler((0,1))
    scaled = scaler.fit_transform(data['Close'].values.reshape(-1,1))
    pred_days = 60
    x_train, y_train = [], []
    for i in range(pred_days, len(scaled)):
        x_train.append(scaled[i-pred_days:i,0])
        y_train.append(scaled[i,0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    models = {}
    results_txt = io.StringIO()

    if "LSTM" in model_types:
        step("Training LSTM…")
        lstm = Sequential([
            LSTM(50, return_sequences=True, input_shape=(x_train.shape[1],1)),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])
        lstm.compile(optimizer='adam', loss='mean_squared_error')
        lstm.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)
        models["LSTM"] = lstm
        print("LSTM trained.", file=results_txt)

    if "CNN" in model_types:
        step("Training CNN…")
        cnn = Sequential([
            Conv1D(64, 2, activation='relu', input_shape=(x_train.shape[1],1)),
            MaxPooling1D(pool_size=2),
            Conv1D(128, 2, activation='relu'),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(50, activation='relu'),
            Dense(1)
        ])
        cnn.compile(optimizer='adam', loss='mean_squared_error')
        cnn.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)
        models["CNN"] = cnn
        print("CNN trained.", file=results_txt)

    # Test set
    step("Preparing test set…")
    test_start = dt.datetime(2022,1,1)
    test_end = dt.datetime.now()
    test = yf.download(ticker, start=test_start, end=test_end)
    if test is None or test.empty:
        return {}, "No recent test data.", []
    actual = test['Close'].values
    total = pd.concat((data['Close'], test['Close']), axis=0)
    model_inputs = total[len(total)-len(test)-pred_days:].values.reshape(-1,1)
    model_inputs = scaler.transform(model_inputs)

    x_test = []
    for i in range(pred_days, len(model_inputs)):
        x_test.append(model_inputs[i-pred_days:i,0])
    x_test = np.array(x_test).reshape(-1, pred_days, 1)

    predictions = {}
    charts = []

    plt.style.use('ggplot')

    for name, mdl in models.items():
        step(f"Predicting ({name})…")
        preds = scaler.inverse_transform(mdl.predict(x_test))
        mae = mean_absolute_error(actual, preds.flatten())
        mse = mean_squared_error(actual, preds.flatten())
        rmse = math.sqrt(mse)
        print(f"{name} MAE=${mae:.2f}, MSE=${mse:.2f}, RMSE=${rmse:.2f}", file=results_txt)

        # chart: actual vs predicted
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(actual, linewidth=2, label="Actual")
        ax.plot(preds, linewidth=2, label="Predicted")
        ax.set_title(f"{ticker} - {name} Actual vs Predicted")
        ax.set_xlabel("Time")
        ax.set_ylabel("Price ($)")
        ax.legend()
        fig.tight_layout()
        charts.append(fig)

        # Next-day prediction
        real_data = np.array([model_inputs[-pred_days:,0]])
        real_data = real_data.reshape((1, pred_days, 1))
        next_price = scaler.inverse_transform(mdl.predict(real_data)).flatten()[0]
        predictions[name] = float(next_price)

    # SHAP deliberately omitted for speed unless you really want to add it back

    return predictions, results_txt.getvalue(), charts

# --- CHAT HELPERS ---
def chat_msg(role, text):
    with st.chat_message("assistant" if role=="assistant" else "user"):
        st.markdown(text)

def push_assistant(text):
    st.session_state.chat_history.append({"role":"assistant","content":text})
    chat_msg("assistant", text)
    st.session_state.last_bot_response = text.strip()

def push_user(text):
    st.session_state.chat_history.append({"role":"user","content":text})
    chat_msg("user", text)
    st.session_state.last_user_query = text

# --- SIDEBAR ---
with st.sidebar:
    st.title("FINBOT")
    st.caption("Streamlit edition")
    nav = st.radio("Go to", ["💬 Chat","📊 Portfolio","📈 Stock Analysis","⚙️ Settings"], index=0)
    st.divider()
    st.selectbox("Financial Literacy Level", ["Beginner","Advanced"], key="literacy")
    st.text_input("Your name", key="user_name")
    st.checkbox("Enable SHAP (slower)", key="enable_shap")
    st.selectbox("Default Model", ["LSTM","CNN","Both"], key="model_choice")
    #st.text_input("NewsAPI key (optional)", key="api_key_news")
    st.divider()
    st.markdown("**Wallet**")
    st.metric("Balance", f"${st.session_state.user_wallet:.2f}")

# --- PAGES ---
if nav == "💬 Chat":
    st.header("Chat with Finbot")

    # Warm welcome on first load
    if len(st.session_state.chat_history) == 0:
        push_assistant("Hi, I'm Finbot. I'm here to help you with financial information and analysis.")
        push_assistant("Please enter **'bye'** to say goodbye.")
        push_assistant("If you need assistance, enter **'help'**.")
        push_assistant("After each response, you can rate if it was helpful!")

    # Show chat history
    for msg in st.session_state.chat_history:
        chat_msg(msg["role"], msg["content"])

    # Input
    user_input = st.chat_input("Type your message…")
    if user_input:
        # Save raw query for feedback linkage
        st.session_state.last_user_query = user_input

        # 1) spell-correct (word-by-word like your app)
        ui = ' '.join([correct(w) for w in user_input.split()])  # :contentReference[oaicite:5]{index=5}
        push_user(user_input)

        # 2) Name change / recognition (same logic as your app)
        if check_name_change(ui):  # :contentReference[oaicite:6]{index=6}
            new_name = name_change(ui)
            if new_name.strip():
                st.session_state.user_name = new_name
                push_assistant(f"- Finbot: Hi, {new_name}")
            else:
                push_assistant("- Finbot: I couldn't catch the new name—try 'change my name to Alex'.")
            st.stop()

        resp = name_response(ui, threshold=0.9)  # trigger memory line like your app
        if resp != 'NOT FOUND':
            push_assistant(f"- Finbot: You're {st.session_state.user_name}, I have a great memory ┑(￣u ￣)┍")
            st.stop()

        # 3) Exit / utilities
        if ui.lower().strip() == "bye":
            push_assistant("Bye!")
            st.stop()

        if ui.lower() in ['feedback stats', 'show feedback', 'rating stats']:
            push_assistant(feedback_stats())
            st.stop()

        # 4) Wallet actions
        if re.search(r'\b(check|wallet)\b.*\b(balance|money)\b', ui.lower()):
            push_assistant(f"- Finbot: Your current wallet balance is: ${st.session_state.user_wallet:.2f}")
            st.stop()

        if ui.lower().startswith("add ") and "wallet" in ui.lower():
            nums = re.findall(r"([\d.]+)", ui)
            if nums:
                push_assistant(add_to_wallet(float(nums[0])))
            else:
                push_assistant("- Finbot: Please provide a valid number.")
            st.stop()

        # 5) Buy/Sell/Price
        if ui.lower().startswith("buy stock"):
            parts = ui.split()
            if len(parts) >= 4:
                ticker = parts[2]
                nums = re.findall(r"([\d.]+)", ui)
                if nums:
                    push_assistant(buy_stock(ticker, float(nums[0])))
                else:
                    push_assistant("- Finbot: Provide purchase amount.")
            else:
                push_assistant("- Finbot: Try: 'Buy stock AAPL with 1000'")
            st.stop()

        if "current price" in ui.lower() or "stock price" in ui.lower():
            words = ui.upper().split()
            ticker = words[-1]
            price = get_current_stock_price(ticker)
            if isinstance(price,(int,float)):
                push_assistant(f"- Finbot: The current price for {ticker} is: ${price:.2f}")
            else:
                push_assistant("- Finbot: Unable to retrieve price. Check ticker.")
            st.stop()

        if "portfolio" in ui.lower():
            df = get_portfolio_table()
            if df.empty:
                push_assistant("- Finbot: Your stock portfolio is empty.")
            else:
                push_assistant("- Finbot: Here's your stock portfolio:")
                st.dataframe(df, use_container_width=True)
            st.stop()

        # 6) Time / Today (from small_talk)
        if " time" in ui.lower() or ui.lower().strip()=="today":
            buf = io.StringIO()
            # small_talk prints to stdout;
            import sys
            old = sys.stdout; sys.stdout = buf
            if " time" in ui.lower():
                time_response('time')
            else:
                time_response('today')
            sys.stdout = old
            push_assistant(buf.getvalue().replace("- Skynet:", "- Finbot:"))
            st.stop()

        # 7) Investment advice (prediction + sentiment)
        if 'invest in' in ui.lower() and 'should i' in ui.lower():
            try:
                words = ui.upper().split()
                ticker = words[words.index("IN")+1]
            except Exception:
                push_assistant("- Finbot: Please specify like 'Should I invest in AAPL?'")
                st.stop()

            with st.status(f"Analyzing {ticker}…", expanded=True) as status:
                st.write("Running prediction model…")
                models = ("CNN",) if st.session_state.model_choice=="CNN" else \
                         ("LSTM",) if st.session_state.model_choice=="LSTM" else ("LSTM","CNN")
                def prog(m): st.write(m)
                preds, logs, charts = predict_stock_price_with_xai(
                    ticker, models, enable_shap=st.session_state.enable_shap, progress_cb=prog
                )
                price_now = get_current_stock_price(ticker) or float("nan")
                best = next(iter(preds.values()), None)
                change_pct = ((best - price_now)/price_now*100) if (best and price_now>0) else float("nan")
                advice = "BUY" if change_pct>10 else "SELL" if change_pct<-10 else "HOLD"
                st.write(logs)
                for fig in charts:
                    st.pyplot(fig, clear_figure=True)

                status.update(label="Running news sentiment…")
                headlines = fetch_news(st.session_state.api_key_news, ticker) if st.session_state.api_key_news else []
                scores = analyze_sentiment(headlines)
                avg = overall_sentiment(scores)
                senti_advice = advice_from_sentiment(avg)

                summary = (f"Advice: {advice}\n"
                           f"Current Price: ${price_now:.2f}\n"
                           f"Predicted Price: ${best:.2f} (Change: {change_pct:.2f}%)\n"
                           f"News Sentiment: {senti_advice} | Avg score: {avg:.2f}")
                push_assistant(summary)
                status.update(label="Analysis complete.", state="complete")
            st.stop()

        # 8) Explicit predict command in chat
        if ui.lower().startswith("predict stock"):
            parts = ui.upper().split()
            ticker = parts[-1] if len(parts)>=3 else None
            if not ticker:
                push_assistant("- Finbot: Provide a ticker, e.g., 'Predict stock TSLA with CNN'")
                st.stop()
            models = ("CNN",) if st.session_state.model_choice=="CNN" else \
                     ("LSTM",) if st.session_state.model_choice=="LSTM" else ("LSTM","CNN")
            with st.status(f"Predicting {ticker}…", expanded=True) as status:
                def prog(m): st.write(m)
                preds, logs, charts = predict_stock_price_with_xai(
                    ticker, models, enable_shap=st.session_state.enable_shap, progress_cb=prog
                )
                st.write(logs)
                for fig in charts:
                    st.pyplot(fig, clear_figure=True)
                if preds:
                    lines = [f"{m}: ${v:.2f}" for m,v in preds.items()]
                    push_assistant("Prediction results for next day:\n" + "\n".join(lines))
                else:
                    push_assistant("- Finbot: Could not generate predictions.")
                status.update(label="Done.", state="complete")
            st.stop()

        # 9) Small talk first, then QA by literacy level (mirrors your main) :contentReference[oaicite:8]{index=8} :contentReference[oaicite:9]{index=9}
        resp = talk_response(ui, threshold=0.9)
        if resp != 'NOT FOUND':
            push_assistant(f"- Finbot: {resp}")
            st.stop()

        # QA answers vary by literacy (Beginner vs Advanced like your flow)
        literacy_for_QA = "Advanced" if st.session_state.literacy == "Advanced" else "Beginner"
        resp2 = answer_Q(ui, threshold=0.1, user_financial_literacy=literacy_for_QA)
        if resp2 != 'NOT FOUND' and not resp2.startswith("Error"):
            push_assistant(f"- Finbot: {resp2}")
        else:
            push_assistant("I'm sorry ˙◠˙ I don't quite understand. Try asking me to predict a stock price or inquire about market trends.")

    # Quick feedback buttons (writes JSON + poor_table.xlsx with last matched QA)
    cols = st.columns(3)
    with cols[0]:
        if st.button("👍 Helpful"):
            save_feedback(st.session_state.last_user_query, st.session_state.last_bot_response, "Good")
            st.toast("Thanks for the positive feedback!", icon="✅")
    with cols[1]:
        comments = st.text_input("Optional feedback note…", key="poor_comment")
        if st.button("👎 Not Helpful"):
            save_feedback(st.session_state.last_user_query, st.session_state.last_bot_response, "Poor", comments)
            save_poor_rating_to_excel(st.session_state.last_user_query, st.session_state.last_bot_response, comments)
            st.toast("Thanks — we recorded your feedback.", icon="📝")
    with cols[2]:
        if st.button("📊 Feedback Stats"):
            st.info(feedback_stats())

elif nav == "📊 Portfolio":
    st.header("Portfolio Management")

    with st.expander("Wallet", expanded=True):
        c1,c2,c3 = st.columns([1,1,2])
        with c1:
            st.metric("Current Balance", f"${st.session_state.user_wallet:.2f}")
        with c2:
            amt = st.number_input("Add funds", min_value=0.0, step=50.0, value=0.0, key="add_amt")
        with c3:
            if st.button("Add", type="primary"):
                msg = add_to_wallet(amt)
                st.success(msg)

    df = get_portfolio_table()
    st.subheader("Stock Holdings")
    if df.empty:
        st.info("Your stock portfolio is empty.")
    else:
        st.dataframe(df, use_container_width=True)

    st.divider()
    c_buy, c_sell = st.columns(2)

    with c_buy:
        st.subheader("Buy Stocks")
        bs_t = st.text_input("Ticker", key="buy_ticker")
        bs_a = st.number_input("Amount ($)", min_value=0.0, step=50.0, value=0.0, key="buy_amount")
        if st.button("Buy", type="primary"):
            st.write(buy_stock(bs_t, bs_a))

    with c_sell:
        st.subheader("Sell Stocks")
        choices = [""] + sorted(list(st.session_state.user_portfolio.keys()))
        ss_t = st.selectbox("Ticker", choices, index=0, key="sell_ticker")
        mode = st.radio("Sell mode", ["All","Shares","Amount ($)"], horizontal=True)
        ss_shares = ss_amount = None
        if mode == "Shares":
            ss_shares = st.number_input("Number of shares", min_value=0.0, step=0.1, value=0.0, key="sell_shares")
        elif mode == "Amount ($)":
            ss_amount = st.number_input("Dollar amount", min_value=0.0, step=50.0, value=0.0, key="sell_amount")
        if st.button("Sell", type="secondary"):
            st.write(sell_stock(ss_t, shares=ss_shares if mode=="Shares" else None,
                                amount=ss_amount if mode=="Amount ($)" else None))

elif nav == "📈 Stock Analysis":
    st.header("Stock Analysis")

    c0, c1, c2 = st.columns([2,1,1])
    with c0:
        symbol = st.text_input("Stock symbol (e.g., AAPL, TSLA)", key="analysis_symbol")
    with c1:
        if st.button("Get Price"):
            if symbol:
                price = get_current_stock_price(symbol)
                if isinstance(price,(int,float)):
                    st.success(f"Current price of {symbol.upper()}: ${price:.2f}")
                    try:
                        data = yf.download(symbol, period="1mo")
                        plt.style.use('ggplot')
                        fig, ax = plt.subplots(figsize=(10,4))
                        ax.plot(data['Close'], linewidth=2)
                        ax.set_title(f"{symbol.upper()} Price - Last Month")
                        ax.set_ylabel("Price ($)")
                        ax.grid(True, alpha=0.3)
                        fig.tight_layout()
                        st.pyplot(fig, clear_figure=True)
                    except Exception as e:
                        st.error(f"Chart error: {e}")
                else:
                    st.error("Price unavailable. Check ticker.")
            else:
                st.info("Enter a symbol first.")
    with c2:
        if st.button("Predict"):
            if not symbol:
                st.info("Enter a symbol first.")
            else:
                models = ("CNN",) if st.session_state.model_choice=="CNN" else \
                         ("LSTM",) if st.session_state.model_choice=="LSTM" else ("LSTM","CNN")
                with st.status(f"Predicting {symbol.upper()}…", expanded=True) as status:
                    def prog(msg): st.write(msg)
                    preds, logs, charts = predict_stock_price_with_xai(
                        symbol, models, enable_shap=st.session_state.enable_shap, progress_cb=prog
                    )
                    st.write(logs)
                    for fig in charts:
                        st.pyplot(fig, clear_figure=True)
                    if preds:
                        lines = [f"{m}: ${v:.2f}" for m,v in preds.items()]
                        st.success("Prediction results for next day:\n" + "\n".join(lines))
                    else:
                        st.error("Could not generate predictions.")
                    status.update(label="Done.", state="complete")

elif nav == "⚙️ Settings":
    st.header("Settings")
    st.write("Adjust your preferences in the sidebar.")
    st.write("• Name")
    st.write("• Literacy level")
    st.write("• Default model")
    st.write("• Toggle SHAP explanations (slower)")
    st.write("• Optional NewsAPI key for sentiment analysis")

    st.subheader("Feedback Tools")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Show Feedback Stats"):
            st.info(feedback_stats())
    with c2:
        path = poor_xlsx_path()
        if os.path.exists(path):
            with open(path, "rb") as f:
                st.download_button("Download poor_ratings.xlsx", data=f, file_name="poor_table.xlsx")
        else:
            st.caption("No poor ratings recorded yet.")
