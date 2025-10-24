from QA import answer_Q, last_matched_question, last_matched_answer
from name_management import name_change
from name_management import check_name_change
from name_management import name_response
from small_talk import talk_response
from spell_check import correct
from small_talk import time_response
from sklearn.metrics import mean_absolute_error, mean_squared_error
from textblob import TextBlob

import json
import os
import FreeSimpleGUI as sg
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import datetime as dt
import requests
import sys
import io
import re
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D, Flatten
import shap
import os

# Suppress TensorFlow logs
import tensorflow as tf

tf.get_logger().setLevel('ERROR')

# To keep a consistent color scheme
COLORS = {
    'primary': '#4CAF50',  # Green - main actions and highlights
    'secondary': '#2196F3',  # Blue - secondary actions
    'accent': '#FF9800',  # Orange - accent color for highlights
    'success': '#4CAF50',  # Green - success messages
    'warning': '#FFC107',  # Yellow - warning messages
    'error': '#F44336',  # Red - error messages and alerts
    'text_primary': 'black',  # Very dark gray - primary text color
    'text_secondary': '#757575',  # Medium gray - secondary text color
    'background_light': '#FAFAFA',  # Very light gray - main background
    'background_medium': '#F5F5F5',  # Light gray - section backgrounds
    'background_dark': '#E0E0E0',  # Dark gray - dividers and borders
    'card_background': '#FFFFFF',  # White - card backgrounds
    'colour_black': '#000000'




}

# Global Variables
user_wallet = 0.0
user_portfolio = {}
last_matched_question = ""
last_matched_answer = ""
last_matched_source = ""
global_predictions = {}
global_prediction_text = ""  # Store the formatted output text


# New function for chat message bubbles
def add_message_to_chat(window, sender, message, user_name="User"):
    """
    Adds a styled message bubble to the chat column with proper text wrapping.

    Args:
        window: The PySimpleGUI window
        sender: 'user' or 'bot'
        message: The message text
        user_name: The user's name (used for user messages)
    """
    # Get the current time for the timestamp
    current_time = datetime.now().strftime("%I:%M %p")

    # Create a unique key for this message
    msg_key = f'-MSG-{datetime.now().strftime("%H%M%S")}-'

    if sender == 'user':
        # Create user message style (right-aligned, blue bubble)
        # Set maximum width for text element and enable word wrapping
        message_element = sg.Text(message, background_color=COLORS['secondary'], text_color='white',
                                  pad=((10, 10), (10, 10)), font=("Helvetica", 10), key=msg_key,
                                  size=(40, None), auto_size_text=False)  # Set max width and disable auto-sizing

        message_column = sg.Column([[message_element]], background_color=COLORS['secondary'], pad=((0, 0), (5, 0)))
        timestamp_element = sg.Text(current_time, font=("Helvetica", 7), text_color=COLORS['text_secondary'],
                                    pad=((0, 5), (0, 10)))

        layout = [[sg.Push(), message_column], [sg.Push(), timestamp_element]]
    else:
        # Create bot message style (left-aligned, light gray bubble)
        # Remove the "- Finbot: " prefix if it exists
        if isinstance(message, str) and message.startswith("- Finbot: "):
            message = message[10:]
        elif isinstance(message, str) and message.startswith("- Finbot: "):
            message = message[12:]
        elif isinstance(message, str) and message.startswith("- Skynet: "):
            message = message[10:]

        # For long messages, use a Multiline element instead of Text
        if len(message) > 80:  # Use Multiline for longer messages
            message_element = sg.Multiline(message, background_color=COLORS['background_dark'],
                                           text_color=COLORS['text_primary'], border_width=0,
                                           size=(50, min(15, message.count('\n') + 3)),
                                           # Adjust height based on content
                                           disabled=True, no_scrollbar=False, key=msg_key,
                                           font=("Helvetica", 10), pad=((10, 10), (10, 10)))
        else:  # Use Text for shorter messages
            message_element = sg.Text(message, background_color=COLORS['background_dark'],
                                      text_color=COLORS['text_primary'],
                                      pad=((10, 10), (10, 10)), font=("Helvetica", 10), key=msg_key,
                                      size=(50, None), auto_size_text=False)  # Set max width and disable auto-sizing

        message_column = sg.Column([[message_element]], background_color=COLORS['background_dark'],
                                   pad=((0, 0), (5, 0)))
        timestamp_element = sg.Text(current_time, font=("Helvetica", 7), text_color=COLORS['text_secondary'],
                                    pad=((5, 0), (0, 10)))

        layout = [[message_column, sg.Push()], [timestamp_element]]

    # Add to chat column
    for row in layout:
        window.extend_layout(window['-CHAT_COLUMN-'], [row])

    # Force a window refresh after adding each row
    window.refresh()

    # Auto-scroll to bottom of chat - use a try/except for each method
    try:
        # First try this method
        window['-CHAT_COLUMN-'].contents_changed()
    except:
        pass

    try:
        # Then try this method
        window['-CHAT_COLUMN-'].Widget.canvas.yview_moveto(1.0)
    except:
        pass

    try:
        # Finally try this method
        window['-CHAT_COLUMN-'].set_vscroll_position(1.0)
    except:
        pass

    # One final refresh to ensure the message is visible
    window.refresh()


# Function to save feedback to a JSON file
def save_feedback(user_query, bot_response, rating, comments=None):
    feedback_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_query": user_query,
        "bot_response": bot_response,
        "rating": rating,
        "comments": comments,
        "financial_literacy_level": user_financial_literacy
    }

    # Create feedback directory if it doesn't exist
    if not os.path.exists("feedback"):
        os.makedirs("feedback")

    # Append to feedback file
    feedback_file = "feedback/finbot_feedback.json"

    # Read existing feedback if file exists
    all_feedback = []
    if os.path.exists(feedback_file):
        try:
            with open(feedback_file, 'r') as f:
                all_feedback = json.load(f)
        except json.JSONDecodeError:
            # If file is corrupted, start fresh
            all_feedback = []

    # Add new feedback
    all_feedback.append(feedback_data)

    # Write back to file
    with open(feedback_file, 'w') as f:
        json.dump(all_feedback, f, indent=4)

    return True


# Function to save poor ratings to Excel file
def view_poor_ratings():
    """
    Opens a new window to display poor ratings from the Excel file.
    """
    poor_ratings_file = "feedback/poor_table.xlsx"
    if not os.path.exists(poor_ratings_file):
        sg.popup("No poor ratings have been recorded yet.", title="Poor Ratings")
        return

    try:
        # Read the poor ratings data
        df = pd.read_excel(poor_ratings_file)

        # If empty
        if df.empty:
            sg.popup("No poor ratings have been recorded yet.", title="Poor Ratings")
            return

        # Create a layout to display the data
        layout = [
            [sg.Text("Poor Ratings Summary", font=("Helvetica", 16, "bold"), pad=((0, 0), (10, 20)),
                     text_color=COLORS['text_primary'])],
            [sg.Table(
                values=df.values.tolist(),
                headings=df.columns.tolist(),
                display_row_numbers=True,
                auto_size_columns=False,
                num_rows=min(25, len(df)),
                col_widths=[15, 40, 40, 40, 40, 20, 10],
                justification='left',
                key='-TABLE-',
                tooltip='Poor Ratings',
                background_color=COLORS['card_background'],
                text_color=COLORS['text_primary']
            )],
            [sg.Button("Export CSV", key='-EXPORT-', size=(12, 1), pad=((5, 10), (20, 5)),
                       button_color=(COLORS['card_background'], COLORS['secondary'])),
             sg.Button("Close", size=(10, 1), pad=((5, 10), (20, 5)),
                       button_color=(COLORS['card_background'], COLORS['primary']))]
        ]

        window = sg.Window("Poor Ratings Data", layout, resizable=True, finalize=True,
                           background_color=COLORS['background_light'])

        while True:
            event, values = window.read()
            if event == sg.WIN_CLOSED or event == "Close":
                break

            if event == '-EXPORT-':
                export_file = sg.popup_get_file('Save CSV As', save_as=True, file_types=(("CSV Files", "*.csv"),))
                if export_file:
                    if not export_file.endswith('.csv'):
                        export_file += '.csv'
                    df.to_csv(export_file, index=False)
                    sg.popup(f"Data exported to {export_file}")

        window.close()

    except Exception as e:
        sg.popup(f"Error viewing poor ratings: {str(e)}", title="Error")


# Function to view poor ratings
def get_feedback_stats():
    feedback_file = "feedback/finbot_feedback.json"
    if not os.path.exists(feedback_file):
        return "No feedback data available yet."

    try:
        with open(feedback_file, 'r') as f:
            all_feedback = json.load(f)

        if not all_feedback:
            return "No feedback data available yet."

        total_ratings = len(all_feedback)
        positive_ratings = sum(1 for item in all_feedback if item["rating"] in ["Good", "Excellent"])
        neutral_ratings = sum(1 for item in all_feedback if item["rating"] == "Neutral")
        negative_ratings = sum(1 for item in all_feedback if item["rating"] in ["Poor", "Very Poor"])

        avg_rating = positive_ratings / total_ratings * 100

        stats = f"Feedback Statistics:\n"
        stats += f"Total responses rated: {total_ratings}\n"
        stats += f"Positive ratings: {positive_ratings} ({positive_ratings / total_ratings * 100:.1f}%)\n"
        stats += f"Neutral ratings: {neutral_ratings} ({neutral_ratings / total_ratings * 100:.1f}%)\n"
        stats += f"Negative ratings: {negative_ratings} ({negative_ratings / total_ratings * 100:.1f}%)\n"
        stats += f"Overall satisfaction: {avg_rating:.1f}%"

        return stats

    except Exception as e:
        return f"Error retrieving feedback statistics: {str(e)}"


def save_poor_rating_to_excel(user_query, bot_response, comments=None):
    # Create feedback directory if it doesn't exist
    if not os.path.exists("feedback"):
        os.makedirs("feedback")

    poor_ratings_file = "feedback/poor_table.xlsx"

    # Create a dataframe with the current poor rating
    current_rating = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_question": user_query,
        "bot_response": bot_response,
        "file_question": last_matched_question if last_matched_question else "No matched question",
        "file_answer": last_matched_answer if last_matched_answer else "No matched answer",
        "source": last_matched_source if last_matched_source else "Unknown",
        "explanation": comments if comments else "",
        "financial_literacy_level": user_financial_literacy
    }

    # Load existing poor ratings if file exists
    if os.path.exists(poor_ratings_file):
        try:
            existing_df = pd.read_excel(poor_ratings_file)
            updated_df = pd.concat([existing_df, pd.DataFrame([current_rating])], ignore_index=True)
        except Exception as e:
            print(f"Error reading existing poor ratings file: {str(e)}")
            updated_df = pd.DataFrame([current_rating])
    else:
        updated_df = pd.DataFrame([current_rating])

    # Save to Excel
    try:
        updated_df.to_excel(poor_ratings_file, index=False)
        print(f"- Finbot: Feedback saved to {poor_ratings_file}")
    except Exception as e:
        print(f"- Finbot: Error saving feedback to Excel: {str(e)}")


# Function to save portfolio data to a JSON file
def save_portfolio_data():
    global user_wallet, user_portfolio

    # Create a data structure to save
    portfolio_data = {
        "wallet_balance": user_wallet,
        "stocks": user_portfolio
    }

    # Create portfolio directory if it doesn't exist
    if not os.path.exists("portfolio"):
        os.makedirs("portfolio")

    # Save to a JSON file
    with open("portfolio/user_portfolio.json", 'w') as f:
        json.dump(portfolio_data, f, indent=4)

    print(f"- Finbot: Portfolio data saved successfully.")
    return True


def load_portfolio_data():
    global user_wallet, user_portfolio

    portfolio_file = "portfolio/user_portfolio.json"

    # Check if the file exists
    if os.path.exists(portfolio_file):
        try:
            with open(portfolio_file, 'r') as f:
                data = json.load(f)

            # Update global variables
            user_wallet = data.get("wallet_balance", 0.0)
            user_portfolio = data.get("stocks", {})

            # Handle legacy portfolio data format if needed
            for ticker, value in user_portfolio.items():
                if not isinstance(value, dict):
                    # Convert old format (just shares) to new format (shares and avg_price)
                    current_price = get_current_stock_price(ticker)
                    if isinstance(current_price, (int, float)):
                        user_portfolio[ticker] = {
                            'shares': float(value),
                            'avg_price': current_price
                        }
                    else:
                        # If can't get current price, use a placeholder
                        user_portfolio[ticker] = {
                            'shares': float(value),
                            'avg_price': 0.0
                        }

            print(f"- Finbot: Portfolio data loaded successfully.")
            return True
        except Exception as e:
            print(f"- Finbot: Error loading portfolio data: {str(e)}")
            return False
    else:
        print(f"- Finbot: No saved portfolio data found.")
        return False


# Function to Add Money to Wallet
def add_to_wallet(amount):
    global user_wallet
    try:
        amount = float(amount)
        if amount > 0:
            user_wallet += amount
            print(f"- Finbot: ${amount:.2f} has been added to your wallet.")

            # Save portfolio data after adding money
            save_portfolio_data()
        else:
            print("- Finbot: Please enter a positive amount to add to your wallet.")
    except ValueError:
        print("- Finbot: That doesn't seem to be a valid amount.")


# Function to Fetch Current Stock Price
def get_current_stock_price(ticker):
    stock = yf.Ticker(ticker)
    todays_data = stock.history(period='1d')
    try:
        price = todays_data['Close'].iloc[-1]
        return price
    except IndexError:
        return "Unable to retrieve the current price. The ticker symbol may be incorrect or data unavailable."


# Modified buy_stock function with persistence
def buy_stock(ticker, amount):
    global user_wallet, user_portfolio
    current_price = get_current_stock_price(ticker)
    if current_price is None:
        print("- Finbot: Sorry, I couldn't retrieve the current price for that symbol.")
        return
    if isinstance(current_price, str):
        print(current_price)  # Error message from the get_current_stock_price function
        return

    # Calculate how many shares the user can buy with the specified amount
    num_shares = float(amount) / float(current_price)

    # Debug print to check calculations
    print(
        f"- Finbot: Calculation details: Amount=${float(amount):.2f}, Price=${float(current_price):.2f}, Shares={num_shares:.4f}")

    if num_shares < 0.01:  # Changed to 0.01 for a more reasonable minimum
        print(
            f"- Finbot: You do not have enough money to buy a meaningful amount of {ticker}. Current price is: ${current_price:.2f}")
        return

    if user_wallet >= float(amount):
        user_wallet -= float(amount)

        # Add the shares to the portfolio with purchase price tracking
        if ticker in user_portfolio:
            # Calculate weighted average price for existing position
            current_shares = user_portfolio[ticker]['shares']
            current_avg_price = user_portfolio[ticker]['avg_price']
            total_current_value = current_shares * current_avg_price
            new_total_value = total_current_value + float(amount)
            new_total_shares = current_shares + num_shares
            new_avg_price = new_total_value / new_total_shares

            user_portfolio[ticker] = {
                'shares': new_total_shares,
                'avg_price': new_avg_price
            }
        else:
            user_portfolio[ticker] = {
                'shares': num_shares,
                'avg_price': current_price
            }

        print(
            f"- Finbot: You have purchased {num_shares:.4f} shares of {ticker} at ${current_price:.2f} per share. Remaining wallet balance is ${user_wallet:.2f}")


        # Save portfolio data after purchase
        save_portfolio_data()
    else:
        print(f"- Finbot: You do not have enough money in your wallet. Your current balance is ${user_wallet:.2f}")


def sell_stock(ticker, shares=None, amount=None):
    """
    Sell stock from the user's portfolio.

    Args:
        ticker (str): The stock ticker symbol to sell
        shares (float, optional): Number of shares to sell. If None, amount parameter is used.
        amount (float, optional): Dollar amount of shares to sell. If None, shares parameter is used.
                                 If both are None, all shares are sold.

    Returns:
        bool: True if the sale was successful, False otherwise
    """
    global user_wallet, user_portfolio

    # Convert ticker to uppercase for consistency
    ticker = ticker.upper()

    # Check if the user owns the stock
    if ticker not in user_portfolio:
        print(f"- Finbot: You don't own any shares of {ticker} in your portfolio.")
        return False

    # Get current stock price
    current_price = get_current_stock_price(ticker)
    if isinstance(current_price, str):
        print(f"- Finbot: {current_price}")
        return False

    # Get total shares owned and average purchase price
    owned_shares = user_portfolio[ticker]['shares']
    avg_purchase_price = user_portfolio[ticker]['avg_price']

    # Determine how many shares to sell
    shares_to_sell = 0

    if shares is not None:
        # Sell specific number of shares
        shares_to_sell = float(shares)
        if shares_to_sell <= 0:
            print("- Finbot: Please enter a positive number of shares to sell.")
            return False
        elif shares_to_sell > owned_shares:
            print(
                f"- Finbot: You only own {owned_shares:.4f} shares of {ticker}, but you're trying to sell {shares_to_sell} shares.")
            return False
    elif amount is not None:
        # Sell specific dollar amount of shares
        amount = float(amount)
        if amount <= 0:
            print("- Finbot: Please enter a positive dollar amount to sell.")
            return False

        shares_to_sell = min(amount / current_price, owned_shares)
        if shares_to_sell < 0.0001:  # Very small number of shares
            print(
                f"- Finbot: The amount ${amount:.2f} is too small to sell any shares of {ticker} at the current price.")
            return False
    else:
        # Sell all shares
        shares_to_sell = owned_shares

    # Calculate the sale value
    sale_value = shares_to_sell * current_price

    # Calculate realized gain/loss
    purchase_value = shares_to_sell * avg_purchase_price
    realized_gain_loss = sale_value - purchase_value

    # Update portfolio
    if shares_to_sell == owned_shares:
        # Remove stock from portfolio if selling all shares
        del user_portfolio[ticker]
    else:
        # Update shares if selling partial position (avg price stays the same)
        user_portfolio[ticker]['shares'] -= shares_to_sell

    # Add proceeds to wallet
    user_wallet += sale_value

    # Save portfolio data
    save_portfolio_data()

    # Print confirmation message
    print(f"- Finbot: Successfully sold {shares_to_sell:.4f} shares of {ticker} at ${current_price:.2f} per share.")
    print(f"- Finbot: ${sale_value:.2f} has been added to your wallet. Your balance is now ${user_wallet:.2f}.")

    # Print gain/loss information
    if realized_gain_loss > 0:
        print(f"- Finbot: You realized a profit of ${realized_gain_loss:.2f} on this trade.")
    elif realized_gain_loss < 0:
        print(f"- Finbot: You realized a loss of ${-realized_gain_loss:.2f} on this trade.")
    else:
        print(f"- Finbot: You broke even on this trade.")

    return True


# Main Prediction Function with Explainable AI (SHAP)
def predict_stock_price_with_xai(company, model_types, window=None, results_key=None):
    # Initialise these variables first to avoid the "might be referenced before assignment" error
    global global_predictions, global_prediction_text
    global_predictions = {}  # Initialise or reset the global predictions
    global_prediction_text = ""
    prediction_window = None
    original_stdout = sys.stdout
    predictions = {}

    try:
        # Create a new window for prediction visualization with improved styling
        prediction_layout = [
            [sg.Text(f"Stock Price Prediction for {company}", font=("Helvetica", 18, "bold"),
                     pad=((0, 0), (10, 20)), text_color=COLORS['text_primary'])],
            [sg.Output(size=(80, 20), font=("Courier New", 10),
                       background_color=COLORS['card_background'], text_color=COLORS['text_primary'])],
            [sg.Button("Close", size=(10, 1), pad=((0, 0), (20, 10)),
                       button_color=(COLORS['card_background'], COLORS['primary']), font=("Helvetica", 10))]
        ]
        prediction_window = sg.Window(f"Stock Prediction - {company}",
                                      prediction_layout,
                                      finalize=True,
                                      background_color=COLORS['background_light'],
                                      element_padding=(10, 5),
                                      margins=(20, 20))

        # Redirect stdout to the new window
        sys.stdout = prediction_window['-LISTBOX-'] if '-LISTBOX-' in prediction_window.AllKeysDict else sys.stdout

        # Fetch historical data
        start = dt.datetime(2012, 1, 1)
        end = dt.datetime.now()
        data = yf.download(company, start=start, end=end)

        # Prepare Data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
        prediction_days = 60
        x_train, y_train = [], []

        for x in range(prediction_days, len(scaled_data)):
            x_train.append(scaled_data[x - prediction_days:x, 0])
            y_train.append(scaled_data[x, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        # Train Models
        model_outputs = {}

        print(f"Training models for {company}...")

        if 'LSTM' in model_types:
            print("Training LSTM model...")
            lstm_model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
                Dropout(0.2),
                LSTM(50, return_sequences=True),
                Dropout(0.2),
                LSTM(50),
                Dropout(0.2),
                Dense(1)
            ])
            lstm_model.compile(optimizer='adam', loss='mean_squared_error')
            lstm_model.fit(x_train, y_train, epochs=25, batch_size=32, verbose=0)
            model_outputs['LSTM'] = lstm_model
            print("LSTM model training complete.")

        if 'CNN' in model_types:
            print("Training CNN model...")
            cnn_model = Sequential([
                Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(x_train.shape[1], 1)),
                MaxPooling1D(pool_size=2),
                Conv1D(filters=128, kernel_size=2, activation='relu'),
                MaxPooling1D(pool_size=2),
                Flatten(),
                Dense(50, activation='relu'),
                Dense(1)
            ])
            cnn_model.compile(optimizer='adam', loss='mean_squared_error')
            cnn_model.fit(x_train, y_train, epochs=25, batch_size=32, verbose=0)
            model_outputs['CNN'] = cnn_model
            print("CNN model training complete.")

        # Test Data Preparation
        test_start = dt.datetime(2022, 1, 1)
        test_end = dt.datetime.now()
        test_data = yf.download(company, start=test_start, end=test_end)
        actual_prices = test_data['Close'].values
        total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

        model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
        model_inputs = model_inputs.reshape(-1, 1)
        model_inputs = scaler.transform(model_inputs)

        x_test = []
        for x in range(prediction_days, len(model_inputs)):
            x_test.append(model_inputs[x - prediction_days:x, 0])

        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        # Improved chart styling
        plt.style.use('ggplot')

        for model_type, model in model_outputs.items():
            # Predict Prices
            print(f"Generating predictions for {model_type}...")
            predicted_prices = model.predict(x_test)
            predicted_prices = scaler.inverse_transform(predicted_prices)

            # Metrics
            mae = mean_absolute_error(actual_prices, predicted_prices.flatten())
            mse = mean_squared_error(actual_prices, predicted_prices.flatten())
            rmse = np.sqrt(mse)
            print(f"Accuracy metrics for {model_type}: MAE=${mae:.2f}, MSE=${mse:.2f}, RMSE=${rmse:.2f}")

            # Store all predicted prices for reference
            predictions[f"{model_type}_historical"] = predicted_prices.flatten()

            # Visualization: Actual vs Predicted Prices - improved styling
            plt.figure(figsize=(10, 5))
            plt.plot(actual_prices, color='#212121', linewidth=2, label='Actual Price')
            plt.plot(predicted_prices, color=COLORS['primary'], linewidth=2, label='Predicted Price')
            plt.title(f'{company} Stock Price Prediction ({model_type})', fontsize=14, fontweight='bold')
            plt.xlabel('Time', fontsize=12)
            plt.ylabel('Price ($)', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=10)
            plt.tight_layout()
            plt.show(block=False)  # Show plot without blocking execution

            # SHAP Integration
            print(f"Generating SHAP values for {model_type}...")

            def model_predict(data):
                data = data.reshape(data.shape[0], data.shape[1], 1)
                return model.predict(data).flatten()

            x_test_flattened = x_test.reshape(x_test.shape[0], x_test.shape[1])  # Flatten for SHAP
            background = x_test_flattened[:100]  # Background for SHAP
            explainer = shap.KernelExplainer(model_predict, background)
            shap_values = explainer.shap_values(x_test_flattened[:10])  # Explain first 10 samples

            # SHAP Summary Plot - improved styling
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, x_test_flattened[:10],
                              feature_names=[f"Day-{i}" for i in range(1, prediction_days + 1)])
            plt.title(f"SHAP Summary Plot ({model_type})", fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show(block=False)  # Show plot without blocking execution

        prediction_output_buffer = io.StringIO()
        temp_stdout = sys.stdout
        sys.stdout = prediction_output_buffer

        # Next-Day Price Prediction
        real_data = [model_inputs[len(model_inputs) - prediction_days:len(model_inputs), 0]]
        real_data = np.array(real_data)
        real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

        prediction_results = []
        for model_type, model in model_outputs.items():
            prediction = model.predict(real_data)
            prediction = scaler.inverse_transform(prediction).flatten()[0]  # Convert to scalar
            predictions[model_type] = prediction
            global_predictions[model_type] = prediction  # Also store in global variable

            # Store the formatted prediction string
            prediction_str = f"{model_type}: ${prediction:.2f}"
            prediction_results.append(prediction_str)

            print("\nNext-Day Price Predictions:")
            print(prediction_str)

        # Join all prediction results with newlines
        global_prediction_text = "\n".join(prediction_results)

        # Capture the output and restore stdout
        sys.stdout = temp_stdout
        global_prediction_output = prediction_output_buffer.getvalue()

        print(global_prediction_output)

        # Event loop for the prediction window
        while True:
            event, values = prediction_window.read()
            if event == sg.WIN_CLOSED or event == "Close":
                break

        # Reset stdout and close the window
        sys.stdout = original_stdout
        if prediction_window:
            prediction_window.close()

        # Update the main UI if window parameter was provided
        if window and results_key:
            # Build a complete results string
            result_text = f"Prediction results for next day:\n{global_prediction_text}\n\n"
            for model_type, pred in predictions.items():
                if model_type not in ['LSTM_historical', 'CNN_historical']:
                    if isinstance(pred, (int, float)):
                        result_text += f"{model_type} model predicts next day's price: ${pred:.2f}\n"
                    elif isinstance(pred, np.ndarray) and len(pred) > 0:
                        result_text += f"{model_type} model predicts next day's price: ${pred[0]:.2f}\n"
            window[results_key].update(result_text)

        return predictions

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        # Ensure stdout is restored in case of error
        sys.stdout = original_stdout
        # Close the window in case of error
        if prediction_window:
            prediction_window.close()

        # Update the main UI with error message if window parameter was provided
        if window and results_key:
            window[results_key].update(f"Error during prediction: {str(e)}")

        return predictions  # Return whatever predictions we have, even if empty

def get_investment_advice(ticker):
    # Initialise variables to avoid reference before assignment errors
    advice_window = None
    original_stdout = sys.stdout

    try:
        # Create a new window for investment advice with improved styling
        advice_layout = [
            [sg.Text(f"Investment Advice for {ticker}", font=("Helvetica", 18, "bold"),
                     pad=((0, 0), (10, 20)), text_color=COLORS['text_primary'])],
            [sg.Output(size=(80, 20), font=("Courier New", 10), key='-OUTPUT-',
                       background_color=COLORS['card_background'], text_color=COLORS['text_primary'])],
            [sg.Button("Close", size=(10, 1), pad=((0, 0), (20, 10)),
                       button_color=(COLORS['card_background'], COLORS['primary']), font=("Helvetica", 10))]
        ]
        advice_window = sg.Window(f"Investment Advice - {ticker}",
                                  advice_layout,
                                  finalize=True,
                                  background_color=COLORS['background_light'],
                                  element_padding=(10, 5),
                                  margins=(20, 20))

        # Redirect stdout to the new window's output element
        original_stdout = sys.stdout
        sys.stdout = advice_window['-OUTPUT-'] if '-OUTPUT-' in advice_window.AllKeysDict else sys.stdout

        print(f"Analyzing {ticker}...")

        # Fetch historical data for the stock
        data = yf.download(ticker, start="2022-01-01", end=dt.datetime.now().strftime('%Y-%m-%d'))
        if data.empty:
            print("No data available for this ticker.")
            return "No data available for this ticker."

        current_price = float(data['Close'].iloc[-1])  # Ensure current_price is a float, not numpy array

        # Calculate moving averages and standard deviation
        short_window = float(data['Close'].rolling(window=20).mean().iloc[-1])
        long_window = float(data['Close'].rolling(window=50).mean().iloc[-1])
        std_dev = float(data['Close'].rolling(window=20).std().iloc[-1])

        print(f"Current Price: ${current_price:.2f}")
        print(f"20-day Moving Average: ${short_window:.2f}")
        print(f"50-day Moving Average: ${long_window:.2f}")
        print(f"20-day Standard Deviation: ${std_dev:.2f}")

        print("\nRunning prediction models...")

        # Using existing predict_stock_price function which expects a company name
        predictions = predict_stock_price_with_xai(ticker,
                                                   ['CNN'])  # This must return a dictionary with arrays/lists

        if 'CNN' not in predictions or not predictions['CNN']:
            print("Prediction currently unavailable for this stock.")
            return "Prediction currently unavailable for this stock."

        predicted_future_price = float(predictions['CNN'])  # Ensure predicted price is a float

        # Calculate the percentage change
        price_change_percent = ((predicted_future_price - current_price) / current_price) * 100

        # Generate advice based on the price change
        advice = "BUY" if price_change_percent > 10 else "SELL" if price_change_percent < -10 else "HOLD"

        print("\n" + "=" * 40)
        print("         INVESTMENT ADVICE")
        print("=" * 40)
        print(f"Advice: {advice}")
        print(f"Current Price: ${current_price:.2f}")
        print(f"Predicted Price: ${predicted_future_price:.2f} (Change: {price_change_percent:.2f}%)")
        print("=" * 40)

        # Technical indicators evaluation
        print("\n" + "=" * 40)
        print("         TECHNICAL INDICATORS")
        print("=" * 40)
        if current_price > short_window:
            print("Price is above 20-day moving average: Bullish signal")
        else:
            print("Price is below 20-day moving average: Bearish signal")

        if current_price > long_window:
            print("Price is above 50-day moving average: Bullish signal")
        else:
            print("Price is below 50-day moving average: Bearish signal")

        if short_window > long_window:
            print("20-day MA is above 50-day MA: Bullish trend")
        else:
            print("20-day MA is below 50-day MA: Bearish trend")
        print("=" * 40)

        # Create a summary for return value
        advice_message = f"Advice: {advice}\n"
        advice_message += f"Current Price: ${current_price:.2f}\n"
        advice_message += f"Predicted Price: ${predicted_future_price:.2f} (Change: {price_change_percent:.2f}%)\n"
        advice_message += f"20-day Moving Average: ${short_window:.2f}\n"
        advice_message += f"50-day Moving Average: ${long_window:.2f}\n"
        advice_message += f"20-day Standard Deviation: ${std_dev:.2f}"

        # Improved chart styling
        plt.style.use('ggplot')

        # Display historical price chart with improved styling
        plt.figure(figsize=(10, 6))
        plt.plot(data['Close'], color=COLORS['primary'], linewidth=2, label='Close Price')
        plt.plot(data['Close'].rolling(window=20).mean(), color=COLORS['secondary'], linewidth=2,
                 linestyle='--',
                 label='20-Day MA')
        plt.plot(data['Close'].rolling(window=50).mean(), color=COLORS['accent'], linewidth=2, linestyle='-.',
                 label='50-Day MA')
        plt.title(f'{ticker} Stock Price with Moving Averages', fontsize=14, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show(block=False)

        # Event loop for the advice window
        while True:
            event, values = advice_window.read()
            if event == sg.WIN_CLOSED or event == "Close":
                break

        # Reset stdout and close the window
        sys.stdout = original_stdout
        if advice_window:
            advice_window.close()

        return advice_message

    except Exception as e:
        error_msg = f"An error occurred: {str(e)}"
        print(error_msg)
        # Ensure stdout is restored in case of error
        sys.stdout = original_stdout
        # Close the window in case of error
        if advice_window:
            advice_window.close()
        return error_msg

# Sentiment analysis functions
def fetch_news(api_key, ticker_symbol):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    query = f"{ticker_symbol} AND (stock OR market OR finance)"
    url = 'https://newsapi.org/v2/everything'

    params = {
        'q': query,
        'from': start_date_str,
        'to': end_date_str,
        'sortBy': 'relevance',
        'apiKey': api_key
    }

    response = requests.get(url, params=params)
    articles = response.json().get('articles', [])
    headlines = [{'headline': article['title'], 'date': article['publishedAt'][:10]} for article in articles]
    return headlines

def analyze_sentiment(headlines):
    sentiment_scores = []
    for headline in headlines:
        analysis = TextBlob(headline['headline'])
        sentiment_scores.append(
            {'date': headline['date'], 'headline': headline['headline'],
             'sentiment': analysis.sentiment.polarity})
    return sentiment_scores

def calculate_overall_sentiment(sentiment_scores):
    if sentiment_scores:
        total_sentiment = sum([score['sentiment'] for score in sentiment_scores])
        average_sentiment = total_sentiment / len(sentiment_scores)
        return average_sentiment
    else:
        return 0

def get_investment_advice_based_on_sentiment(average_sentiment):
    if average_sentiment > 0.1:
        return "BUY - Positive sentiment detected."
    elif average_sentiment < -0.1:
        return "SELL - Negative sentiment detected."
    else:
        return "HOLD - Neutral sentiment."

def check_wallet_balance():
    global user_wallet
    print(f"- Finbot: Your current wallet balance is: ${user_wallet:.2f}")

def list_portfolio():
    global user_portfolio
    if not user_portfolio:
        print("- Finbot: Your stock portfolio is empty.")
    else:
        print("- Finbot: Here's your stock portfolio:")
        for ticker, shares in user_portfolio.items():
            print(f"  {ticker}: {shares:.4f} shares")

#  Custom Output Filter Class
class FilteredStdout(io.StringIO):
    def write(self, txt):
        #  Filter out lines containing "Epoch", "loss:", "accuracy:", progress bars
        if not re.search(r"Epoch|loss:|accuracy:|\[=+>", txt):
            super().write(txt)  # Keep only chatbot-related responses

# Global variable to store user's financial literacy level
user_financial_literacy = "Beginner"

has_switched_level = False  # Flag to track if level has been printed

def track_matched_qa(user_query, bot_response, source):
    global last_matched_question, last_matched_answer, last_matched_source

    last_matched_source = source

    if source == "small_talk.csv":
        try:
            df = pd.read_csv('./dataset/small_talk.csv')
            # Find the closest match to the bot_response in Answer column
            for idx, row in df.iterrows():
                if row['Answer'].strip() == bot_response.strip():
                    last_matched_question = row['Question']
                    last_matched_answer = row['Answer']
                    return
        except Exception as e:
            print(f"Error tracking response from small_talk: {str(e)}")

    elif source == "question_answering.xlsx":
        try:
            # If this is from QA.py, the last_matched_question and last_matched_answer
            # should already be set by answer_Q function
            # We just need to make sure they're not empty
            if not last_matched_question and not last_matched_answer:
                # Try to find the match in the Excel file
                df = pd.read_excel('./dataset/question_answering.xlsx')
                for idx, row in df.iterrows():
                    # Check all literacy level columns
                    for col in ['Beginner', 'Intermediate', 'Advanced']:
                        if col in df.columns and str(row[col]).strip() == bot_response.strip():
                            last_matched_question = row['Question']
                            last_matched_answer = bot_response
                            return
        except Exception as e:
            print(f"Error tracking response from QA: {str(e)}")

    # If we get here, we couldn't find an exact match, so use what we know
    if not last_matched_question:
        last_matched_question = user_query
    if not last_matched_answer:
        last_matched_answer = bot_response

# Function to get portfolio data for display in the table
def get_portfolio_data():
    global user_portfolio
    data = []
    for ticker, details in user_portfolio.items():
        shares = details['shares']
        avg_price = details['avg_price']
        current_price = get_current_stock_price(ticker)

        if isinstance(current_price, (int, float)):
            current_value = shares * current_price
            purchase_value = shares * avg_price
            gain_loss = current_value - purchase_value
            percent_change = (gain_loss / purchase_value) * 100 if purchase_value > 0 else 0

            gain_loss_str = f"${gain_loss:.2f} ({percent_change:.1f}%)"
            gain_loss_color = 'green' if gain_loss > 0 else 'red' if gain_loss < 0 else 'black'

            data.append([
                ticker,
                f"{shares:.2f}",
                f"${avg_price:.2f}",
                f"${current_price:.2f}",
                f"${current_value:.2f}",
                gain_loss_str
            ])
    return data

# New GUI implementation that replaces the original chatbot_window
def create_finbot_app():
    global user_wallet, user_portfolio, user_financial_literacy, has_switched_level
    global last_matched_question, last_matched_answer, last_matched_source

    # Load saved portfolio data
    load_portfolio_data()

    # Set the theme to match our colour scheme
    sg.theme_background_color(COLORS['background_light'])
    sg.theme_text_element_background_color(COLORS['background_light'])
    sg.theme_element_background_color(COLORS['card_background'])
    sg.theme_text_color(COLORS['text_primary'])
    sg.theme_input_background_color(COLORS['card_background'])
    sg.theme_input_text_color(COLORS['text_primary'])

    # Initialise user_name variable
    user_name = "(User)"

    # Last message tracking for feedback
    last_user_query = ""
    last_bot_response = ""

    # Define the layout for the Chat section with improved styling
    chat_layout = [
        [sg.Text("Chat with Finbot", font=("Helvetica", 18, "bold"), pad=((0, 0), (0, 20)),
                 text_color=COLORS['text_primary'])],

        # Column for message bubbles
        [sg.Column([], key='-CHAT_COLUMN-', scrollable=True, vertical_scroll_only=True,
                   size=(500, 400), background_color=COLORS['background_light'])],

        [sg.InputText(key='-USER_INPUT-', size=(60, 1), do_not_clear=False,
                      font=("Helvetica", 10), pad=((0, 10), (10, 15)),
                      background_color=COLORS['card_background'], text_color=COLORS['text_primary']),
         sg.Button("Send", button_color=(COLORS['colour_black'], COLORS['primary']),
                   size=(10, 1), font=("Helvetica", 10), bind_return_key=True)],

        [sg.Text("Rate last response:", pad=((0, 10), (20, 10)), font=("Helvetica", 10))],
        [sg.Button("👍 Helpful", key='-RATE_GOOD-', size=(15, 1), pad=((0, 10), (0, 0)),
                   button_color=(COLORS['colour_black'], COLORS['success']), font=("Helvetica", 10)),
         sg.Button("👎 Not Helpful", key='-RATE_POOR-', size=(15, 1), pad=((0, 10), (0, 0)),
                   button_color=(COLORS['colour_black'], COLORS['error']), font=("Helvetica", 10))]
         #sg.Button("📊 View Feedback Stats", key='-FEEDBACK_STATS-', size=(20, 1), pad=((0, 10), (0, 0)),
                   #button_color=(COLORS['colour_black'], COLORS['secondary']), font=("Helvetica", 10)),
         #sg.Button("View Poor Ratings", key='-VIEW_POOR_RATINGS-', size=(20, 1), pad=((0, 10), (0, 0)),
                   #button_color=(COLORS['colour_black'], COLORS['accent']), font=("Helvetica", 10))]
    ]

    # layout for the Portfolio section

    # layout for the Stock Analysis section
    portfolio_layout = [
        [sg.Text("Portfolio Management", font=("Helvetica", 18, "bold"), pad=((0, 0), (0, 20)),
                 text_color=COLORS['text_primary'])],
        [sg.Frame("Wallet", [
            [sg.Text("Current Balance:", size=(15, 1), font=("Helvetica", 10), pad=((5, 10), (10, 10))),
             sg.Text(f"${user_wallet:.2f}", size=(15, 1), key='-WALLET_BALANCE-',
                     font=("Helvetica", 10, "bold"),
                     text_color=COLORS['success'], pad=((5, 10), (10, 10)))],
            [sg.Button("Add Funds", key='-ADD_FUNDS-', size=(10, 1), font=("Helvetica", 10),
                       button_color=(COLORS['colour_black'], COLORS['primary']), pad=((5, 10), (5, 10))),
             sg.InputText(size=(15, 1), key='-FUND_AMOUNT-', font=("Helvetica", 10), pad=((5, 10), (5, 10))),
             sg.Button("Update", key='-UPDATE_FUNDS-', size=(10, 1), font=("Helvetica", 10),
                       button_color=(COLORS['colour_black'], COLORS['secondary']), pad=((5, 10), (5, 10)))]
        ], pad=((0, 0), (0, 20)), relief=sg.RELIEF_SOLID, title_color='#000000', border_width=1)],

        [sg.Frame("Stock Holdings", [
            [sg.Table(values=get_portfolio_data(),
                      headings=["Ticker", "Shares", "Avg Price", "Current Price", "Current Value", "Gain/Loss"],
                      auto_size_columns=False,
                      col_widths=[8, 10, 12, 12, 15, 15],
                      justification='right',
                      num_rows=10,
                      font=("Helvetica", 10),
                      background_color='#FFFFFF',
                      text_color='#000000',
                      alternating_row_color='#E6F2FF',
                      key='-PORTFOLIO_TABLE-',
                      tooltip='Stock Holdings',
                      pad=((5, 5), (10, 10)))]
        ], pad=((0, 0), (0, 20)), relief=sg.RELIEF_SOLID, title_color='#000000', border_width=1)],

        [sg.Frame("Buy Stocks", [
            [sg.Text("Stock Symbol:", font=("Helvetica", 10), pad=((5, 5), (10, 10))),
             sg.InputText(key='-STOCK_SYMBOL-', size=(15, 1), font=("Helvetica", 10), pad=((5, 10), (10, 10))),
             sg.Text("Amount ($):", font=("Helvetica", 10), pad=((5, 5), (10, 10))),
             sg.InputText(key='-STOCK_AMOUNT-', size=(15, 1), font=("Helvetica", 10), pad=((5, 10), (10, 10))),
             sg.Button("Buy", key='-BUY_STOCK-', size=(10, 1), font=("Helvetica", 10),
                       button_color=(COLORS['colour_black'], COLORS['primary']), pad=((5, 10), (10, 10)))]
        ], relief=sg.RELIEF_SOLID,title_color='#000000', border_width=1)],
        [sg.Frame("Sell Stocks", [
            [sg.Text("Stock Symbol:", font=("Helvetica", 10), pad=((5, 5), (10, 10))),
             sg.Combo([], key='-SELL_STOCK_SYMBOL-', size=(15, 1), font=("Helvetica", 10),
                      pad=((5, 10), (10, 10)), readonly=True),
             sg.Radio("Sell Shares:", "SELL_TYPE", default=True, key='-SELL_SHARES_RADIO-',
                      font=("Helvetica", 10), pad=((5, 5), (10, 10)))],
            [sg.Text("Number of Shares:", font=("Helvetica", 10), pad=((5, 5), (10, 10))),
             sg.InputText(key='-SELL_SHARES-', size=(15, 1), font=("Helvetica", 10),
                          pad=((5, 10), (10, 10))),
             sg.Radio("Sell Amount ($):", "SELL_TYPE", key='-SELL_AMOUNT_RADIO-',
                      font=("Helvetica", 10), pad=((5, 5), (10, 10)))],
            [sg.Text("Dollar Amount:", font=("Helvetica", 10), pad=((5, 5), (10, 10))),
             sg.InputText(key='-SELL_AMOUNT-', size=(15, 1), font=("Helvetica", 10),
                          pad=((5, 10), (10, 10))),
             sg.Checkbox("Sell All", key='-SELL_ALL-', font=("Helvetica", 10),
                         pad=((5, 5), (10, 10)))],
            [sg.Button("Sell", key='-SELL_STOCK-', size=(10, 1), font=("Helvetica", 10),
                       button_color=(COLORS['colour_black'], COLORS['error']),
                       pad=((5, 10), (10, 10)))]
        ], relief=sg.RELIEF_SOLID,title_color='#000000', border_width=1, pad=((0, 0), (20, 0)))]
    ]
    stock_analysis_layout = [
        [sg.Text("Stock Analysis", font=("Helvetica", 18, "bold"), pad=((0, 0), (0, 20)),
                 text_color=COLORS['text_primary'])],
        [sg.Frame("Price Information", [
            [sg.Text("Stock Symbol:", font=("Helvetica", 10), pad=((5, 5), (10, 10))),
             sg.InputText(key='-ANALYSIS_SYMBOL-', size=(15, 1), font=("Helvetica", 10),
                          pad=((5, 10), (10, 10))),
             sg.Button("Get Price", key='-GET_PRICE-', size=(10, 1), font=("Helvetica", 10),
                       button_color=(COLORS['colour_black'], COLORS['primary']), pad=((5, 10), (10, 10))),
             sg.Button("Predict", key='-PREDICT_PRICE-', size=(10, 1), font=("Helvetica", 10),
                       button_color=(COLORS['colour_black'], COLORS['secondary']), pad=((5, 10), (10, 10)))]
        ], pad=((0, 0), (0, 20)), relief=sg.RELIEF_SOLID,title_color='#000000', border_width=1)],
        [sg.Frame("Price Chart", [
            # [sg.Canvas(size=(500, 300), key='-CHART_CANVAS-', pad=((10, 10), (10, 10)))]
        ], pad=((0, 0), (0, 20)), relief=sg.RELIEF_SOLID,title_color='#000000', border_width=1)],
        [sg.Frame("Analysis Results", [
            [sg.Multiline(size=(70, 8), key='-ANALYSIS_RESULTS-', disabled=True, font=("Helvetica", 10),
                          background_color=COLORS['card_background'], text_color=COLORS['text_primary'],
                          pad=((10, 10), (10, 10)))]
        ], relief=sg.RELIEF_SOLID, title_color='#000000',border_width=1)]
    ]

    # layout for the Settings section
    settings_layout = [
        [sg.Text("Settings", font=("Helvetica", 18, "bold"), pad=((0, 0), (0, 20)),
                 text_color=COLORS['text_primary'])],
        [sg.Frame("User Information", [
            [sg.Text("Your Name:", font=("Helvetica", 10), pad=((5, 5), (10, 10))),
             sg.InputText(key='-USER_NAME-', size=(20, 1), default_text=user_name, font=("Helvetica", 10),
                          pad=((5, 10), (10, 10))),
             sg.Button("Save", key='-SAVE_NAME-', size=(10, 1), font=("Helvetica", 10),
                       button_color=(COLORS['colour_black'], COLORS['primary']), pad=((5, 10), (10, 10)))]
        ], pad=((0, 0), (0, 20)), relief=sg.RELIEF_SOLID, title_color='#000000',border_width=1)],
        [sg.Frame("Feedback and Ratings", [
            #[sg.Button("View Feedback Statistics", key='-VIEW_STATS-', size=(20, 1), font=("Helvetica", 10),
                       #button_color=(COLORS['colour_black'], COLORS['secondary']), pad=((10, 10), (10, 10))),
             #sg.Button("View Poor Ratings", key='-VIEW_POOR-', size=(20, 1), font=("Helvetica", 10),
                       #button_color=(COLORS['colour_black'], COLORS['accent']), pad=((10, 10), (10, 10)))]
        ], pad=((0, 0), (0, 20)), relief=sg.RELIEF_SOLID, title_color='#000000',border_width=1)],
        [sg.Frame("Prediction Model Settings", [
            [sg.Text("Default Model:", font=("Helvetica", 10), pad=((5, 5), (10, 5)))],
            [sg.Radio("LSTM", "MODEL", key='-MODEL_LSTM-', font=("Helvetica", 10),
                      pad=((10, 10), (5, 10))),
             sg.Radio("CNN", "MODEL", default=True, key='-MODEL_CNN-', font=("Helvetica", 10), pad=((10, 10), (5, 10))),
             sg.Radio("Both", "MODEL", key='-MODEL_BOTH-', font=("Helvetica", 10), pad=((10, 10), (5, 10)))]
        ], relief=sg.RELIEF_SOLID, title_color='#000000',border_width=1)]
    ]

    # Main layout with sidebar
    layout = [
        [sg.Column([
            [sg.Text("FINBOT", size=(20, 2), font=("Helvetica", 20, "bold"), justification='left',
                     text_color=COLORS['primary'], pad=((10, 0), (10, 20)))],
            [sg.Button("💬 Chat", key='-NAV_CHAT-', size=(20, 2),
                       button_color=(COLORS['colour_black'], COLORS['primary']),
                       font=("Helvetica", 10, "bold"), pad=((5, 5), (5, 5)))],
            [sg.Button("📊 Portfolio", key='-NAV_PORTFOLIO-', size=(20, 2),
                       button_color=(COLORS['colour_black'], COLORS['background_dark']),
                       font=("Helvetica", 10), pad=((5, 5), (5, 5)))],
            [sg.Button("📈 Stock Analysis", key='-NAV_ANALYSIS-', size=(20, 2),
                       button_color=(COLORS['colour_black'], COLORS['background_dark']),
                       font=("Helvetica", 10), pad=((5, 5), (5, 5)))],
            [sg.Button("⚙️ Settings", key='-NAV_SETTINGS-', size=(20, 2),
                       button_color=(COLORS['colour_black'], COLORS['background_dark']),
                       font=("Helvetica", 10), pad=((5, 5), (5, 5)))],
            [sg.Text("Financial Literacy Level:", font=("Helvetica", 10), pad=((10, 10), (50, 5)))],
            [sg.Combo(["Beginner", "Advanced"], default_value=user_financial_literacy,
                      key='-LITERACY_LEVEL-', size=(18, 1), font=("Helvetica", 10),
                      readonly=True, pad=((10, 10), (0, 10)),
                      background_color=COLORS['card_background'], text_color=COLORS['text_primary'])],
            [sg.Button("🚪 Exit", key='-EXIT-', size=(20, 2),
                       button_color=(COLORS['colour_black'], COLORS['error']),
                       font=("Helvetica", 10, "bold"), pad=((5, 5), (50, 5)))]
        ], background_color=COLORS['background_medium'], size=(200, 600), pad=0),

            sg.VSeparator(),  # vertical separator for visual clarity

            sg.Column([
                # Chat section
                [sg.pin(sg.Column(chat_layout, key='-SECTION_CHAT-', visible=True,
                                  pad=((20, 20), (20, 20)), background_color=COLORS['background_light']))],

                # Portfolio section
                [sg.pin(sg.Column(portfolio_layout, key='-SECTION_PORTFOLIO-', visible=False,
                                  pad=((20, 20), (20, 20)), background_color=COLORS['background_light']))],

                # Stock Analysis section
                [sg.pin(sg.Column(stock_analysis_layout, key='-SECTION_ANALYSIS-', visible=False,
                                  pad=((20, 20), (20, 20)), background_color=COLORS['background_light']))],

                # Settings section
                [sg.pin(sg.Column(settings_layout, key='-SECTION_SETTINGS-', visible=False,
                                  pad=((20, 20), (20, 20)), background_color=COLORS['background_light']))]
            ], background_color=COLORS['background_light'], pad=0)]
    ]

    window = sg.Window("Finbot - Financial Assistant", layout, finalize=True, size=(1000, 600),
                       resizable=True, margins=(0, 0), background_color=COLORS['background_light'])

    # Chat section with a welcome message
    welcome_message = "- Finbot: Hi, I'm Finbot. I'm here to help you with financial information and analysis."
    add_message_to_chat(window, 'bot', welcome_message)
    add_message_to_chat(window, 'bot', "- Finbot: Please enter 'bye' if you want to say goodbye.")
    add_message_to_chat(window, 'bot', "- Finbot: If you need assistance, enter 'help'.")
    add_message_to_chat(window, 'bot', "- Finbot: After each response, you can rate if it was helpful!")

    # Get user name at startup
    user_name_input = sg.popup_get_text("May I have your name?", "Name Input", default_text="(User)",
                                        font=("Helvetica", 10), text_color=COLORS['text_primary'],
                                        background_color=COLORS['background_light'])
    if user_name_input:
        user_name = user_name_input
        add_message_to_chat(window, 'bot', f"- Finbot: Hi, {user_name}, glad to meet you! How can I help you?")
        window['-USER_NAME-'].update(user_name)
        if user_name.lower() == 'finbot':
            add_message_to_chat(window, 'bot',
                                "- Finbot: Oh, cool!! we have the same name... this might get confusing.")

    # Active section tracker
    active_section = '-SECTION_CHAT-'
    active_nav = '-NAV_CHAT-'

    # API key
    api_key = '9491467042934eeb9a7fa58400031b8a'

    # Function to capture bot response
    def capture_bot_response(output_text):
        nonlocal last_bot_response
        last_bot_response = output_text.strip()
        return True

    # Event loop
    while True:
        event, values = window.read()

        # Exit events
        if event == sg.WIN_CLOSED or event == '-EXIT-':
            add_message_to_chat(window, 'bot', "- Finbot: Bye!")
            window.close()
            break

        # Navigation events
        if event == '-NAV_CHAT-':
            window[active_section].update(visible=False)
            window['-SECTION_CHAT-'].update(visible=True)
            window[active_nav].update(button_color=(COLORS['colour_black'], COLORS['background_dark']))
            window['-NAV_CHAT-'].update(button_color=(COLORS['colour_black'], COLORS['primary']))
            active_section = '-SECTION_CHAT-'
            active_nav = '-NAV_CHAT-'

        elif event == '-NAV_PORTFOLIO-':
            window[active_section].update(visible=False)
            window['-SECTION_PORTFOLIO-'].update(visible=True)
            window[active_nav].update(button_color=(COLORS['colour_black'], COLORS['background_dark']))
            window['-NAV_PORTFOLIO-'].update(button_color=(COLORS['colour_black'], COLORS['primary']))
            active_section = '-SECTION_PORTFOLIO-'
            active_nav = '-NAV_PORTFOLIO-'

            # Update portfolio data
            window['-WALLET_BALANCE-'].update(f"${user_wallet:.2f}")
            window['-PORTFOLIO_TABLE-'].update(values=get_portfolio_data())

            # Update the dropdown with current portfolio stocks
            window['-SELL_STOCK_SYMBOL-'].update(values=list(user_portfolio.keys()))

        elif event == '-NAV_ANALYSIS-':
            window[active_section].update(visible=False)
            window['-SECTION_ANALYSIS-'].update(visible=True)
            window[active_nav].update(button_color=(COLORS['colour_black'], COLORS['background_dark']))
            window['-NAV_ANALYSIS-'].update(button_color=(COLORS['colour_black'], COLORS['primary']))
            active_section = '-SECTION_ANALYSIS-'
            active_nav = '-NAV_ANALYSIS-'

        elif event == '-NAV_SETTINGS-':
            window[active_section].update(visible=False)
            window['-SECTION_SETTINGS-'].update(visible=True)
            window[active_nav].update(button_color=(COLORS['colour_black'], COLORS['background_dark']))
            window['-NAV_SETTINGS-'].update(button_color=(COLORS['colour_black'], COLORS['primary']))
            active_section = '-SECTION_SETTINGS-'
            active_nav = '-NAV_SETTINGS-'

        # Handle rating
        elif event == '-RATE_GOOD-':
            if last_bot_response:
                save_feedback(last_user_query, last_bot_response, "Good")
                add_message_to_chat(window, 'bot',
                                    "- Finbot: Thanks for the positive feedback! I'm glad that was helpful.")

        elif event == '-RATE_POOR-':
            if last_bot_response:
                # Ask for additional comments
                comments = sg.popup_get_text("What could have been better?", "Feedback Comments",
                                             font=("Helvetica", 10), background_color=COLORS['background_light'])
                save_feedback(last_user_query, last_bot_response, "Poor", comments)

                # Also save to the poor_table.xlsx file
                save_poor_rating_to_excel(last_user_query, last_bot_response, comments)

                add_message_to_chat(window, 'bot',
                                    "- Finbot: Thanks for your feedback. I'll try to do better next time.")

        # Feedback stats
        elif event == '-FEEDBACK_STATS-' or event == '-VIEW_STATS-':
            stats = get_feedback_stats()
            add_message_to_chat(window, 'bot', f"- Finbot: {stats}")

            # Also show information about poor ratings
            poor_ratings_file = "feedback/poor_table.xlsx"
            if os.path.exists(poor_ratings_file):
                try:
                    df = pd.read_excel(poor_ratings_file)
                    poor_count = len(df)
                    add_message_to_chat(window, 'bot',
                                        f"- Finbot: There are {poor_count} responses rated as 'Poor'.")
                    add_message_to_chat(window, 'bot',
                                        f"- Finbot: You can view detailed poor ratings in {poor_ratings_file}")
                except Exception as e:
                    add_message_to_chat(window, 'bot', f"- Finbot: Error reading poor ratings file: {str(e)}")
            else:
                add_message_to_chat(window, 'bot', "- Finbot: No poor ratings recorded yet.")

        # View poor ratings
        elif event == '-VIEW_POOR_RATINGS-' or event == '-VIEW_POOR-':
            view_poor_ratings()

        # Send message handling
        elif event == "Send":
            # Get the user input
            ui = values['-USER_INPUT-']
            if not ui.strip():  # Skip empty inputs
                continue

            # Store user query for potential feedback
            last_user_query = ui

            # Clear input field before processing
            window['-USER_INPUT-'].update('')
            window.refresh()

            # Display user message in chat using the new bubble style
            add_message_to_chat(window, 'user', ui, user_name)

            # Check if financial literacy level has changed
            new_level = None
            if values['-LITERACY_LEVEL-'] != user_financial_literacy:
                new_level = values['-LITERACY_LEVEL-']

            # Update literacy level if changed
            if new_level:
                user_financial_literacy = new_level
                add_message_to_chat(window, 'bot',
                                    f"- Finbot: Your financial literacy level is now **{user_financial_literacy}**.")
                has_switched_level = False  # Reset flag for next switch

            # Handle bye command
            if ui.lower() == 'bye':
                add_message_to_chat(window, 'bot', "- Finbot: Bye!")
                break

            # Handle feedback stats command
            if ui.lower() in ['feedback stats', 'show feedback', 'rating stats']:
                stats = get_feedback_stats()
                add_message_to_chat(window, 'bot', f"- Finbot: {stats}")
                capture_bot_response(stats)
                continue

            # Apply spell correction
            ui = ' '.join([correct(word) for word in ui.split()])

            # Check for name recognition
            response = name_response(ui, threshold=0.9)
            if response != 'NOT FOUND':
                response_text = f"You're {user_name}, I have a great memory ┑(￣u ￣)┍"
                add_message_to_chat(window, 'bot', f"- Finbot: {response_text}")
                capture_bot_response(response_text)
                continue

            # Check for investment advice
            if 'invest in' in ui.lower() and 'should i' in ui.lower():
                words = ui.split()
                if 'in' in words:
                    ticker_index = words.index('in') + 1
                    if ticker_index < len(words):
                        ticker = words[ticker_index].upper()

                        # Print message to let user know advice is being generated
                        add_message_to_chat(window, 'bot',
                                            f"- Finbot: Analyzing investment potential for {ticker}. A new window will open shortly.")
                        add_message_to_chat(window, 'bot', f"- Finbot: This may take a few minutes. Please wait...")

                        # Get investment advice (this will open a new window)
                        advice = get_investment_advice(ticker)

                        # Print a summary in the main chat
                        add_message_to_chat(window, 'bot', f"- Finbot: Investment analysis for {ticker} complete.")
                        first_line = advice.split('\n')[0]
                        add_message_to_chat(window, 'bot', f"- Finbot: {first_line}")

                        # Get sentiment analysis
                        ticker_symbol = ticker
                        news_headlines = fetch_news(api_key, ticker_symbol)
                        sentiment_scores = analyze_sentiment(news_headlines)
                        average_sentiment = calculate_overall_sentiment(sentiment_scores)
                        sentiment_advice = get_investment_advice_based_on_sentiment(average_sentiment)

                        add_message_to_chat(window, 'bot', f"- Finbot: News sentiment analysis: {sentiment_advice}")
                        add_message_to_chat(window, 'bot',
                                            f"- Finbot: Average sentiment score: {average_sentiment:.2f} (-1 to +1 scale)")

                        # Capture the response for potential feedback
                        capture_bot_response(advice)
                    else:
                        response_text = "Please specify the stock ticker correctly, e.g., 'Should I invest in AAPL'."
                        add_message_to_chat(window, 'bot', f"- Finbot: {response_text}")
                        capture_bot_response(response_text)
                continue

            # Check for wallet balance
            if 'check balance' in ui.lower() or 'wallet balance' in ui.lower() or 'how much money' in ui.lower() or 'my wallet' in ui.lower() or 'the wallet' in ui.lower():
                # Use stdout redirection to capture output from the function
                original_stdout = sys.stdout
                captured_output = io.StringIO()
                sys.stdout = captured_output

                check_wallet_balance()

                # Restore stdout and get output
                sys.stdout = original_stdout
                result = captured_output.getvalue()

                add_message_to_chat(window, 'bot', result)
                capture_bot_response(result)
                continue

            # Check for add money request
            if 'add money' in ui.lower() or 'add to wallet' in ui.lower():
                amount = sg.popup_get_text("- Finbot: How much money would you like to add to your wallet?",
                                           "Add Money", font=("Helvetica", 10),
                                           background_color=COLORS['background_light'])
                if amount and amount.replace('.', '', 1).isdigit():
                    # Use stdout redirection to capture output
                    original_stdout = sys.stdout
                    captured_output = io.StringIO()
                    sys.stdout = captured_output

                    add_to_wallet(float(amount))

                    # Restore stdout and get output
                    sys.stdout = original_stdout
                    result = captured_output.getvalue()

                    add_message_to_chat(window, 'bot', result)
                    window['-WALLET_BALANCE-'].update(f"${user_wallet:.2f}")
                    capture_bot_response(result)
                else:
                    response_text = "Please enter a valid number."
                    add_message_to_chat(window, 'bot', f"- Finbot: {response_text}")
                    capture_bot_response(response_text)
                continue

            # Check for buy stock request
            if 'buy stock' in ui.lower():
                ticker = sg.popup_get_text(
                    "Which company stock would you like to buy? Please provide the ticker symbol.",
                    "Buy Stock", font=("Helvetica", 10),
                    background_color=COLORS['background_light'])
                amount = sg.popup_get_text("How much money would you like to spend on this purchase?",
                                           "Purchase Amount", font=("Helvetica", 10),
                                           background_color=COLORS['background_light'])
                if ticker and amount and amount.replace('.', '', 1).isdigit():

                    # Using stdout redirection to capture output
                    original_stdout = sys.stdout
                    captured_output = io.StringIO()
                    sys.stdout = captured_output

                    buy_stock(ticker.upper(), float(amount))

                    # Restore stdout and get output
                    sys.stdout = original_stdout
                    result = captured_output.getvalue()

                    add_message_to_chat(window, 'bot', result)
                    window['-WALLET_BALANCE-'].update(f"${user_wallet:.2f}")
                    window['-PORTFOLIO_TABLE-'].update(values=get_portfolio_data())
                    # Also update the sell stock dropdown
                    window['-SELL_STOCK_SYMBOL-'].update(values=list(user_portfolio.keys()))
                    capture_bot_response(result)
                else:
                    response_text = "Please provide valid input for both the ticker and amount."
                    add_message_to_chat(window, 'bot', f"- Finbot: {response_text}")
                    capture_bot_response(response_text)
                continue

            # Check for name change
            if check_name_change(ui):
                user_name = name_change(ui)
                response_text = f"Hi, {user_name}"
                add_message_to_chat(window, 'bot', f"- Finbot: {response_text}")
                window['-USER_NAME-'].update(user_name)
                capture_bot_response(response_text)
                continue

            # Check for prediction request
            if 'predict' in ui and 'stock' in ui:
                company = sg.popup_get_text("What is the ticker symbol of the company you want to predict?",
                                            "Predict Stock", font=("Helvetica", 10),
                                            background_color=COLORS['background_light'])
                if company:
                    model_choice = sg.popup_get_text("Which model(s) do you want to use (LSTM, CNN, Both)?",
                                                     "Model Choice", font=("Helvetica", 10),
                                                     background_color=COLORS['background_light'])
                    model_types = []
                    if model_choice:
                        if model_choice.lower() == 'both':
                            model_types = ['LSTM', 'CNN']
                        elif model_choice.upper() in ['LSTM', 'CNN']:
                            model_types = [model_choice.upper()]

                        if model_types:
                            # Print message about prediction running
                            add_message_to_chat(window, 'bot',
                                                f"- Finbot: Starting prediction for {company}. A new window will open shortly.")
                            add_message_to_chat(window, 'bot',
                                                f"- Finbot: This may take a few minutes. Please wait...")

                            # Call the prediction function
                            predictions = predict_stock_price_with_xai(company, model_types)

                            # Report results back in chat
                            if predictions:
                                add_message_to_chat(window, 'bot',
                                                    f"- Finbot: Prediction results for {company}: ")
                                for model_type, prediction in predictions.items():
                                    if model_type not in ['LSTM_historical', 'CNN_historical']:
                                        if isinstance(prediction, (int, float)):
                                            add_message_to_chat(window, 'bot',
                                                                f"- Finbot: {model_type} model predicts next day's stock price of {company} as ${prediction:.2f}")
                                        elif isinstance(prediction, np.ndarray) and len(prediction) > 0:
                                            add_message_to_chat(window, 'bot',
                                                                f"- Finbot: {model_type} model predicts next day's stock price of {company} as ${prediction[0]:.2f}")
                                        else:
                                            add_message_to_chat(window, 'bot',
                                                                f"- Finbot: Analysis complete for {company} using {model_type} model.")
                            else:
                                add_message_to_chat(window, 'bot',
                                                    f"- Finbot: Could not generate predictions for {company}.")

                            # Capture final response for feedback
                            capture_bot_response(f"Prediction results for {company}")
                        else:
                            response_text = "Invalid model choice. Please choose LSTM, CNN, or Both."
                            add_message_to_chat(window, 'bot', f"- Finbot: {response_text}")
                            capture_bot_response(response_text)
                    else:
                        response_text = "No model selected. Prediction cancelled."
                        add_message_to_chat(window, 'bot', f"- Finbot: {response_text}")
                        capture_bot_response(response_text)
                else:
                    response_text = "No ticker symbol provided. Prediction cancelled."
                    add_message_to_chat(window, 'bot', f"- Finbot: {response_text}")
                    capture_bot_response(response_text)
                continue

            # Check for current price request
            if 'current price' in ui.lower() or 'stock price' in ui.lower():
                words = ui.split()
                if 'price' in words:
                    ticker_index = words.index('price') + 2 if 'price' in words else None
                    if ticker_index and ticker_index < len(words):
                        ticker = words[ticker_index]
                        current_price = get_current_stock_price(ticker)
                        if isinstance(current_price, str):
                            add_message_to_chat(window, 'bot', f"- Finbot: {current_price}")
                            capture_bot_response(current_price)
                        else:
                            response_text = f"The current price for {ticker.upper()} is: ${current_price:.2f}"
                            add_message_to_chat(window, 'bot', f"- Finbot: {response_text}")
                            capture_bot_response(response_text)
                    else:
                        response_text = "Please tell me the ticker symbol of the stock."
                        add_message_to_chat(window, 'bot', f"- Finbot: {response_text}")
                        capture_bot_response(response_text)
                continue

            # Check for portfolio request
            if 'portfolio' in ui.lower():
                # Use stdout redirection to capture output
                original_stdout = sys.stdout
                captured_output = io.StringIO()
                sys.stdout = captured_output

                list_portfolio()

                # Restore stdout and get output
                sys.stdout = original_stdout
                result = captured_output.getvalue()

                add_message_to_chat(window, 'bot', result)
                capture_bot_response(result)
                continue

            # Check for time request
            if 'time' in ui and not any(
                    phrase in ui.lower() for phrase in ['add time', 'more time', 'some time']):
                # Use stdout redirection to capture output
                original_stdout = sys.stdout
                captured_output = io.StringIO()
                sys.stdout = captured_output

                time_response('time')

                # Restore stdout and get output
                sys.stdout = original_stdout
                result = captured_output.getvalue()

                # Replace "Skynet" with "Finbot" in the output
                result = result.replace("- Skynet:", "- Finbot:")

                add_message_to_chat(window, 'bot', result)
                capture_bot_response(result)
                continue

            # Check for date request
            if 'today' in ui:
                # Use stdout redirection to capture output
                original_stdout = sys.stdout
                captured_output = io.StringIO()
                sys.stdout = captured_output

                time_response('today')

                # Restore stdout and get output
                sys.stdout = original_stdout
                result = captured_output.getvalue()

                # Replace "Skynet" with "Finbot" in the output
                result = result.replace("- Skynet:", "- Finbot:")

                add_message_to_chat(window, 'bot', result)
                capture_bot_response(result)
                continue

            # Check for help request
            if ui.lower() == 'help':
                response_text = "Here's how I can assist you:\n\n"
                response_text += "1. Check Wallet Balance:\n"
                response_text += "   - Example: 'How much money in my wallet?'\n"
                response_text += "   - Example: 'What's my wallet balance?'\n"
                response_text += "   - Example: 'Check my balance'\n\n"

                response_text += "2. Add Money to Wallet:\n"
                response_text += "   - Example: 'Add $500 to my wallet'\n"
                response_text += "   - Example: 'Can I put $200 into my wallet?'\n"
                response_text += "   - Example: 'Deposit $1000 to my wallet'\n\n"

                response_text += "3. Buy Stocks:\n"
                response_text += "   - Example: 'Buy stock AAPL with $1000'\n"
                response_text += "   - Example: 'Can you purchase shares of TSLA for $500?'\n"
                response_text += "   - Example: 'I'd like to buy 10 shares of AMZN'\n\n"

                response_text += "4. Predict Stock Price:\n"
                response_text += "   - Example: 'Predict stock price of Tesla using LSTM'\n"
                response_text += "   - Example: 'What's the future price of AAPL with CNN?'\n"
                response_text += "   - Example: 'Can you forecast TSLA's price tomorrow using both models?'\n\n"

                response_text += "5. Get Investment Advice:\n"
                response_text += "   - Example: 'Should I invest in AAPL?'\n"
                response_text += "   - Example: 'Is TSLA a good investment?'\n"
                response_text += "   - Example: 'Do you recommend buying shares in GOOG?'\n\n"

                response_text += "6. Check Current Stock Prices:\n"
                response_text += "   - Example: 'What's the current price of TSLA?'\n"
                response_text += "   - Example: 'How much is AAPL stock worth right now?'\n"
                response_text += "   - Example: 'Show me the price of GOOGL'\n\n"

                response_text += "7. View Portfolio:\n"
                response_text += "   - Example: 'Show me my portfolio'\n"
                response_text += "   - Example: 'What stocks do I own?'\n"
                response_text += "   - Example: 'What's in my stock portfolio?'\n\n"

                response_text += "8. Small Talk & FAQs:\n"
                response_text += "   - Example: 'Hi, how are you?'\n"
                response_text += "   - Example: 'Tell me a joke'\n"
                response_text += "   - Example: 'What time is it?'\n"
                response_text += "   - Example: 'What do you know about Tesla?'\n\n"

                response_text += "Try these out to get started!"

                add_message_to_chat(window, 'bot', f"- Finbot: {response_text}")
                capture_bot_response(response_text)
                continue

            # Try small talk module
            response = talk_response(ui, threshold=0.9)
            if response != 'NOT FOUND':
                add_message_to_chat(window, 'bot', f"- Finbot: {response}")

                # Track matched question and answer for feedback
                track_matched_qa(ui, response, "small_talk.csv")

                capture_bot_response(response)
                continue

            # Try Q&A module
            response = answer_Q(ui, threshold=0.1, user_financial_literacy=user_financial_literacy)
            if response != 'NOT FOUND':
                add_message_to_chat(window, 'bot', f"- Finbot: {response}")

                # Set source and track matched question and answer
                last_matched_source = "question_answering.xlsx"
                track_matched_qa(ui, response, "question_answering.xlsx")

                capture_bot_response(response)
            else:
                # Default response if no match is found
                default_response = "I'm sorry ˙◠˙ I don't quite understand. Try asking me to predict a stock price or inquire about market trends."
                add_message_to_chat(window, 'bot', f"- Finbot: {default_response}")

                # Clear matched question and answer when no match found
                last_matched_question = ""
                last_matched_answer = ""
                last_matched_source = ""

                capture_bot_response(default_response)

            # Portfolio section events
        elif event == '-ADD_FUNDS-':
            amount = sg.popup_get_text("How much would you like to add to your wallet?",
                                       "Add Funds", font=("Helvetica", 10),
                                       background_color=COLORS['background_light'])
            if amount and amount.replace('.', '', 1).isdigit():
                # Use stdout redirection to capture output
                original_stdout = sys.stdout
                captured_output = io.StringIO()
                sys.stdout = captured_output

                add_to_wallet(float(amount))

                # Restore stdout and get output
                sys.stdout = original_stdout
                result = captured_output.getvalue()

                window['-WALLET_BALANCE-'].update(f"${user_wallet:.2f}")
                sg.popup(result, title="Wallet Update", font=("Helvetica", 10),
                         background_color=COLORS['background_light'])

        elif event == '-UPDATE_FUNDS-':
            amount = values['-FUND_AMOUNT-']
            if amount and amount.replace('.', '', 1).isdigit():
                # Use stdout redirection to capture output
                original_stdout = sys.stdout
                captured_output = io.StringIO()
                sys.stdout = captured_output

                add_to_wallet(float(amount))

                # Restore stdout and get output
                sys.stdout = original_stdout
                result = captured_output.getvalue()

                window['-WALLET_BALANCE-'].update(f"${user_wallet:.2f}")
                window['-FUND_AMOUNT-'].update('')
                sg.popup(result, title="Wallet Update", font=("Helvetica", 10),
                         background_color=COLORS['background_light'])

        elif event == '-BUY_STOCK-':
            symbol = values['-STOCK_SYMBOL-'].strip().upper()
            amount = values['-STOCK_AMOUNT-']

            if symbol and amount and amount.replace('.', '', 1).isdigit():
                # Use stdout redirection to capture output
                original_stdout = sys.stdout
                captured_output = io.StringIO()
                sys.stdout = captured_output

                buy_stock(symbol, float(amount))

                # Restore stdout and get output
                sys.stdout = original_stdout
                result = captured_output.getvalue()

                window['-WALLET_BALANCE-'].update(f"${user_wallet:.2f}")
                window['-PORTFOLIO_TABLE-'].update(values=get_portfolio_data())
                window['-STOCK_SYMBOL-'].update('')
                window['-STOCK_AMOUNT-'].update('')
                # Update sell stock dropdown
                window['-SELL_STOCK_SYMBOL-'].update(values=list(user_portfolio.keys()))
                sg.popup(result, title="Stock Purchase", font=("Helvetica", 10),
                         background_color=COLORS['background_light'])

        elif event == '-SELL_STOCK-':
            symbol = values['-SELL_STOCK_SYMBOL-']
            sell_all = values['-SELL_ALL-']
            shares_radio = values['-SELL_SHARES_RADIO-']
            amount_radio = values['-SELL_AMOUNT_RADIO-']

            if not symbol:
                sg.popup("Please select a stock to sell", title="Error", font=("Helvetica", 10),
                         background_color=COLORS['background_light'])
                continue

            # Determine what to sell based on radio buttons and checkbox
            if sell_all:
                # Sell all shares
                shares_value = None
                amount_value = None
            elif shares_radio:
                # Sell specific number of shares
                shares_input = values['-SELL_SHARES-'].strip()
                if not shares_input or not shares_input.replace('.', '', 1).isdigit():
                    sg.popup("Please enter a valid number of shares", title="Error", font=("Helvetica", 10),
                             background_color=COLORS['background_light'])
                    continue
                shares_value = float(shares_input)
                amount_value = None
            elif amount_radio:
                # Sell specific dollar amount
                amount_input = values['-SELL_AMOUNT-'].strip()
                if not amount_input or not amount_input.replace('.', '', 1).isdigit():
                    sg.popup("Please enter a valid dollar amount", title="Error", font=("Helvetica", 10),
                             background_color=COLORS['background_light'])
                    continue
                shares_value = None
                amount_value = float(amount_input)

            # Use stdout redirection to capture output
            original_stdout = sys.stdout
            captured_output = io.StringIO()
            sys.stdout = captured_output

            # Attempt to sell the stock
            sell_stock(symbol, shares_value, amount_value)

            # Restore stdout and get output
            sys.stdout = original_stdout
            result = captured_output.getvalue()

            # Update the UI
            window['-WALLET_BALANCE-'].update(f"${user_wallet:.2f}")
            window['-PORTFOLIO_TABLE-'].update(values=get_portfolio_data())
            window['-SELL_STOCK_SYMBOL-'].update(values=list(user_portfolio.keys()))
            window['-SELL_SHARES-'].update('')
            window['-SELL_AMOUNT-'].update('')
            window['-SELL_ALL-'].update(False)

            # Show the result in a popup
            sg.popup(result, title="Stock Sale", font=("Helvetica", 10),
                     background_color=COLORS['background_light'])

        # Add these event handlers for UI improvements
        elif event == '-SELL_SHARES_RADIO-':
            window['-SELL_ALL-'].update(False)  # Uncheck "Sell All"

        elif event == '-SELL_AMOUNT_RADIO-':
            window['-SELL_ALL-'].update(False)  # Uncheck "Sell All"

        elif event == '-SELL_ALL-':
            if values['-SELL_ALL-']:
                # If "Sell All" is checked, clear and disable other inputs
                window['-SELL_SHARES-'].update('')
                window['-SELL_AMOUNT-'].update('')

            # Stock Analysis section events
        elif event == '-GET_PRICE-':
            symbol = values['-ANALYSIS_SYMBOL-'].strip().upper()
            if symbol:
                price = get_current_stock_price(symbol)
                if isinstance(price, (int, float)):
                    # Update results area
                    window['-ANALYSIS_RESULTS-'].update(f"Current price of {symbol}: ${price:.2f}")

                    # Create a simple chart with improved styling
                    try:
                        data = yf.download(symbol, period="1mo")
                        plt.style.use('ggplot')
                        plt.figure(figsize=(10, 4))
                        plt.plot(data['Close'], color=COLORS['primary'], linewidth=2)
                        plt.title(f"{symbol} Price - Last Month", fontsize=14, fontweight='bold')
                        plt.ylabel("Price ($)", fontsize=12)
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()


                        # Save chart to temp file for display
                        chart_filename = "temp_chart.png"
                        plt.savefig(chart_filename)
                        plt.close()

                        # Show in popup since we can't directly attach to canvas
                        chart_layout = [
                            [sg.Image(filename=chart_filename)],
                            [sg.Button("Close", size=(10, 1),
                                       button_color=(COLORS['colour_black'], COLORS['primary']),
                                       font=("Helvetica", 10))]
                        ]
                        chart_window = sg.Window(f"{symbol} Chart", chart_layout, finalize=True,
                                                 background_color=COLORS['background_light'])

                        # Wait for window close
                        chart_window.read(close=True)

                    except Exception as e:
                        window['-ANALYSIS_RESULTS-'].update(f"Error creating chart: {str(e)}")
                else:
                    window['-ANALYSIS_RESULTS-'].update(price)  # Error message

        elif event == '-PREDICT_PRICE-':
            symbol = values['-ANALYSIS_SYMBOL-'].strip().upper()
            if symbol:
                # Determine which model to use
                model_types = []
                if values['-MODEL_LSTM-']:
                    model_types.append('LSTM')
                elif values['-MODEL_CNN-']:
                    model_types.append('CNN')
                elif values['-MODEL_BOTH-']:
                    model_types.extend(['LSTM', 'CNN'])
                else:
                    model_types.append('CNN')  # Default

                window['-ANALYSIS_RESULTS-'].update(
                    f"Predicting stock price for {symbol}...\nThis may take a few minutes. A new window will open with detailed results.")

                # Call prediction function
                predict_stock_price_with_xai(symbol, model_types, window, '-ANALYSIS_RESULTS-')

            # Settings section events
        elif event == '-SAVE_NAME-':
            new_name = values['-USER_NAME-'].strip()
            if new_name:
                user_name = new_name
                sg.popup(f"Name updated to: {user_name}", title="Settings Updated", font=("Helvetica", 10),
                         background_color=COLORS['background_light'])

            # Financial literacy level change
        elif event == '-LITERACY_LEVEL-':
            new_level = values['-LITERACY_LEVEL-']
            if new_level != user_financial_literacy:
                user_financial_literacy = new_level
                sg.popup(f"Financial literacy level is now {user_financial_literacy}", title="Settings Updated",
                         font=("Helvetica", 10), background_color=COLORS['background_light'])

            # Clean up any temporary files
        if os.path.exists('temp_chart.png'):
            try:
                os.remove('temp_chart.png')
            except:
                pass

   # Reset stdout and close the window if we ever exit the loop
    sys.stdout = sys.__stdout__
    window.close()


if __name__ == "__main__":
   create_finbot_app()