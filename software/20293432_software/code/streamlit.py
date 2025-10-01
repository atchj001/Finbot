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