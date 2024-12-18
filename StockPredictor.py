import yfinance as yf
import datetime
import time
import requests
import io
from hmmlearn.hmm import GaussianHMM
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# step 1: a) download data sets from yahoo finance using yfinance library
# Problem: model does not converge
# Solution: need enough data to fit model, or it may lead to inaccurate transition probabilities in the hidden Markov process
# we tried 1y (model does not coverge), 5y, and 24y
data = yf.download("^GSPC", start="2000-01-01", end="2024-11-01")
data.head()
data.shape

# step 1: b) preprocessing
# In machine learning, we split the data into training data for the model
# and test data to reference against the predictions we made using our model
# this test data serves a way to compare actual vs predicted results to refine our model and eliminate entropy
# there are many methods to split the data, but for simplicity, in our examples, we tried the 80-20 and 66-33
train_size = int(0.8 * data.shape[0])
print('Here is the size of the training data', train_size)
train_data = data.iloc[0:train_size]
test_data = data.iloc[train_size+1:]

# step 1: c) features of opening and closing, low, high for fractional changes
# This will serve as a basis for predicting closing prices
def augment_features(dataframe):
    fracocp = ((dataframe['Close']-dataframe['Open'])/dataframe['Open']).values.flatten() # close
    frachp = ((dataframe['High']-dataframe['Open'])/dataframe['Open']).values.flatten() # high
    fraclp = ((dataframe['Open']-dataframe['Low'])/dataframe['Open']).values.flatten() # low

    new_dataframe = pd.DataFrame({'delOpenClose': fracocp,
                                 'delHighOpen': frachp,
                                 'delLowOpen': fraclp})
    new_dataframe.set_index(dataframe.index)
    return new_dataframe

def extract_features(dataframe):
    return np.column_stack((dataframe['delOpenClose'], dataframe['delHighOpen'], dataframe['delLowOpen']))

# To generate permutations of values for the features we take the Cartesian product across a range of values for each feature.
# We assume a few things here to reduce model complexity:
# 1. We assume that the distribution of each features is across an evenely spaced interval instead of being fully continuous
# 2. We assume possible values 50, 10, 10 for the start and end of the intervals
test_augmented = augment_features(test_data)
fracocp = test_augmented['delOpenClose']
frachp = test_augmented['delHighOpen']
fraclp = test_augmented['delLowOpen']

sample_space_fracocp = np.linspace(fracocp.min(), fracocp.max(), 50)
sample_space_fraclp = np.linspace(fraclp.min(), frachp.max(), 10)
sample_space_frachp = np.linspace(frachp.min(), frachp.max(), 10)

possible_outcomes = np.array(list(itertools.product(sample_space_fracocp, sample_space_frachp, sample_space_fraclp)))

### Checking predictions
# We use the data of the last 50 hidden days to predict the closing price of the current day, and we repeat those for 300 days (this value does not matter at all)
num_latent_days = 50
# num_days_to_predict = 300
num_days_to_predict = 4

# step 3: With observations, we will derive hidden factors in a Hidden Markov Model with hmmlearn machine learning library
#    a. These latent factors will vary from company to company, hard to fit one linear model of a certain subset of variables for all companies.
# Use GaussianHMM for gaussian emissions
# Fit the model with 10 hidden components (or states) to our training data.
# 10 is arbitrary, we can also do a grid search among a possible set of values for the number of hidden states to see which works the best (see advanced materials)

# model = GaussianHMM(n_components=10)
model = GaussianHMM(n_components=20)
feature_train_data = augment_features(train_data)
features_train = extract_features(feature_train_data)

features_train.shape # print the structure of the data

model.fit(features_train)

# step 4: Generate possible values for each of the features and then check how they score with a sequence of test data.
# step 5: The set of possible values that leads to the highest score is then used to predict the closing price for that day.
# For each of the days that we are going to predict closing prices for,
# we are going to take the test data for the previous num_latent_days and try each of the outcomes in possible_outcomes to see which sequence generates the highest score.
# The outcome that generates the highest score is then used to make the predictions for that day's closing price.
predicted_close_prices = []
for i in tqdm(range(num_days_to_predict)):
    # Calculate start and end indices
    previous_data_start_index = max(0, i - num_latent_days)
    previous_data_end_index = max(0, i)
    # Acquire test data features for these days
    previous_data = extract_features(augment_features(test_data.iloc[previous_data_start_index:previous_data_end_index]))
    
    outcome_scores = []
    for outcome in possible_outcomes:
        # Append each outcome one by one with replacement to see which sequence generates the highest score
        # total_data = np.row_stack((previous_data, outcome))
        total_data = np.vstack((previous_data, outcome))
        # outcome_scores.append(model.score(total_data))
        outcome_scores.append(np.vstack([model.score(total_data)]))
        
    # Take the most probable outcome as the one with the highest score
    most_probable_outcome = possible_outcomes[np.argmax(outcome_scores)]
    predicted_close_prices.append(test_data.iloc[i]['Open'] * (1 + most_probable_outcome[0]))

plt.figure(figsize=(30,10), dpi=80)
x_axis = np.array(test_data.index[0:num_days_to_predict], dtype='datetime64[ms]')
plt.plot(x_axis, test_data.iloc[0:num_days_to_predict]['Close'], 'b+-', label="Actual close prices")
plt.plot(x_axis, predicted_close_prices, 'ro-', label="Predicted close prices")
plt.legend()
plt.savefig(f'StockPredictorModel2/predictions_plot.png', format='png')
plt.show()

# step 6: plotting the error between the actual and predicted avalue to see how well our model fits
# and predicts the data. we can use these error to determine the best number of hidden states

ae = abs(test_data.iloc[0:num_days_to_predict]['Close'] - predicted_close_prices[0])

plt.figure(figsize=(30,10), dpi=80)

plt.plot(x_axis, ae, 'go-', label="Error")
plt.legend()
plt.savefig(f'StockPredictorModel2/error_plot.png', format='png')
plt.show()

# step 6: adjust range of values