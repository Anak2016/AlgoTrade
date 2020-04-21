# import keras
import matplotlib.pyplot as plt
import math
import os
import random
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import tensorflow
from plotly import tools
from plotly.offline import iplot
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam


def sma():
    pass


# df['pandas_SMA_3'] = df.iloc[:, 1].rolling(window=3).mean()


class Agent:
    def __init__(self, state_size, is_eval=False, model_name=""):
        # os.makedirs('logs/scalars/')
        logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        os.makedirs(logdir)
        # self.tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(log_dir=logdir)
        # self.tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
        self.tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(
            log_dir=logdir)

        self.state_size = state_size  # normalized previous days
        self.action_size = 3  # sit, buy, sell
        self.memory = deque(maxlen=1000)
        self.inventory = []
        self.model_name = model_name
        self.is_eval = is_eval

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        # self.model = load_model("models/no exp replay/" + model_name) if is_eval else self._model()
        self.model = load_model(
            "models/" + model_name) if is_eval else self._model()
        self.loss_hist = []
        self.loss_val = None
        self.reward_hist = []

    # def get_loss_value(self):
    # 	history = self.model.fit()

    def _model(self):
        model = Sequential()
        model.add(Dense(units=64, input_dim=self.state_size, activation="relu"))
        model.add(Dense(units=32, activation="relu"))
        model.add(Dense(units=8, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.summary()
        model.compile(loss="mse", optimizer=Adam(lr=0.001))

        return model

    def act(self, state):
        if not self.is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        options = self.model.predict(state)
        return np.argmax(options[0])

    # def train_model(self, state, target_f):
    # 	history = self.model.fit(state, target_f, epochs=1, verbose=0)
    # 	return history.history['val_loss']

    def expReplay(self, batch_size):
        mini_batch = []
        l = len(self.memory)
        for i in range(l - batch_size + 1, l):
            mini_batch.append(self.memory[i])

        for state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(
                    self.model.predict(next_state)[0])

            target_f = self.model.predict(state)
            target_f[0][action] = target
            # history = self.model.fit(state, target_f, epochs=1, verbose=2, callbacks=[self.tensorboard_callback])
            history = self.model.fit(state, target_f, epochs=1, verbose=0)
            self.loss_val = history.history['loss'][0]
            self.loss_hist.append(self.loss_val)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# prints formatted price
def formatPrice(n):
    return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))


# returns the vector containing stock data from a fixed file
def getStockDataVec(key):
    vec = []
    # lines = open("../input/GSPC.csv", "r").read().splitlines() # this is where I have to change
    lines = open("price-volume-data-for-all-us-stocks-etfs/Stocks/googl.us.txt",
                 "r").read().splitlines()
    df = pd.read_csv(
        "price-volume-data-for-all-us-stocks-etfs/Stocks/googl.us.txt")
    # df = df.set_index('Date')

    for line in lines[1:]:
        vec.append(float(line.split(",")[4]))

    return df, vec


# returns the sigmoid
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# returns an an n-day state representation ending at time t
def getState(data, t, n):
    d = t - n + 1
    # TODO what is the vlaue in each entry represent?
    block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[
                                                          0:t + 1]  # pad with t0
    res = []
    for i in range(n - 1):
        res.append(sigmoid(block[i + 1] - block[i]))

    return np.array([res])


# def run_test(data,window_size, model_name):
# 	stock_name, window_size, episode_count = '^GSPC', 12, 5
# 	agent = Agent(window_size,is_eval=True, model_name = model_name)
#
# 	l = len(data) -1
# 	for t in range(l):
# 		state = getState(data, t , window_size + 1 )
# 		agent.model.predict(state)

def run_model(data, episode_count, window_size, batch_size, model_name=None,
              is_eval=None, verbose=True):
    reward_hist = []
    total_reward = 0
    assert isinstance(is_eval, bool), ''
    if is_eval:
        agent = Agent(window_size, is_eval=is_eval, model_name=model_name)
    else:
        agent = Agent(window_size)

    l = len(data) - 1
    for e in range(episode_count + 1):
        print("Episode " + str(e) + "/" + str(episode_count))
        state = getState(data, 0, window_size + 1)

        total_profit = 0
        agent.inventory = []

        for t in range(l):
            action = agent.act(state)

            # sit
            next_state = getState(data, t + 1, window_size + 1)
            reward = 0

            if action == 1:  # buy
                agent.inventory.append(data[t])
            # print("Buy: " + formatPrice(data[t]))

            elif action == 2 and len(agent.inventory) > 0:  # sell
                bought_price = agent.inventory.pop(0)
                reward = max(data[t] - bought_price, 0)
                total_profit += data[t] - bought_price
            # print("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))

            done = True if t == l - 1 else False
            agent.memory.append((state, action, reward, next_state, done))

            total_reward += reward

            reward_hist.append(total_reward)
            state = next_state

            if done:
                print("--------------------------------")
                print("Total Profit: " + formatPrice(total_profit))
                print("--------------------------------")

            # TODO why does it have to be more than batch size?
            if len(agent.memory) > batch_size and not is_eval:
                agent.expReplay(batch_size)

            if verbose:
                if e % 50 == 0:
                    print(f'loss={agent.loss_val}, reward={total_reward}')

        # if e % 10 == 0:
        if not os.path.exists('models'):
            os.makedirs('models')

        if not is_eval:
            agent.model.save("models/model_ep" + str(e))

        if is_eval:
            return


def plot_train_test(train, test, date_split):
    f, axs = plt.subplots(1,2)
    axs[0].plot(train['Close'].to_numpy().tolist())
    axs[1].plot(test['Close'].to_numpy().tolist())
    plt.show()


def plot_loss_reward(total_losses, total_rewards):
    figure = tools.make_subplots(rows=1, cols=2,
                                 subplot_titles=('loss', 'reward'),
                                 print_grid=False)
    figure.append_trace(
        go.Scatter(y=total_losses, mode='lines', line=dict(color='skyblue')), 1,
        1)
    figure.append_trace(
        go.Scatter(y=total_rewards, mode='lines', line=dict(color='orange')), 1,
        2)
    figure['layout']['xaxis1'].update(title='epoch')
    figure['layout']['xaxis2'].update(title='epoch')
    figure['layout'].update(height=400, width=900, showlegend=False)
    iplot(figure)


def get_technical_indicators(dataset):

    # Create 7 and 21 days Moving Average
    dataset['ma7'] = dataset['Close'].rolling(window=7).mean()
    dataset['ma21'] = dataset['Close'].rolling(window=21).mean()

    # Create MACD
    # dataset['26ema'] = dataset['Close'].rolling(window=27).mean()
    # dataset['12ema'] = dataset['Close'].rolling(window=12).mean()
    dataset['ema12'] = dataset['Close'].ewm(span=12).mean()
    dataset['ema27'] = dataset['Close'].ewm(span=27).mean()

    dataset['MACD'] = (dataset['ema12'] - dataset['ema27'])
    # Create Bollinger Bands
    dataset['20sd'] = dataset['Close'].rolling(20).std()
    dataset['upper_band'] = dataset['ma21'] + (dataset['20sd'] * 2)
    dataset['lower_band'] = dataset['ma21'] - (dataset['20sd'] * 2)

    # Create Exponential moving average
    dataset['ema'] = dataset['Close'].ewm(com=0.5).mean()

    # Create Momentum
    dataset['momentum'] = dataset['Close'] - 1

    plt.plot(dataset['Close'])
    plt.plot(dataset['ma7'])
    plt.plot(dataset['ema12'])
    # plt.plot(dataset['MACD'])
    plt.plot(dataset['20sd'])
    plt.plot(dataset['upper_band'])
    plt.plot(dataset['lower_band'])
    plt.plot(dataset['ema'])
    plt.plot(dataset['momentum'])
    plt.show()

    return dataset

if __name__ == '__main__':
    stock_name, window_size, episode_count = '^GSPC', 12, 5

    is_eval = False
    df, data = getStockDataVec(stock_name)

    # y = datetime.strptime('2010-09-07', '%Y-%m-%d')
    data_split = int(len(data) / 2)
    # data = df.index[data_split]
    train_set, test_set = data[:data_split], data[data_split:]
    batch_size = 32

    df = df.set_index('Date')
    train_set_df, test_set_df = df.iloc[:data_split], df.iloc[data_split:]

    get_technical_indicators(df)

    plot_train_test(train_set_df, test_set_df, list(df.index)[data_split])
    run_model(train_set, episode_count=episode_count, window_size=window_size,
              batch_size=batch_size, is_eval=is_eval)

# for i in range(6):
# 	model_name = f'model_ep{i}'
# 	run_model(test_set,episode_count=episode_count,window_size=window_size, batch_size=batch_size, model_name=model_name,is_eval=not is_eval, verbose=False)
