import gym
from gym import spaces, error, utils
from gym.spaces import Discrete, Box
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import finta
TA = finta.TA


def add_indicators(df):

    # Clean Data
    df['Date'] = pd.to_datetime(df['Date'])                         # Change date col to datetime obj
    df.dropna(inplace=True)                                         # drop nans
    df = df[['Date', 'Close', 'Volume', 'Open', 'High', 'Low']]     # Reorder cols
#     df.columns = df.columns.str.lower()
    df.reset_index(inplace=True)
    df = df.iloc[: , 1:]
    df = df.round()

    # Add indicators
    df['SMA7'] = TA.SMA(df, 7)
    df['ADX'] = TA.ADX(df)
    df['MI'] = TA.MI(df)
    df['ATR'] = TA.ATR(df)
    df['OBV'] = TA.OBV(df)
    df['MFI'] = TA.MFI(df)
    df['MFI'].replace(-np.inf, 0, inplace=True) # replace infinity
    df['RSI'] = TA.RSI(df)
    df['STOCH_RSI'] = TA.STOCHRSI(df)

    # Clean
    df = df.ffill()
    df.reset_index(inplace=True)
    df = df.iloc[:, 1:]

    return df

def normalize_columns(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, df, df_with_dates, lookback_window=30, INITIAL_BALANCE=10000):
        super(TradingEnv, self).__init__()    

        self.df_with_dates = df_with_dates
        self.df= df.iloc[: , 1:]
        self.initial_balance = INITIAL_BALANCE
        self.net_worth = INITIAL_BALANCE
        self.balance = INITIAL_BALANCE
        self.last_year_net_worth = INITIAL_BALANCE
        self.crypto_held = 0
        self.normalize_value = 1000
        self.fees = 0.08 # Interactive brokers fees = $5 or 0.08%
        self.lookback_window = lookback_window

        self.trades = []
        self.all_trades = []
        self.daily_returns = []
        self.all_net_worths = []
        self.yearly_returns = []
        
        self.starting_point = np.random.randint(self.lookback_window, len(self.df) - 1) # Leave 1 day at the start and end
        self.current_step = self.starting_point
        
        obs = self.df.iloc[self.current_step - self.lookback_window - 1:self.current_step-1]
        obs = obs.values.flatten()
        
        self.observation_space = Box(
            shape = (obs.shape[0],), # PASS FLATTEN SHAPE
            low = self.df.min().min(),
            high = self.df.max().max(),)
        
        self.action_space = spaces.MultiDiscrete([2,6]) # 3 actions and 5% max betting
        # self.action_space = Discrete(2)


    def print_bet_chart(self):
        self.bet_df = pd.DataFrame(self.trades)
        # COUNT THE BUY / SELL / HOLD TIMES
        # x_axis = [i for i in range(self.timestep)]
        # x_axis = [i for i in range(len(bet_df))]
        # plt.figure()
        # ax = plt.subplot()
        # plt.title('Net Worth')
        # plt.ylabel('Dollars')
        # plt.xlabel('Time')
        # ax.legend()
        # ax.plot(x_axis, bet_df['Net Worth'], label="Net Worth")
        # plt.savefig('./temp/Stock_Performance.png', bbox_inches='tight')
        # self.bet_df.to_csv('Bets.csv', index=False)
        # plt.clf() # clear figure

        # Plot two figures, one of the price
        # Save all runs and get the average at the end of each run
        # Plot them all with a for loop


    def reset(self, INITIAL_BALANCE=1000):

        self.initial_balance = INITIAL_BALANCE
        self.net_worth = INITIAL_BALANCE
        self.balance = INITIAL_BALANCE
        self.last_year_net_worth = INITIAL_BALANCE
        self.crypto_held = 0
        self.normalize_value = 1000

        if len(self.trades) > 0:
            self.all_trades.append(self.trades)
     
        self.trades = []
        
        self.starting_point = np.random.randint(self.lookback_window, len(self.df) - 1)
        self.current_step = self.starting_point
        obs = self.df.iloc[self.current_step - self.lookback_window - 1:self.current_step-1]
        obs = obs.values.flatten()
        
        return obs
        
    def step(self, actions):
        
        # Using MultiDiscrete
        action = actions[0]
        dollar_amount = math.floor(actions[1] / 100 * self.balance) # Percentage of stock to buy
        action_type = 'NA'

        # # # Betting with 5% Try only betting 4%
        # action_type = actions
        # dollar_amount = math.floor(0.05 * self.balance)

        
        self.crypto_bought = 0
        self.crypto_sold = 0
        

        current_price = self.df_with_dates.loc[self.current_step + 1, 'Open']
        date = self.current_step


        # Hold
        if action == 2:
            action_type = 'Hold'     

        if action == 0 and current_price > 0:

            # Buy Stock
            if (dollar_amount * (1 - self.fees)) >= current_price:
                self.crypto_bought = math.floor(dollar_amount  * (1 - self.fees)) // current_price
                self.balance -= self.crypto_bought * current_price
                self.crypto_held += self.crypto_bought
            action_type = 'Buy'

        elif action == 1 and self.crypto_held > 1 and current_price > 0:

            # Sell Stock
            # Sell all stock if dollar amount > stock held
            if (dollar_amount * (1 - self.fees)) >= current_price:
                self.crypto_sold = math.floor(dollar_amount * (1 - self.fees)) // current_price
                if self.crypto_sold > self.crypto_held:
                    self.crypto_sold = self.crypto_held
                    self.crypto_held = 0
                else:
                    self.crypto_held -= self.crypto_sold
                self.balance += self.crypto_sold * current_price
            action_type = 'Sell'
            

        self.prev_net_worth = self.net_worth
        self.net_worth = self.balance + self.crypto_held * current_price
        self.trades.append({'Date' : self.df_with_dates.loc[date, 'Date'], 'Balance': self.balance, \
                                        'Crypto Held': self.crypto_held, 'Net Worth': self.net_worth, \
                                        'Dollar Amount': dollar_amount, 'Action': action, 'Price': current_price})

        
        

        # Calculate average net worth for a week for reward
        profit = (self.net_worth - self.prev_net_worth)
        self.daily_returns.append(profit)
        
#         if profit < 10 and profit > 0:
#             reward = 0
#         else :
        reward = profit * 5
        
        if len(self.daily_returns) > 7:
            self.daily_returns.pop(0)
            
        reward = sum(self.daily_returns) / len(self.daily_returns)
        
#         if action == 2: # disincentivize holding (but so should reducing the reward for profit < 10)
#             reward -= 15
        
        years = (self.current_step - self.starting_point) / 251 # 251 Trading days in a year
        target = self.initial_balance * (1 + (years * 0.1))
        if years >= 1:
            if self.net_worth > self.initial_balance * (1 + (years * 0.1)):
                reward += 50 # Give it a big reward for passing 10% rewards each year

        # Calculate if done
        if self.net_worth <= self.initial_balance/2 or self.current_step >= len(self.df) - 2:
            done = True
        else:
            done = False

        # Get next observation
        obs = self.df.iloc[self.current_step - self.lookback_window - 1:self.current_step-1]
        obs = obs.values.flatten()
        info = {}

        if (self.current_step - self.starting_point) % 251 == 0: # Modulus to keep a yearly track record
            self.print_bet_chart()
            
            
            yearly_return = round(((self.net_worth - self.last_year_net_worth) / self.last_year_net_worth) * 100, 2) # Cal this above and put step function into many smaller ones
            self.all_net_worths.append(self.net_worth)
            self.yearly_returns.append(yearly_return)
            with open('./evaluations/yearly_returns.txt', 'w') as file:
                file.write(str(self.yearly_returns))
#             print("Year: ", int(years))
#             print('Date: ', str(self.df_with_dates.loc[date, 'Date'])[:10])
#             # print("Target: ", target) # calculate this better
#             print('Last Networth: ', self.last_year_net_worth)
#             print("Networth: ", self.net_worth)
#             # print("Avg Net Worths: ", round(sum(self.all_net_worths)/len(self.all_net_worths))) 
#             print('Yearly Return: ', yearly_return, '%')
#             print('Avg Yearly Returns: ', round(sum(self.yearly_returns)/len(self.yearly_returns)), '%')
#             print('')

            self.last_year_net_worth = self.net_worth

        
        self.current_step += 1

        return obs, reward, done, info


    def render(self):
        print("Current step: ", self.current_step)
        print("Balance: ", self.balance, "Holdings: ", self.crypto_held)
        print("Net worth: ", self.net_worth)

    
    def close(self):
        pass