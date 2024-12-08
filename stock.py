import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Stock:
    def __init__(self, file_path='C:/Users/adird/Downloads/stock prices sp predict/sony.csv', window_size=20):
        df = pd.read_csv(file_path)
        df['Close/Last'] = df['Close/Last'].str.replace('$', '', regex=False).astype(float)
        self.close_last_first_500 = df['Close/Last'].iloc[:100 + window_size].to_numpy()
        self.window_size = window_size

        self.min_value = np.min(self.close_last_first_500)
        self.max_value = np.max(self.close_last_first_500)
        self.close_last_first_500 = (self.close_last_first_500 - self.min_value) / (self.max_value - self.min_value)

    def get_price(self, index):
        if 0 <= index < len(self.close_last_first_500):
            return self.close_last_first_500[index + self.window_size]
        else:
            return 0
            print ('index out of range')

    def get_length(self):
        return len(self.close_last_first_500)

    def get_stock_price_until_that_index(self, index):
        return self.close_last_first_500[index:self.window_size + index]


stock = Stock('C:/Users/adird/Downloads/stock prices sp predict/sony.csv', 20)
print(stock.get_price(1))
