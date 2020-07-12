import pandas as pd
import sys
import os



buy = {}

loss = []
quantity = {}
file = sys.argv[1]
fil = file[:-4]
df = pd.read_csv(file)
open = df["Open"]
date = df['Date']
close1 = df["Close"]
oname = fil+'_output.csv'

l = len(df)
print(l)

symbol = file

def Main():
    i=0
    quantity_1 = 1  # any value between 1-4 : 1 =100%, 2=50%, 3 = 33%, 4 = 25%, 5 = 20% and so on...
    max_amount = 1000  # Maximum authorized amount
    loss_limit = -50  # Maximum loss limit to terminate the trading in dollar
    buy_percent = 0.009  # percent at which it should buy, currently 0.1% = 0.1/100 = 0.001
    sell_percent = 0.007  # percent at which it should sell, currently 0.1%
    loss_percent = -0.01  # stop loss if price falls, currently -0.3%
    transaction = 150  # number of maximum transactions
    buy_range = 0.011  # allowed buy upto, currently 0.4%
    total_profit = 0
    sleep_time = 45  # according to candle interval 15 for 5 MINUTE, 30 for 30 MINUTE, 45 for 1 HOUR
    spent_amount = 0
    count = 0
    buy_open = []  # to avoid buying at same candle
    while i<l:
        open1 = open[i]
        close = close1[i]
        date1 = date[i]
        i+=1
        try:
            if open1 not in buy_open:
                # print(buy_percent * open1, buy_range * open1)
                if (close >= (1 + buy_percent) * open1) and (symbol not in buy.keys()) and close < (1 + buy_range) * open1:
                    if spent_amount <= max_amount:
                        count += 1
                        quantity[symbol] = (max_amount / (quantity_1 * close))
                        quantity1 = quantity[symbol]
                        buy_open.append(open1)


                        spent_amount += close * quantity1
                        buy[symbol] = close
                        print('Bought ' + symbol + ' at ' + str(close))

                        df1 = pd.DataFrame({'Datetime': [date1], 'Buy/Sell': ['Buy'],
                                            'Quantity': [quantity1], 'Price': [close], 'Profit/loss': [0]})
                        if not os.path.isfile('Binance.csv'):
                            df1.to_csv(oname, index=False)
                        else:
                            df1.to_csv(oname, index=False, mode='a', header=False)

            if symbol in buy:
                if (close >= buy[symbol] * (1 + sell_percent)) or (close <= (1 + loss_percent) * buy[symbol]):

                    profit = close - buy[symbol]
                    max_amount += profit
                    quantity1 = quantity[symbol]
                    spent_amount -= quantity1 * buy[symbol]
                    total_profit = profit * quantity1
                    print("SELL " + symbol + " at " + str(close))
                    print("Profit made " + str(total_profit))

                    df2 = pd.DataFrame({'Datetime': [date1], 'Buy/Sell': ['Sell'],
                                        'Quantity': [quantity_1], 'Price': [close], 'Profit/loss': [total_profit]})
                    df2.to_csv(oname, index=False, mode='a', header=False)

                    loss.append(total_profit)
                    buy.pop(symbol)  # Removing the sold symbol

            else:
                print("Hold " + symbol + " at " + str(close))
                print("Profit made " + str(total_profit))

                df3 = pd.DataFrame({'Datetime': [date1], 'Buy/Sell': ['Hold'],
                                    'Quantity': [quantity_1], 'Price': [close], 'Profit/loss': [total_profit]})
                df3.to_csv(oname, index=False, mode='a', header=False)




            if (loss_limit > sum(loss)) or (count > int(transaction)):
                print("Quitting....")
                raise SystemExit


        except KeyboardInterrupt:
            print("Total Profit " + str(sum(loss)))
            fi = pd.read_csv(oname)
            print(fi.tail(2))
            break

        except SystemExit:
            fi = pd.read_csv(oname)
            print(fi.tail(2))
            print("Exit")
            print("Total Profit " + str(sum(loss)))
            break

    fi = pd.read_csv(oname)
    print(fi.tail(2))

if __name__ == "__main__":
    Main()