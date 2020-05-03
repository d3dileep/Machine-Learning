import shrimpy
import time
import argparse

api_key = "iZHsmlsCReb9S6zVO05Vxy8ONQYK8J3CfshgNiRh3HlRShPULMj8EYBClftHBqi1"
api_secret = "4IHk54oeSmmoXGQqWNgi24SJ1uHaTSEBfN48nOhYex8ATFFOj2WoWZQfDFD0pzu1"

client = shrimpy.ShrimpyApiClient(api_key, api_secret)


def Scanner(args):
    count = 0
    time_list = []
    symbols = ['BNB', 'XRP', 'LTC','BTC','ETH','BCH']  # User Request symbol
    while True:
        close = []
        ticker = client.get_ticker('binance')
        count += 1
        for req in symbols:
            for item in ticker:

                if item['symbol'] == req:
                    close.append(float(item["priceUsd"]))

        time_list.append(close)

        if len(time_list) == 21:
            time_list.pop(0)
      
        if len(time_list) <=5:
            continue
        percent = [((round(x/y,4) - 1) * 100) for x, y in zip(time_list[-1], time_list[0])]

        for check in percent:

            if check >= args.change:
                index = percent.index(check)
                print(symbols[index])
        # when candle frame of 1-minute chosen.
        if args.interval == '1M':
            time.sleep(3)
        # when candle size of 3-minute chosen
        if args.interval == '3M':
            time.sleep(9)
        #when candle size of 5-minute chosen
        if args.interval == '5M':
            time.sleep(15)
        #when candle size of 15-minute chosen
        if args.interval == '15M':
            time.sleep(45)
        # candle size of 30-minute chosen
        if args.interval == '30M':
            time.sleep(90)
        # when candle size of 1-hour chosen
        if args.interval == '1H':
            time.sleep(180)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('interval', default="5M", help='choose from 1M, 3M, 5M, 15M, 30M, 1H')
    parser.add_argument('change', default=0.1, type=float, help='choose threshold percentage')
    args = parser.parse_args()
    Scanner(args)
