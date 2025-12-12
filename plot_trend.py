import csv
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

def extract_btc_usdt_swap_funding(data_folder, output_file):
    files = glob(f"{data_folder}/allswap-fundingrates-*.csv")
    result = []
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('instrument_name') == 'BTC-USDT-SWAP':
                    result.append([
                        row['instrument_name'],
                        row['funding_rate'],
                        row['funding_time']
                    ])
    # 写入目标文件
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['instrument_name', 'funding_rate', 'funding_time'])
        writer.writerows(result)
# 提取BTC-USDT-SWAP资金费率数据
extract_btc_usdt_swap_funding('套利系统/data', '套利系统/btc_usdt_swap_funding.csv')


# 读取OKX价格（BTC-USDT-candlesticks），时间戳除1000
def load_okx_price(filename):
    prices = []
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts = int(row['open_time']) // 1000
                price = float(row['close'])
                prices.append({'timestamp': ts, 'price': price})
            except Exception:
                continue
    return prices

# 读取Binance价格（BTCUSDT-1m），时间戳除1000000
def load_binance_price(filename):
    prices = []
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            try:
                ts = int(row[0]) // 1000000
                price = float(row[4])
                prices.append({'timestamp': ts, 'price': price})
            except Exception:
                continue
    return prices

# 读取OKX资金费率，时间戳除1000
def load_okx_funding(filename):
    fundings = []
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts = int(row['funding_time']) // 1000
                rate = float(row['funding_rate'])
                fundings.append({'timestamp': ts, 'funding_rate': rate})
            except Exception:
                continue
    return fundings

# 读取Binance资金费率，时间戳为汉字时间
def load_binance_funding(filename):
    fundings = []
    with open(filename, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                contracts = row.get('Contracts', '').strip()
                rate_str = row.get('Funding Rate', '').strip()
                time_str = row.get('Time', '').strip()
                if 'BTCUSDT' in contracts:
                    ts = int(datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S").timestamp())
                    rate = float(rate_str.replace('%','')) / 100
                    fundings.append({'timestamp': ts, 'funding_rate': rate})
            except Exception:
                continue
    return fundings

glob_okx_price = glob("套利系统/data/BTC-USDT-candlesticks-2025-*.csv")
glob_binance_price = glob("套利系统/data/BTCUSDT-1m-2025-*.csv")
file_okx_funding = "套利系统/btc_usdt_swap_funding.csv"
file_binance_funding = "套利系统/data/Funding Rate History_BTCUSDT Perpetual_2025-12-09.csv"

# 批量读取所有数据文件
okx_prices, binance_prices = [], []
for f in glob_okx_price:
    okx_prices += load_okx_price(f)
for f in glob_binance_price:
    binance_prices += load_binance_price(f)
okx_fundings = load_okx_funding(file_okx_funding)
binance_fundings = load_binance_funding(file_binance_funding)

# 构建DataFrame
df_binance_price = pd.DataFrame(binance_prices)
df_okx_price = pd.DataFrame(okx_prices)
df_binance_funding = pd.DataFrame(binance_fundings)
df_okx_funding = pd.DataFrame(okx_fundings)

# 统一用timestamp为索引
df_binance_price.set_index('timestamp', inplace=True)
df_okx_price.set_index('timestamp', inplace=True)
df_binance_funding.set_index('timestamp', inplace=True)
df_okx_funding.set_index('timestamp', inplace=True)

# 合并所有数据，按时间戳对齐
df = pd.DataFrame(index=sorted(set(df_binance_price.index) | set(df_okx_price.index) | set(df_binance_funding.index) | set(df_okx_funding.index)))
df['binance_price'] = df_binance_price['price']
df['okx_price'] = df_okx_price['price']
df['binance_funding'] = df_binance_funding['funding_rate']
df['okx_funding'] = df_okx_funding['funding_rate']

# 时间戳转datetime
df['datetime'] = pd.to_datetime(df.index, unit='s')

# 筛选时间段
mask = (df['datetime'] >= pd.Timestamp('2025-11-24')) & (df['datetime'] <= pd.Timestamp('2025-12-07 16:00:00'))
df = df.loc[mask]

# 绘图
plt.figure(figsize=(14, 6))
plt.subplot(2, 1, 1)
plt.plot(df['datetime'], df['binance_price'], label='Binance Price')
plt.plot(df['datetime'], df['okx_price'], label='OKX Price')
plt.legend()
plt.title('BTC-USDT Price Trend')
plt.ylabel('Price')

plt.subplot(2, 1, 2)
plt.step(df['datetime'], df['binance_funding'], where='post', label='Binance Funding Rate')
plt.step(df['datetime'], df['okx_funding'], where='post', label='OKX Funding Rate')
plt.legend()
plt.title('BTC-USDT Funding Rate Trend')
plt.ylabel('Funding Rate')
plt.xlabel('Time')

plt.tight_layout()
plt.show()