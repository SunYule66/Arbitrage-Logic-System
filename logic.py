import pandas as pd
import csv
import json
import numpy as np
import datetime
from glob import glob

class ArbitrageSystem:
    def __init__(self, X, Y, A, B, N, M, P, Q):
        self.X = X #套利差价触发阈值（百分比）
        self.Y = Y #资金费率差触发阈值（百分比）    
        self.A = A #可忽视的差价百分比阈值
        self.B = B #可忽视的资金费率差价百分比阈值
        self.N = N #历史记录数据的小时数量（小时）
        self.M = M #资金费率不利持续时间（小时）
        self.P = P #价差盈利百分比
        self.Q = Q #价差亏损百分比
        self.positions = []

    def direction(self, price_spread, funding_spread):
        # 判断方向是否一致
        return (price_spread >= 0 and funding_spread >= 0) or (price_spread < 0 and funding_spread < 0)

    def check_open(self, df, idx):
        # 获取当前和历史N小时数据
        current = df.iloc[idx]
        current_ts = current.name
        # 以时间窗口（小时）筛选历史
        window_start = current_ts - self.N * 3600
        history = df.loc[(df.index >= window_start) & (df.index < current_ts)]
        # 使用相对价差（百分比）判断
        price_spread = (current['price_a'] - current['price_b']) / current['price_b']
        funding_spread = current['funding_a'] - current['funding_b']
        avg_price_spread = ((history['price_a'] - history['price_b']) / history['price_b']).mean() if not history.empty else 0
        same_direction = self.direction(price_spread, funding_spread)

        # 差价套利开仓条件
        if price_spread >= self.X and price_spread > avg_price_spread:
            if same_direction and funding_spread < self.Y:
                return self.open_position('差价套利', '条件a', current, price_spread, funding_spread, avg_price_spread, '相同')
            elif not same_direction and funding_spread < self.B:
                return self.open_position('差价套利', '条件b', current, price_spread, funding_spread, avg_price_spread, '不同')

        # 资金费率套利开仓条件
        if funding_spread >= self.Y and all(history['funding_a'] - history['funding_b'] >= self.Y):
            if same_direction and price_spread < self.X:
                return self.open_position('资金费率套利', '条件a', current, price_spread, funding_spread, avg_price_spread, '相同')
            elif not same_direction and price_spread < self.A:
                return self.open_position('资金费率套利', '条件b', current, price_spread, funding_spread, avg_price_spread, '不同')

        # 组合套利开仓条件
        if same_direction and price_spread >= self.X and price_spread > avg_price_spread and funding_spread >= self.Y and all(history['funding_a'] - history['funding_b'] >= self.Y):
            return self.open_position('组合套利', '条件', current, price_spread, funding_spread, avg_price_spread, '相同')
        return None

    def open_position(self, mode, cond, current, price_spread, funding_spread, avg_price_spread, direction):
        # 记录开仓信息并返回，便于后续平仓跟踪
        position = {
            '触发模式': mode,
            '触发条件': cond,
            '开仓差价': price_spread,
            '开仓资金费率差': funding_spread,
            '开仓历史平均值': avg_price_spread,
            '开仓方向': direction,
            '开仓价格a': current['price_a'],
            '开仓价格b': current['price_b'],
            '开仓资金费率a': current['funding_a'],
            '开仓资金费率b': current['funding_b'],
            '开仓时间戳': current.name,
            '平仓': False,
            '平仓信息': None
        }
        self.positions.append(position)
        return position

    def check_close(self, df, idx):
        # 遍历所有未平仓的持仓，判断是否满足平仓条件
        closed_any = False
        for pos in self.positions:
            if pos['平仓']:
                continue
            current = df.iloc[idx]
            price_spread = (current['price_a'] - current['price_b']) / current['price_b']
            funding_spread = current['funding_a'] - current['funding_b']
            # 差价套利
            if pos['触发模式'] == '差价套利':
                # 条件a（相同方向差价套利）
                if pos['触发条件'] == '条件a':
                    # 平仓条件a: 价格回归盈利
                    if price_spread >= self.P:
                        self.close_position(pos, current, '价格回归盈利')
                        closed_any = True
                        continue
                    # 平仓条件b: 资金费率反转止损
                    # 价差无利可图
                    if price_spread < self.X:
                        # 资金费率方向反转（赚取变成支付）
                        open_funding_spread = pos['开仓资金费率差']
                        funding_direction_reversed = (open_funding_spread > 0 and funding_spread < 0) or (open_funding_spread < 0 and funding_spread > 0)
                        # 资金费率数值 > A 且持续时间 > M
                        if funding_direction_reversed and abs(funding_spread) > self.A:
                            # 检查持续时间
                            close_ts = current.name
                            open_ts = pos['开仓时间戳']
                            hours = (close_ts - open_ts) / 3600
                            if hours >= self.M:
                                self.close_position(pos, current, '资金费率反转止损')
                                closed_any = True
                                continue
                        # 平仓条件c: 价差亏损止损
                        if price_spread <= -self.Q:
                            self.close_position(pos, current, '价差亏损止损')
                            closed_any = True
                            continue
                # 条件b（不同方向差价套利）
                elif pos['触发条件'] == '条件b':
                    # 平仓条件a: 价格回归盈利
                    if price_spread >= self.P:
                        self.close_position(pos, current, '价格回归盈利')
                        closed_any = True
                        continue
                    # 平仓条件b: 资金费率扩大止损
                    if price_spread < self.X:
                        # 需要支付资金费率
                        if funding_spread < 0:
                            # 资金费率阈值从 < A 变成 > B
                            open_funding_spread = pos['开仓资金费率差']
                            if abs(open_funding_spread) < self.A and abs(funding_spread) > self.B:
                                # 检查持续时间
                                close_ts = current.name
                                open_ts = pos['开仓时间戳']
                                hours = (close_ts - open_ts) / 3600
                                if hours >= self.M:
                                    self.close_position(pos, current, '资金费率扩大止损')
                                    closed_any = True
                                    continue
                        # 平仓条件c: 价差亏损止损
                        if price_spread <= -self.Q:
                            self.close_position(pos, current, '价差亏损止损')
                            closed_any = True
                            continue
            # 资金费率套利
            elif pos['触发模式'] == '资金费率套利':
                # 平仓条件a: 资金费率收敛或反转
                open_funding_spread = pos['开仓资金费率差']
                funding_direction_reversed = (open_funding_spread > 0 and funding_spread < 0) or (open_funding_spread < 0 and funding_spread > 0)
                if abs(funding_spread) < self.B or funding_direction_reversed:
                    self.close_position(pos, current, '资金费率收敛或反转')
                    closed_any = True
                    continue
                # 平仓条件b: 价差盈利平仓
                if funding_spread > 0 and price_spread >= self.P:
                    self.close_position(pos, current, '价差盈利平仓')
                    closed_any = True
                    continue
                # 平仓条件c: 价差亏损止损
                if price_spread <= -self.Q:
                    self.close_position(pos, current, '价差亏损止损')
                    closed_any = True
                    continue
            # 组合套利
            elif pos['触发模式'] == '组合套利':
                open_funding_spread = pos['开仓资金费率差']
                funding_direction_reversed = (open_funding_spread > 0 and funding_spread < 0) or (open_funding_spread < 0 and funding_spread > 0)
                # 平仓条件a: 资金费率收敛/反转或价差盈利
                if abs(funding_spread) <= self.B or funding_direction_reversed or price_spread >= self.P:
                    self.close_position(pos, current, '资金费率收敛/反转或价差盈利')
                    closed_any = True
                    continue
                # 平仓条件b: 价差亏损止损
                if price_spread <= -self.Q:
                    self.close_position(pos, current, '价差亏损止损')
                    closed_any = True
                    continue
        return closed_any

    def has_open_positions(self):
        return any(not p['平仓'] for p in self.positions)

    def close_position(self, pos, current, reason):
        pos['平仓'] = True
        pos['平仓信息'] = {
            '平仓时间戳': current.name,
            '平仓价格a': current['price_a'],
            '平仓价格b': current['price_b'],
            '平仓资金费率a': current['funding_a'],
            '平仓资金费率b': current['funding_b'],
            '平仓原因': reason
        }

    def run(self, df):
        for idx in range(len(df)):
            self.check_open

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

# 实例化套利系统，参数可根据实际需求调整
system = ArbitrageSystem(
    X=0.000068,  # 差价触发阈值
    Y=0.000038, # 资金费率差触发阈值
    A=0.000235, # 可忽略差价阈值
    B=0.00014,# 可忽略资金费率差阈值
    N=5,    # 历史小时数
    M=5,     # 资金费率不利持续时间
    P=0.0049,  # 盈利平仓阈值
    Q=0.000062   # 亏损止损阈值
)

# 预处理数据，构造套利逻辑所需字段
df['price_a'] = df['okx_price']
df['price_b'] = df['binance_price']
df['funding_a'] = df['okx_funding']
df['funding_b'] = df['binance_funding']

# 对齐时间序列并清洗缺失值，避免 NaN 造成开平仓判断异常
df.sort_index(inplace=True)

# 先记录原始可用数据的最后时间戳，再做前向填充，避免前值填充把缺失尾段误判为可用
last_valids = []
for col in ['price_a', 'price_b', 'funding_a', 'funding_b']:
    lv = df[col].last_valid_index()
    if lv is not None:
        last_valids.append(lv)
cutoff = min(last_valids) if last_valids else None

# 再填充，随后按原始可用截止截断
df[['price_a', 'price_b', 'funding_a', 'funding_b']] = df[['price_a', 'price_b', 'funding_a', 'funding_b']].ffill()
if cutoff is not None:
    df = df.loc[:cutoff]
# 截断后再 dropna，避免尾部缺口导致的虚假平仓
df = df.dropna(subset=['price_a', 'price_b', 'funding_a', 'funding_b'])


# 模拟实时交易：持仓未平仓前不再开新仓
for idx in range(len(df)):
    closed_now = system.check_close(df, idx)
    if system.has_open_positions():
        continue
    if closed_now:
        # 本周期刚平仓，为避免同一周期即刻再开仓，跳过本周期
        continue
    system.check_open(df, idx)

# 输出开仓和平仓信息到文件
def convert(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {k: convert(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert(i) for i in obj]
    return obj

with open('套利系统/arbitrage_positions.json', 'w', encoding='utf-8') as f:
    json.dump(convert(system.positions), f, ensure_ascii=False, indent=2)

# === 计算总收益率（价格收益率 + 资金费率收益，不含手续费/滑点）===
def calc_price_return(position):
    if not position.get('平仓'):
        return 0.0
    open_spread = position['开仓差价']
    open_a = position['开仓价格a']
    open_b = position['开仓价格b']
    close_a = position['平仓信息']['平仓价格a']
    close_b = position['平仓信息']['平仓价格b']
    # 假设 spread>0 时开仓为 空a 多b；spread<0 时为 多a 空b
    if open_spread >= 0:
        # 两腿各占50%资金：空a收益率 + 多b收益率
        ret = 0.5 * ((open_a - close_a) / open_a) + 0.5 * ((close_b - open_b) / open_b)
    else:
        # 多a收益率 + 空b收益率
        ret = 0.5 * ((close_a - open_a) / open_a) + 0.5 * ((open_b - close_b) / open_b)
    return float(ret)

def calc_funding_return(position, funding_df):
    if not position.get('平仓'):
        return 0.0
    open_ts = position['开仓时间戳']
    close_ts = position['平仓信息']['平仓时间戳']
    # 提取持仓期间的资金费率序列（含开仓、平仓时刻）
    seg = funding_df.loc[(funding_df.index >= open_ts) & (funding_df.index <= close_ts)]
    if seg.empty:
        return 0.0
    open_spread = position['开仓差价']
    # spread>0 -> 空a 多b；spread<0 -> 多a 空b
    sign_a = -1 if open_spread >= 0 else 1
    sign_b = -sign_a
    # 资金费率加权：按区间时长（小时）累计；两腿各占50%资金
    seg = seg.sort_index()
    ts_list = list(seg.index) + [close_ts]
    total = 0.0
    for i, ts in enumerate(seg.index):
        next_ts = ts_list[i+1]
        hours = max(0, (next_ts - ts) / 3600)
        rate_eff = 0.5 * seg.loc[ts, 'funding_a'] * sign_a + 0.5 * seg.loc[ts, 'funding_b'] * sign_b
        total += rate_eff * hours
    return float(total)

# 计算并打印总收益率（复利）
funding_df = df[['funding_a', 'funding_b']].copy().ffill().dropna()
cum_return = 1.0
for pos in system.positions:
    price_ret = calc_price_return(pos)
    funding_ret = calc_funding_return(pos, funding_df)
    total_ret = price_ret + funding_ret
    cum_return *= (1 + total_ret)

final_return_pct = (cum_return - 1) * 100
print(f"最终收益率: {final_return_pct:.4f}%")

