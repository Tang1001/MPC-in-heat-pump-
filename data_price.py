import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取CSV文件
df = pd.read_csv('./data/slow_Price.csv',header=None)
print(df.head())

# df_5min = df.repeat(12).reset_index(drop=True)
data_5min = np.repeat(df.values.flatten(), 12)
df_5min = pd.DataFrame(data_5min.reshape(df.shape[0], -1))
print(df_5min[0:20])

day_1_prices = df.iloc[1].values
print(day_1_prices)


plt.figure(figsize=(15, 8))  # for example, (15, 8) makes the figure 15 units wide and 8 units high

# for i in range(10):
#     plt.plot(df.iloc[i], label=f"Day {i+1}",drawstyle='steps-post')
#
# plt.title('Electricity Prices for the First 10 Days')
# plt.xlabel('Hour')
# plt.ylabel('Price')
#
# hours = list(range(24))
# plt.xticks(hours, [str(hour) for hour in hours])
# plt.legend()
# plt.grid()
# plt.show()


# 绘制前10天的数据，只选择从7am到7pm的数据
for i in range(10):
    plt.plot(df.iloc[i, 7:20], label=f"Day {i+1}", drawstyle='steps-post')

plt.title('Electricity Prices from 7 AM to 7 PM for the First 10 Days')
plt.xlabel('Hour')
plt.ylabel('Price (Euro)')

# 设置x轴标签，从7am到7pm
hours = list(range(7, 20))
plt.xticks(hours, [str(hour) for hour in hours])

plt.legend()
plt.grid()
plt.show()