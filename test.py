import matplotlib.pyplot as plt
import numpy as np
import math

# パラメータ
eps_end = 0.98
eps_start = 0.1
max_len = 1000000
eps_decay = max_len * 15 / 100

# ステップ数の範囲を生成
steps_done_values = np.arange(0, max_len, 100)

# εの値を計算
eps_values = eps_end + (eps_start - eps_end) * np.exp(-1. * steps_done_values / eps_decay)

print(eps_values)

# グラフの描画
plt.plot(steps_done_values, eps_values, label='ε Threshold')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.legend()
plt.title('ε Threshold Decay Over Steps')
plt.xlabel('Steps Done')
plt.ylabel('ε Threshold')
plt.show()