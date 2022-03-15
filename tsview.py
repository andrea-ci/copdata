# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn theme.
sns.set_theme(style='darkgrid')

df = pd.read_csv('napoli-2017-2021.csv', sep = ':')
df.sort_values('ingestionDate', ascending=True, inplace=True)
df.index = pd.to_datetime(df['ingestionDate'])

df = df[['numShips']]
df.dropna(inplace = True)

# Average by month.
df_mean = df.groupby(pd.Grouper(freq='M')).mean()

lbls = [s.strftime('%Y-%m') for s in df_mean.index]

fig = plt.figure(figsize = (14, 8))
plt.plot(df_mean.index, df_mean.numShips)
plt.xticks(df_mean.index, lbls, rotation=70)
plt.xlabel('Date (year)')
plt.ylabel('Urban ratio [%]')
plt.grid(True)
plt.title('Napoli - ships')
fig.subplots_adjust(bottom = 0.2)
plt.savefig('ships_napoli.png', facecolor = 'white', dpi = 300)
