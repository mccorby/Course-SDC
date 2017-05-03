import matplotlib.pyplot as plt
import pandas as pd

train = pd.read_csv('./data/driving_log.csv')

print("Train data shape:", train.shape)

print(train.keys())

plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)

print(train.steering.describe())

# Let's check for skewness
print("Skew is:", train.steering.skew())
plt.hist(train.steering, color='blue')
plt.show()
