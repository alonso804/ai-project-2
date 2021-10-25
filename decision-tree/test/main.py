import matplotlib.pyplot as plt
import seaborn as sns

errors = [
    2.6,
    5,
    4.4,
    3.2,
    3.8,
    4.2,
    4.2,
    3.6,
    6.6,
    5.2
]

sns.distplot(errors)
plt.savefig(f'./test.png')
