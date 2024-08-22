# Databricks notebook source
# MAGIC %md
# MAGIC talk about power and effect size

# COMMAND ----------

# MAGIC %md
# MAGIC explore vs. exploit

# COMMAND ----------

import numpy as np
from scipy.stats import norm, t

import matplotlib.pyplot as plt

# COMMAND ----------

np.random.seed(123)

# COMMAND ----------

N=1000
mu=5
sigma=2
X = np.random.randn(N)*sigma + mu

# COMMAND ----------

# MAGIC %md
# MAGIC Z confidence interval

# COMMAND ----------

mu_hat = np.mean(X)
sigma_hat = np.std(X, ddof=1)
z_left = norm.ppf(.025)
z_right = norm.ppf(.975)
lower = mu_hat + z_left * sigma_hat / np.sqrt(N)
upper = mu_hat + z_right * sigma_hat / np.sqrt(N)

# COMMAND ----------

print(lower, mu_hat, upper)

# COMMAND ----------

# MAGIC %md
# MAGIC T confidence Interval

# COMMAND ----------

mu_hat = np.mean(X)
sigma_hat = np.std(X, ddof=1)
t_left = t.ppf(.025, df=N-1)
t_right = t.ppf(.975, df=N-1)
lower = mu_hat + z_left * sigma_hat / np.sqrt(N)
upper = mu_hat + z_right * sigma_hat / np.sqrt(N)

# COMMAND ----------

print(lower, mu_hat, upper)

# COMMAND ----------

# MAGIC %md
# MAGIC experiment

# COMMAND ----------

def experiment():
    X = np.random.randn(N)*sigma + mu
    mu_hat = np.mean(X)
    sigma_hat = np.std(X, ddof=1)
    t_left = t.ppf(.025, df=N-1)
    t_right = t.ppf(.975, df=N-1)
    lower = mu_hat + z_left * sigma_hat / np.sqrt(N)
    upper = mu_hat + z_right * sigma_hat / np.sqrt(N)
    return mu > lower and mu < upper

# COMMAND ----------

def multi_experiment(M):
    results = [experiment() for i in range(M)]
    return np.mean(results)

# COMMAND ----------

multi_experiment(10000)

# COMMAND ----------

# MAGIC %md
# MAGIC # Bandits

# COMMAND ----------

# MAGIC %md
# MAGIC ## Epsilon Greedy

# COMMAND ----------

num_trials = 10000
eps = 0.1
bandit_probs = [0.2, 0.5, 0.8]

# COMMAND ----------

class Bandit:
    def __init__(self, p):
        #p is the win rate
        self.p = p
        self.p_estimate = 0
        self.N = 0 #samples collected so far

    def pull(self):
        #draw a 1 with probability p
        return np.random.random() < self.p
    
    def update(self, x):
        #x is 0 or 1
        self.N += 1
        self.p_estimate = ((self.N - 1)*self.p_estimate + x) / self.N

# COMMAND ----------

def experiment():
    bandits = [Bandit(p) for p in bandit_probs]

    rewards = np.zeros(num_trials)
    num_times_explored = 0
    num_times_exploited = 0
    num_optimal = 0
    optimal_j = np.argmax([b.p for b in bandits]) #you won't know this in real life
    print("optimal j:", optimal_j)

    for i in range(num_trials):

        if np.random.random() < eps:
            num_times_explored += 1
            j = np.random.randint(len(bandits))
        else:
            num_times_exploited += 1
            j = np.argmax([b.p_estimate for b in bandits])

        if j == optimal_j:
            num_optimal += 1

        #get a reward
        x = bandits[j].pull()

        #udpate reward array
        rewards[i] = x

        #update the bandit
        bandits[j].update(x)


    #coll info
    for b in bandits:
        print('mean estimate:', b.p_estimate)

    print('total reward earned:', rewards.sum())
    print('overall win rate:', rewards.sum() / num_trials)
    print('times explored:', num_times_explored)
    print('times exploited:', num_times_exploited)
    print('times used optimal:', num_optimal)

    cumulative_rewards = np.cumsum(rewards)
    win_rates = cumulative_rewards / (np.arange(num_trials) + 1)
    plt.plot(win_rates)
    plt.plot(np.ones(num_trials)*np.max(bandit_probs))
    plt.show()

# COMMAND ----------

experiment()

# COMMAND ----------


