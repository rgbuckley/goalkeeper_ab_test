# Databricks notebook source
# MAGIC %md
# MAGIC #Introduction

# COMMAND ----------

# MAGIC %md
# MAGIC Penalty kick shootouts are a key part of tournament soccer. If the match is tied after regulation and/or overtime, a shootout determines the winner. Goalkeepers are crucial in these moments. A goalkeeper who is able to save multiple shots greatly increases his or her team's chance of winning. 

# COMMAND ----------

# MAGIC %md
# MAGIC Managers face a tough decision when going into a shootout. Most teams have two or three goalkeepers on the roster, and the starting goalkeeper might not be the best for the specialized skill of blocking shots in a shootout. Several managers will substitute the goalkeeper right before the shootout so the goalkeeper with the highest *save rate* will participate in the shootout. 

# COMMAND ----------

# MAGIC %md
# MAGIC But how does a manager determine which goalkeeper has the highest *save rate*? This project explores this question.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Notes and Caveats

# COMMAND ----------

# MAGIC %md
# MAGIC ### Assumptions

# COMMAND ----------

# MAGIC %md
# MAGIC For simplicity throughout the exercise, we will assume that each goalkeeper has an inherent *true save rate*. This does not change over time or with practice. Whether the goalkeeper saves a given shot is a random variable. The goalkeeper saves the shot with probability *true save rate* or the shot is scored.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Other Applications

# COMMAND ----------

# MAGIC %md
# MAGIC The rest of this notebook will explore how a coach might use data to determine which goalkeeper on her roster has the highest save rate. The applications for this kind of algorithm, however, are much broader than sports. Similar approaches could be used for:
# MAGIC - a marketing analyst determining which ad has the highest click-through rate
# MAGIC - a drug manufacturer determining which drug is most effective
# MAGIC - a manager determining which price is most likely to convince a customer to buy a product

# COMMAND ----------

# MAGIC %md
# MAGIC ### Attribution

# COMMAND ----------

# MAGIC %md
# MAGIC For much of what I learned I am indebted to [The Lazy Programmer](https://swirecc.udemy.com/user/lazy-programmer/) who has a fantastic course on Udemy on this topic. 
# MAGIC
# MAGIC If you are interested in learning more about A/B Testing, I highly recommend the [course](https://swirecc.udemy.com/course/bayesian-machine-learning-in-python-ab-testing/). 
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Approach 1: Classical A/B Testing

# COMMAND ----------

# MAGIC %md
# MAGIC Let's start with the *Classical* approach from statistics (this is also the *frequentist method*)  

# COMMAND ----------

# MAGIC %md
# MAGIC The coach wants to know which goalkeeper (Buffon or Casillas) has a higher *save rate*. The coach plans an experiment.
# MAGIC
# MAGIC > During training sessions, each goalkeeper will face a number of penalty kick shots from several different shooters. The coach will record whether each shot is saved and use this data to determine which goalkeeper has the higher save rate or if they are the same.

# COMMAND ----------

# MAGIC %md
# MAGIC The coach knows that the average save rate among all goalkeepers is about [25%](https://theanalyst.com/2024/05/premier-league-penalties-like-free-goal). The coach reasons that it is important to be able to detect a difference of 5% or greater between the true save rates of each goalkeeper.
# MAGIC
# MAGIC To ensure the experiment picks up the difference (if it exists), *how many shots should each goalkeeper face?*

# COMMAND ----------

# MAGIC %md
# MAGIC The coach is not sure how to answer that question. She googles around online and lands on this [calculator](https://www2.ccrb.cuhk.edu.hk/stat/proportion/Casagrande.htm) from the center for clinical research and biostats. 

# COMMAND ----------

gk1 = 0.25
gk2 = 0.3 #the minimum difference the coach wants to detect

r=1 #each goalkeeper should face the same number of shots
alpha = 0.25 #type I error, rejecting true null. Coach is not too concerned if she thinks a goalkeeper is better when they are actually the same. The cost is using an unnecessary substitute, which is not as critical as the other error... 
beta = 0.01 #type II error, failing to reject false null. Coach really wants to make sure she knows the difference if there is one
power = 1 - beta 

# COMMAND ----------

# MAGIC %md
# MAGIC The coach puts the values into the calculator and...

# COMMAND ----------

# MAGIC %md
# MAGIC ![picture of sample calculator](pics/sample_calc.png)

# COMMAND ----------

# MAGIC %md
# MAGIC Each goalkeeper needs to face 1964 shots to ensure we pick up the difference!!!

# COMMAND ----------

# MAGIC %md
# MAGIC ## Simulation

# COMMAND ----------

# MAGIC %md
# MAGIC Let's take a brief pause from the story and simulate what this would look like.

# COMMAND ----------

#set up the experiment
import numpy as np
from scipy.stats import ttest_ind

def t_experiment(p1_name, p1, p2_name, p2, n, seed):
    np.random.seed(seed)  # Set the seed for reproducibility

    #simulate each goalkeeper facing n shots. 1 is a save, 0 is a goal
    p1_results = np.random.choice([1, 0], size=n, p=[p1, 1-p1])
    p2_results = np.random.choice([1, 0], size=n, p=[p2, 1-p2])

    # Perform a 2-sided test
    t_stat, p_value = ttest_ind(p1_results, p2_results, alternative='two-sided')

    #interpret test
    if p_value < alpha:
        r = 'GOALKEEPERS HAVE STATISTICALLY DIFFERENT SAVE RATES'
    else:
        r = 'GOALKEEPERS HAVE STATISTICALLY EQUIVALENT SAVE RATES'

    #print results
    print(p1_name, "observed save rate:", p1_results.mean())
    print(p2_name, "observed save rate:", p2_results.mean())
    print('t-statistic:', t_stat)
    print('p-value:', p_value)
    print()
    print(r)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Scenario 1

# COMMAND ----------

# MAGIC %md
# MAGIC In scenario 1, let's say Buffon has a save rate of 25% and Casillas has a save rate of 30%

# COMMAND ----------

buffon_rate = 0.25
casillas_rate = 0.3
n=1964
seed=123 #for reproducibility

t_experiment('Buffon', buffon_rate, 'Casillas', casillas_rate, n, 123)

# COMMAND ----------

# MAGIC %md
# MAGIC We correctly identified that Casillas is the better goalkeeper

# COMMAND ----------

# MAGIC %md
# MAGIC ### Scenario 2 

# COMMAND ----------

# MAGIC %md
# MAGIC But now let's say they are a bit closer. Buffon's save rate is 28% and Casillas's is 30%

# COMMAND ----------

buffon_rate = 0.28
casillas_rate = 0.3
n=1964
seed=123 #for reproducibility

t_experiment('Buffon', buffon_rate, 'Casillas', casillas_rate, n, 123)

# COMMAND ----------

# MAGIC %md
# MAGIC We spent nearly 4000 penalty kicks to not even pick up the different save rates

# COMMAND ----------

# MAGIC %md
# MAGIC ### Scenario 3

# COMMAND ----------

# MAGIC %md 
# MAGIC Now let's say the opposite, Casillas is much better than Buffon. Buffon's save rate is 20% and Casillas's is 30%

# COMMAND ----------

buffon_rate = 0.2
casillas_rate = 0.3
n=1964
seed=123 #for reproducibility

t_experiment('Buffon', buffon_rate, 'Casillas', casillas_rate, n, 123)

# COMMAND ----------

# MAGIC %md 
# MAGIC Our test clearly picks up the difference. But we could have gotten the same information in far fewer trials, with each goalkeeper facing only 40 shots (2% of the original planned experiment)

# COMMAND ----------

buffon_rate = 0.2
casillas_rate = 0.3
n=40
seed=123 #for reproducibility

t_experiment('Buffon', buffon_rate, 'Casillas', casillas_rate, n, 123)

# COMMAND ----------

# MAGIC %md
# MAGIC Our simulation laid bare some of the drawbacks of the classical approach to this problem:
# MAGIC
# MAGIC - The coach does not want to spend hours taking 4000 penalty kicks. That time could be spent on more impactful training
# MAGIC - If the goalkeepers have similar, but not *equivalent*, save rates (as in scenario 2), we still will not identify the correct goalkeeper
# MAGIC - If the goalkeepers are vastly different (as in scenario 3), we don't want to complete the experiment
# MAGIC - What if the team faces a shootout before they can complete the experiment? They need a way to choose which goalkeeper with incomplete data

# COMMAND ----------

# MAGIC %md
# MAGIC This exposes a tension called *The Explore Exploit Dilemma*.
# MAGIC
# MAGIC On one hand we want to explore and collect more data (have both goalkeepers face more shots).
# MAGIC
# MAGIC On the other hand, we want to exploit and make sure the better goalkeeper faces as many shots as possible, since he will save more.
# MAGIC
# MAGIC Our first approach does not handle this tension in a satisfactory way. Let's turn to a new way of thinking.

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


