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
# MAGIC - While there are tests (like the [Chi-squared](https://en.wikipedia.org/wiki/Chi-squared_test)) that scale to multiple goalkeepers, they often only test if one goalkeeper is _different_, not which one is best
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

# MAGIC %md
# MAGIC #Approach 2: Greedy Epsilon

# COMMAND ----------

# MAGIC %md
# MAGIC The coach asks her assistant if there is a better option for an experiment

# COMMAND ----------

# MAGIC %md
# MAGIC ## Epsilon Greedy

# COMMAND ----------



# COMMAND ----------

#set up a class to model each goalkeeper
class Goalkeeper:
    def __init__(self, p, name):
        #p is the true save rate
        self.p = p
        self.p_estimate = 0 #start with an estimate of 0
        self.N = 0 #samples collected so far
        self.name = name #assign name to goalkeeper

    def face_shot(self):
        #save the shot (1) with probability p
        return np.random.random() < self.p
    
    def update(self, x):
        #x is 0 or 1
        self.N += 1
        self.p_estimate = ((self.N - 1)*self.p_estimate + x) / self.N

# COMMAND ----------

def experiment():
    #get a list of all goalkeepers
    gks = [Goalkeeper(p, name) for p, name in zip(gk_save_rates, gk_names)]

    #initialize results (0 is a goal, 1 is a save)
    results = np.zeros(num_trials)

    #track number of times we explore vs. exploit
    num_times_explored = 0
    num_times_exploited = 0
    #track number of times we selected the best gk to face a shot
    num_best_gk = 0
    best_gk_j = np.argmax([gk.p for gk in gks]) #we don't know this in real life
    print('Best GK: ', [gk.name for gk in gks][np.argmax([gk.p for gk in gks])])
    print()

    #The goalkeepers will face a number of shots
    for i in range(num_trials):
        #SELECT GK

        #if the random value is less than epsilon, we explore by choosing a goalkeeper at random 
        if np.random.random() < eps:
            num_times_explored += 1 #track explore
            gk_selected = np.random.randint(len(gks)) #choose a goalkeeper at random
        else: #otherwise use the best gk
            num_times_exploited += 1 #track exploit
            gk_selected = np.argmax([gk.p_estimate for gk in gks]) #choose gk with current best rate

        if gk_selected == best_gk_j:
            num_best_gk += 1

        #FACE THE SHOT

        #face the shot. 1 if a save, 0 if a goal
        save = gks[gk_selected].face_shot()
        #udpate results array
        results[i] = save
        #update the estimate of gk save rate
        gks[gk_selected].update(save)

    
    #PRINT EXPERIMENT RESULTS
    print(f"{'Goalkeeper':<15} {'True Save Rate':<15} {'Shots Faced':<5} {'Est. Save Rate':<5}")
    for gk in gks:
        print(f'{gk.name:<15} {gk.p:<15} {gk.N:<5} {round(gk.p_estimate,3):<5}')

    print()
    print('Summary Stats')
    print('Total Saves:', results.sum())
    print('Overall Save Rate:', results.sum() / num_trials)
    print('Times Explored:', num_times_explored)
    print('Times Exploited:', num_times_exploited)
    print(f'Times {[gk.name for gk in gks][np.argmax([gk.p for gk in gks])]} faced the shot:', num_best_gk)

    # cumulative_rewards = np.cumsum(rewards)
    # win_rates = cumulative_rewards / (np.arange(num_trials) + 1)
    # plt.plot(win_rates)
    # plt.plot(np.ones(num_trials)*np.max(bandit_probs))
    # plt.show()

# COMMAND ----------

num_trials = 500
eps = 0.1
gk_save_rates = [0.2, 0.25, 0.3]
gk_names = ['Buffon', 'Casillas', 'Neuer']

# COMMAND ----------

np.random.seed(33346383)  # Set the seed for reproducibility

experiment()

# COMMAND ----------

results
