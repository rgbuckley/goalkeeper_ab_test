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
# MAGIC The coach has two goalkeepers on her roster: Buffon and Casillas. She wants to know which goalkeeper has a higher *save rate*. The coach plans an experiment.
# MAGIC
# MAGIC > During training sessions, each goalkeeper will face a number of penalty kick shots from several different shooters. The coach will record whether each shot is saved and use this data to determine which goalkeeper has the higher save rate or if they are the same.

# COMMAND ----------

# MAGIC %md
# MAGIC *How many shots should each goalkeeper face during the experiment?*
# MAGIC
# MAGIC The coach knows that the average save rate among all goalkeepers is about [25%](https://theanalyst.com/2024/05/premier-league-penalties-like-free-goal). She reasons that it is important to be able to detect a difference of 5% or greater between the true save rates of each goalkeeper. 
# MAGIC
# MAGIC To decide how many shots each goalkeeper should face, the coach googles around online and lands on this [calculator](https://www2.ccrb.cuhk.edu.hk/stat/proportion/Casagrande.htm) from the center for clinical research and biostats. 

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
# MAGIC Our experiment clearly picks up the difference. But did we need to shoot nearly 4000 penalty kicks to learn that Casillas is superior? 
# MAGIC
# MAGIC No. We could have gotten the same information with each goalkeeper facing only 40 shots (2% of the original planned experiment)

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
# MAGIC On one hand we want to **_explore_** and collect more data (have both goalkeepers face more shots).
# MAGIC
# MAGIC On the other hand, we want to _**exploit**_ and make sure the better goalkeeper faces as many shots as possible, since he will save more.
# MAGIC
# MAGIC Our first approach does not handle this tension in a satisfactory way. Let's turn to a new way of thinking.

# COMMAND ----------

# MAGIC %md
# MAGIC #Approach 2: Greedy Epsilon

# COMMAND ----------

# MAGIC %md
# MAGIC The team has made a couple additions to its roster. First, the team has added a new goalkeeper named Neuer. The coach now has to select between the three goalkeepers which has the best save rate.
# MAGIC
# MAGIC The team has also added an analyst to use data to improve performance. The coach meets with the analyst and pitches him the problem of deciding which goalkeeper has the highest save rate. The coach wants to vet what the analyst suggests before doing the experiment, so they decide to simulate the analyst's approach and talk about the results.

# COMMAND ----------

# MAGIC %md
# MAGIC The analyst proposes this experiment.
# MAGIC
# MAGIC > Assume all goalkeepers have a save rate of 0. We then select a random number between 1 and 10.
# MAGIC - If we select 1, choose a goalkeeper at random and have them face the next shot. Update our estimated save rate for that goalkeeper based on if they save the shot or not.
# MAGIC - If we select a number between 2 and 10, choose the goalkeeper with the highest estimated save rate to face the next shot. Update our estimated save rate for that goalkeeper based on if they save the shot or not.  

# COMMAND ----------

# MAGIC %md
# MAGIC ## Simulation

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt

blue = '#0358B4' #italian blue
blue_shade = '#afd5fe'
red = '#F81635' #spanish red
red_shade = '#fc9ca9'
teal = '#056E73' #german retro teal
teal_shade = '#89f4fa'
black = '#000000'
gray =  '#BCC3C1'

# COMMAND ----------

# MAGIC %md
# MAGIC ### Code to Run Simulation

# COMMAND ----------

#set up a class to model each goalkeeper
class Goalkeeper:
    def __init__(self, p, name):
        self.p = p #p is the true save rate
        self.p_estimate = 0 #start with an estimate of 0
        self.N = 0 #samples collected so far
        self.name = name #assign name to goalkeeper

    def face_shot(self):
        #save the shot with probability p
        return np.random.random() < self.p
    
    def update(self, x):
        #x is 0 or 1
        self.N += 1 #record the faced shot
        self.p_estimate = ((self.N - 1)*self.p_estimate + x) / self.N #update our estimate of p

# COMMAND ----------

def experiment(num_trials, epsilon):
    """
    Run an epsilon-greedy experiment

    num_trials (int): the number of trials in the experiment
    epsilon (float): the probability of exploration, needs to be between 0 and 1

    df (dataframe): the results of the experiment 
    """

    #get a list of all goalkeepers
    gks = [Goalkeeper(p, name) for p, name in zip(gk_save_rates, gk_names)]

    #initialize results
    trial_ids = []
    exploit_results = []
    gk_results = []
    save_results = []
    p_est_results = []
    chosen_results = []
     
    #For each trial in the experiment, decided whether to exploit, select your gk, and face the shot
    for i in range(num_trials):
        #SELECT GK
        trial_ids.append(i+1) #record trial id

        #if the random value is less than epsilon, we explore by choosing a goalkeeper at random 
        if np.random.random() < epsilon:
            exploit_results.append(False) #record explore
            gk_selected = np.random.randint(len(gks)) #choose a goalkeeper at random
            gk_results.append(gks[gk_selected].name) #record which gk we selected
        else: #otherwise use the best gk
            exploit_results.append(True) #recrod exploit
            gk_selected = np.argmax([gk.p_estimate for gk in gks]) #choose gk with current best rate
            gk_results.append(gks[gk_selected].name) #record which gk we selected

        #FACE THE SHOT

        #face the shot. 1 if a save, 0 if a goal
        save = gks[gk_selected].face_shot()
        #udpate results array
        save_results.append(save)
        #update the estimate of gk save rate
        gks[gk_selected].update(save)
        #record estimates
        p_est_results.append([gk.p_estimate for gk in gks])
        #record chosen gk after this trial
        chosen_results.append(gks[np.argmax([gk.p_estimate for gk in gks])].name)

    #determine array of optimal choice
    opt = gks[np.argmax([gk.p for gk in gks])].name #optimal gk
    opt_results = [g == opt for g in gk_results]
    
    #compile results in a dataframe
    df = pd.DataFrame({
        **{
            'trial_id': trial_ids,
            'exploit': exploit_results,
            'goalkeeper': gk_results,
            'optimal_goalkeeper': opt_results,
            'save': save_results,
            },
        **{gks[i].name+'_est_p': [x[i] for x in p_est_results] for i in range(len(gks))},
        **{
            'top_goalkeeper': chosen_results,
            }
        }
                      )
    
    #add true save rates
    for g in gks:
        df[g.name+'_p'] = g.p
    
    #Add cumulative results
    df['cumulative_exploit'] = df['exploit'].cumsum()
    df['cumulative_exploit_rate'] = df['cumulative_exploit'] / df['trial_id']
    df['target_exploit_rate'] = 1-epsilon
    df['cumulative_optimal_goalkeeper'] = df['optimal_goalkeeper'].cumsum()
    df['cumulative_optimal_rate'] = df['cumulative_optimal_goalkeeper'] / df['trial_id']
    df['target_optimal_rate'] = 1
    df['cumulative_saves'] = df['save'].cumsum()
    df['cumulative_save_rate'] = df['cumulative_saves'] / df['trial_id']
    df['target_save_rate'] = np.max([gk.p for gk in gks])
    
    #record parameters
    df['num_trials'] = num_trials
    df['epsilon'] = epsilon
    
    return df

# COMMAND ----------

def simulation(num_experiments, num_trials, epsilon):
    """
    Run a simulation of epsilon greedy experiments

    num_experiments (int): the number of experiments in the simulation
    num_trials (int): the number of trials per experiment
    epsilon (float): the probability of exploration, needs to be between 0 and 1

    sim_df (dataframe): the results of the simulation 
    """

    #initialize df
    i = 1
    np.random.seed(i) #set seed
    sim_df = experiment(num_trials, epsilon)
    sim_df['experiment_id'] = i

    #iterate through other trials
    for i in range(1, num_experiments):
        np.random.seed(i+1)  #set seed
        df = experiment(num_trials, epsilon)
        df['experiment_id'] = i+1
        sim_df = pd.concat([sim_df, df])

    return sim_df


# COMMAND ----------

# MAGIC %md
# MAGIC ### Code to Plot Results

# COMMAND ----------

def plot_exploit_rate_experiment(df):
    """
    Given a dataframe output from a single experiment, plot the cumulative exploit rate
    """

    fig, ax = plt.subplots(figsize=(6, 4))

    #plot target and actual exploit rates
    ax.plot(df["trial_id"], df["cumulative_exploit_rate"], label='Actual Exploit Rate', color=black)
    ax.plot(df["trial_id"], df["target_exploit_rate"], label='Target Exploit Rate', color=gray, linestyle="--")

    #y should be betwee 0 and 1
    ax.set_ylim(0, 1) 

    #labels
    ax.set_xlabel('Trial ID')
    ax.set_ylabel('Actual Exploit Rate')
    ax.set_title('Cumulative Exploit Rate over Trials', fontsize=10, loc='left')
    ax.legend(fontsize=8)

    plt.show()

# COMMAND ----------

def plot_save_rate_experiment(df):
    """
    Given a dataframe output from a single experiment, plot the cumulative save rate
    """

    fig, ax = plt.subplots(figsize=(6, 4))

    #plot target and actual exploit rates
    ax.plot(df["trial_id"], df["cumulative_save_rate"], label='Actual Save Rate', color=black)
    ax.plot(df["trial_id"], df["target_save_rate"], label='Target Save Rate', color=gray, linestyle="--")

    #y should be betwee 0 and 1
    ax.set_ylim(0, 1) 

    #labels
    ax.set_xlabel('Trial ID')
    ax.set_ylabel('Actual Save Rate')
    ax.set_title('Cumulative Save Rate over Trials', fontsize=10, loc='left')
    ax.legend(fontsize=8)

    plt.show()

# COMMAND ----------

def plot_optimal_rate_experiment(df):
    """
    Given a dataframe output from a single experiment, plot the cumulative optimal rate
    """

    fig, ax = plt.subplots(figsize=(6, 4))

    #plot target and actual exploit rates
    ax.plot(df["trial_id"], df["cumulative_optimal_rate"], label='Actual Optimal Rate', color=black)
    ax.plot(df["trial_id"], df["target_optimal_rate"], label='Target Optimal Rate', color=gray, linestyle="--")

    #y should be betwee 0 and 1
    ax.set_ylim(0, 1.05) 

    #labels
    ax.set_xlabel('Trial ID')
    ax.set_ylabel('Actual Optimal Rate')
    ax.set_title('Cumulative Optimal Rate over Trials', fontsize=10, loc='left')
    ax.legend(fontsize=8)

    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run Simulation

# COMMAND ----------

num_experiments = 100
num_trials = 1000
eps = 0.1
gk_save_rates = [0.2, 0.25, 0.3]
gk_names = ['Buffon', 'Casillas', 'Neuer']

df_sim = simulation(num_experiments,num_trials,eps)

# COMMAND ----------

df_exp = df_sim[df_sim['experiment_id']==33]

# COMMAND ----------

plot_exploit_rate_experiment(df_exp)

# COMMAND ----------

plot_save_rate_experiment(df_exp)

# COMMAND ----------

plot_optimal_rate_experiment(df_exp)

# COMMAND ----------

df_exp

# COMMAND ----------

fig, ax = plt.subplots(figsize=(6, 4))

#plot target and actual exploit rates
ax.plot(df_exp["trial_id"], df_exp["Buffon_est_p"], label='Buffon Estimated Save Rate', color=blue_shade)
ax.plot(df_exp["trial_id"], df_exp["Buffon_p"], label='Buffon True Save Rate', color=blue, linestyle="--")

ax.plot(df_exp["trial_id"], df_exp["Casillas_est_p"], label='Casillas Estimated Save Rate', color=red_shade)
ax.plot(df_exp["trial_id"], df_exp["Casillas_p"], label='Casillas True Save Rate', color=red, linestyle="--")

ax.plot(df_exp["trial_id"], df_exp["Neuer_est_p"], label='Neuer Estimated Save Rate', color=teal_shade)
ax.plot(df_exp["trial_id"], df_exp["Neuer_p"], label='Neuer True Save Rate', color=teal, linestyle="--")

#y should be betwee 0 and 1
ax.set_ylim(0, 1) 

#labels
ax.set_xlabel('Trial ID')
ax.set_ylabel('Save Rate')
ax.set_title('Estimated Save Rates over Trials', fontsize=10, loc='left')
ax.legend(fontsize=8)

plt.show()

# COMMAND ----------

df_test2 = df_exp[df_exp['trial_id'] < 201]

# COMMAND ----------

df_test3 = pd.concat([df_test, df_test2])

# COMMAND ----------

df_test3

# COMMAND ----------



# COMMAND ----------

for i, r in df_test3.iterrows():
    print(i,r)

# COMMAND ----------

# Assuming 'top_goalkeeper' column contains categorical values that we want to color by, 'trial_id' ranges from 1 to 1000, and 'experiment_id' identifies a row

colors = ['blue', 'red', 'teal']
color_map = dict(zip(gk_names, colors))

# Create a dataframe for the heatmap with 2 rows for each 'experiment_id'
experiment_ids = df_test3['experiment_id'].unique()
heatmap_data = pd.DataFrame(index=experiment_ids, columns=np.arange(1, 201))

# Fill the dataframe based on 'top_goalkeeper' column
for index, row in df_test3.iterrows():
    goalkeeper = row['top_goalkeeper']
    trial_id = row['trial_id']
    experiment_id = row['experiment_id']
    heatmap_data.loc[experiment_id, trial_id] = color_map[goalkeeper]

# Convert colors to numeric values for heatmap
color_to_num = {color: i for i, color in enumerate(colors)}
heatmap_data_numeric = heatmap_data.replace(color_to_num)

# Plotting
fig, ax = plt.subplots(figsize=(15, 2))
sns.heatmap(heatmap_data_numeric, cmap=colors, cbar=False, ax=ax, linewidths=0.5, linecolor='gray')

# Label each row with experiment id
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
ax.set_ylabel('Experiment ID')

# Label every 10th column with trial id
ax.set_xticks(range(0, 200, 10))
ax.set_xticklabels(range(0, 200, 10), rotation=0)
ax.set_xlabel('Trial ID')

ax.set_title('Distribution of Top Goalkeepers across Trials')

plt.show()

# COMMAND ----------

ax.get_xticklabels()

# COMMAND ----------

a = 

# COMMAND ----------

a

# COMMAND ----------

import matplotlib.pyplot as plt

# Create a Text object
text_obj = plt.text(x=0.5, y=0.5, s='Hello World')

# Display the plot
plt.show()

# COMMAND ----------

heatmap_data_numeric

# COMMAND ----------

df_exp['test'] = 45

# COMMAND ----------

df_grp = (
    df
    .groupby('trial_id')
    .agg(
        optimal_rate=('cumulative_optimal_rate', 'mean'),
        optimal_rate_std=('cumulative_optimal_rate', 'std'),
        save_rate=('cumulative_save_rate', 'mean'),
        save_rate_std=('cumulative_save_rate', 'std'),
        target_save_rate=('target_save_rate', 'max')
        )
)

# COMMAND ----------

df_grp = df_grp.reset_index()

df_grp['save_rate_max'] = df_grp['save_rate'] + df_grp['save_rate_std']
df_grp['save_rate_min'] = df_grp['save_rate'] - df_grp['save_rate_std']

# COMMAND ----------

fig, ax = plt.subplots(figsize=(10, 6))

# Plot save_rate and target_save_rate
ax.plot(df_grp["trial_id"], df_grp["save_rate"], label='Save Rate')
ax.plot(df_grp["trial_id"], df_grp["target_save_rate"], label='Target Save Rate')

# Shade between save_rate_max and save_rate_min
ax.fill_between(df_grp["trial_id"], df_grp["save_rate_min"], df_grp["save_rate_max"], color='gray', alpha=0.5, label='Confidence Interval')

ax.set_xlabel('Trial ID')
ax.set_ylabel('Mean Cumulative Save Rate')
ax.set_title('Cumulative Save Rate by Experiment Over Trials', fontsize=14)
ax.legend()

plt.show()

# COMMAND ----------

# Assuming 'experiment_id' is a column in the dataframe and each 'experiment_id' represents a different experiment
# We will plot each experiment as a separate line in the plot

fig, ax = plt.subplots(figsize=(10, 6))

# Get unique experiment IDs
experiment_ids = df['experiment_id'].unique()

# Loop through each experiment ID and plot it
for experiment_id in experiment_ids:
    subset_df = df[df['experiment_id'] == experiment_id]
    ax.plot(subset_df["trial_id"], subset_df["cumulative_save_rate"], label=f'Experiment {experiment_id}')

ax.plot(df["trial_id"], df["target_save_rate"], label='target')
ax.set_xlabel('Trial ID')
ax.set_ylabel('Cumulative Save Rate')
ax.set_title('Cumulative Save Rate by Experiment Over Trials', fontsize=14)
# ax.legend()

plt.show()

# COMMAND ----------



# COMMAND ----------

#track number of times we explore vs. exploit
    num_times_explored = 0
    num_times_exploited = 0
    #track number of times we selected the best gk to face a shot
    num_best_gk = 0
    best_gk_j = np.argmax([gk.p for gk in gks]) #we don't know this in real life
    print('Best GK: ', [gk.name for gk in gks][np.argmax([gk.p for gk in gks])])
    print()

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
