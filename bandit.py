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
import seaborn as sns

#goalkeeper colors
blue = '#0358B4' #italian blue
blue_shade = '#afd5fe'
red = '#F81635' #spanish red
red_shade = '#fc9ca9'
teal = '#056E73' #german retro teal
teal_shade = '#89fad8'

#generic colors
black = '#000000' #actuals
gray =  '#BCC3C1' #uncertainty
orange = '#FF7F0E' #target

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

# MAGIC %md
# MAGIC #### Experiment

# COMMAND ----------

def plot_exploit_rate_experiment(df):
    """
    Given a dataframe output from a single experiment, plot the cumulative exploit rate
    """

    fig, ax = plt.subplots(figsize=(6, 4))

    #plot target and actual exploit rates
    ax.plot(df["trial_id"], df["cumulative_exploit_rate"], label='Actual Exploit Rate', color=black)
    ax.plot(df["trial_id"], df["target_exploit_rate"], label='Target Exploit Rate', color=orange, linestyle="--")

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
    ax.plot(df["trial_id"], df["target_save_rate"], label='Target Save Rate', color=orange, linestyle="--")

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
    ax.plot(df["trial_id"], df["target_optimal_rate"], label='Target Optimal Rate', color=orange, linestyle="--")

    #y should be betwee 0 and 1
    ax.set_ylim(0, 1.05) 

    #labels
    ax.set_xlabel('Trial ID')
    ax.set_ylabel('Actual Optimal Rate')
    ax.set_title('Cumulative Optimal Rate over Trials', fontsize=10, loc='left')
    ax.legend(fontsize=8)

    plt.show()

# COMMAND ----------

def plot_p_est_experiment(df):
    """
    Given a dataframe output from a single experiment, plot the estimated and true save rates for each goalkeeper over time
    """

    fig, ax = plt.subplots(figsize=(6, 4))

    #plot target and actual exploit rates
    ax.plot(df["trial_id"], df["Buffon_est_p"], label='Buffon Estimated Save Rate', color=blue_shade)
    ax.plot(df["trial_id"], df["Buffon_p"], label='Buffon True Save Rate', color=blue, linestyle="--")

    ax.plot(df["trial_id"], df["Casillas_est_p"], label='Casillas Estimated Save Rate', color=red_shade)
    ax.plot(df["trial_id"], df["Casillas_p"], label='Casillas True Save Rate', color=red, linestyle="--")

    ax.plot(df["trial_id"], df["Neuer_est_p"], label='Neuer Estimated Save Rate', color=teal_shade)
    ax.plot(df["trial_id"], df["Neuer_p"], label='Neuer True Save Rate', color=teal, linestyle="--")

    #y should be betwee 0 and 1
    ax.set_ylim(0, 1) 

    #labels
    ax.set_xlabel('Trial ID')
    ax.set_ylabel('Save Rate')
    ax.set_title('Estimated Save Rates over Trials', fontsize=10, loc='left')
    ax.legend(fontsize=8)

    plt.show()

# COMMAND ----------

def plot_top_goalkeeper_experiment(df):
    """
    Given a dataframe output from a single experiment, plot the estimated best goalkeeper over time
    """

    # PREPARE DATAFRAME

    #map gk_names to colors
    colors = ['blue', 'red', 'teal']
    color_map = {'Buffon': colors[0], 'Casillas': colors[1], 'Neuer': colors[2]}

    #create df for heatmap. experiments are rows; columns are trials
    experiment_ids = df['experiment_id'].unique()
    trial_ids = df['trial_id'].unique()
    heatmap_data = pd.DataFrame(index=experiment_ids, columns=trial_ids)

    #fill heatmap using 'top_goalkeeper' column mapped to color
    for index, row in df.iterrows():
        goalkeeper = row['top_goalkeeper']
        trial_id = row['trial_id']
        experiment_id = row['experiment_id']
        heatmap_data.loc[experiment_id, trial_id] = color_map[goalkeeper]

    #convert colors to numeric values for heatmap
    color_to_num = {color: i for i, color in enumerate(colors)}
    heatmap_data_numeric = heatmap_data.replace(color_to_num)

    # PLOTTING

    #plot heatmap
    fig, ax = plt.subplots(figsize=(15, 2))
    sns.heatmap(heatmap_data_numeric, cmap=colors, cbar=False, ax=ax)

    #label experiment id
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_ylabel('Experiment ID')

    #label every 100th trial id
    ax.set_xticks(range(0, max(trial_ids), 100))
    ax.set_xticklabels(range(0, max(trial_ids), 100), rotation=0)
    ax.set_xlabel('Trial ID')

    #set title
    ax.set_title('Top Goalkeeper After Each Trial', fontsize=10, loc='left')
    #add legend
    handles = [plt.Rectangle((0,0),1,1, color=color_map[gk]) for gk in color_map]
    ax.legend(handles, color_map.keys(), title="Goalkeepers", bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Simulation

# COMMAND ----------

def plot_exploit_rate_simulation(df):
    """
    Given a dataframe output from a simulation, plot the cumulative exploit rate
    """

    #aggregate experiments
    df_grp = (
        df
        .groupby('trial_id')
        .agg(
            exploit_rate_mu=('cumulative_exploit_rate', 'mean'),
            exploit_rate_std=('cumulative_exploit_rate', 'std'),
            target_exploit_rate=('target_exploit_rate', 'max')
            )
    )

    #reset index
    df_grp = df_grp.reset_index()

    #calculate range
    df_grp['exploit_rate_max'] = df_grp['exploit_rate_mu'] + df_grp['exploit_rate_std']
    df_grp['exploit_rate_min'] = df_grp['exploit_rate_mu'] - df_grp['exploit_rate_std']

    #plot data
    fig, ax = plt.subplots(figsize=(6, 4))

    #plot average and target rates
    ax.plot(df_grp["trial_id"], df_grp["exploit_rate_mu"], label='Avg. Exploit Rate', color=black)
    ax.plot(df_grp["trial_id"], df_grp["target_exploit_rate"], label='Target Exploit Rate', color=orange, linestyle="--")

    # Shade between standard deviation
    ax.fill_between(df_grp["trial_id"], df_grp["exploit_rate_min"], df_grp["exploit_rate_max"], color=gray, label='Std. Dev.')

    #y should be betwee 0 and 1
    ax.set_ylim(0, 1) 

    #labels
    ax.set_xlabel('Trial ID')
    ax.set_ylabel('Exploit Rate')
    ax.set_title('Estimated Cumulative Exploit Rate over Trials', fontsize=10, loc='left')
    ax.legend(fontsize=8)

    plt.show()

# COMMAND ----------

def plot_save_rate_simulation(df):
    """
    Given a dataframe output from a simulation, plot the cumulative save rate
    """

    #aggregate experiments
    df_grp = (
        df
        .groupby('trial_id')
        .agg(
            save_rate_mu=('cumulative_save_rate', 'mean'),
            save_rate_std=('cumulative_save_rate', 'std'),
            target_save_rate=('target_save_rate', 'max')
            )
    )

    #reset index
    df_grp = df_grp.reset_index()

    #calculate range
    df_grp['save_rate_max'] = df_grp['save_rate_mu'] + df_grp['save_rate_std']
    df_grp['save_rate_min'] = df_grp['save_rate_mu'] - df_grp['save_rate_std']

    #plot data
    fig, ax = plt.subplots(figsize=(6, 4))

    #plot average and target rates
    ax.plot(df_grp["trial_id"], df_grp["save_rate_mu"], label='Avg. Save Rate', color=black)
    ax.plot(df_grp["trial_id"], df_grp["target_save_rate"], label='Target Save Rate', color=orange, linestyle="--")

    #shade between standard deviation
    ax.fill_between(df_grp["trial_id"], df_grp["save_rate_min"], df_grp["save_rate_max"], color=gray, label='Std. Dev.')

    #y should be betwee 0 and 1
    ax.set_ylim(0, 1) 

    #labels
    ax.set_xlabel('Trial ID')
    ax.set_ylabel('Save Rate')
    ax.set_title('Estimated Cumulative Save Rate over Trials', fontsize=10, loc='left')
    ax.legend(fontsize=8)

    plt.show()

# COMMAND ----------

def plot_optimal_rate_simulation(df):
    """
    Given a dataframe output from a simulation, plot the cumulative optimal rate
    """

    #aggregate experiments
    df_grp = (
        df
        .groupby('trial_id')
        .agg(
            optimal_rate_mu=('cumulative_optimal_rate', 'mean'),
            optimal_rate_std=('cumulative_optimal_rate', 'std'),
            target_optimal_rate=('target_optimal_rate', 'max')
            )
    )

    #reset index
    df_grp = df_grp.reset_index()

    #calculate range
    df_grp['optimal_rate_max'] = df_grp['optimal_rate_mu'] + df_grp['optimal_rate_std']
    df_grp['optimal_rate_min'] = df_grp['optimal_rate_mu'] - df_grp['optimal_rate_std']

    #plot data
    fig, ax = plt.subplots(figsize=(6, 4))

    #plot average and target rates
    ax.plot(df_grp["trial_id"], df_grp["optimal_rate_mu"], label='Avg. Optimal Rate', color=black)
    ax.plot(df_grp["trial_id"], df_grp["target_optimal_rate"], label='Target Optimal Rate', color=orange, linestyle="--")

    #shade between standard deviation
    ax.fill_between(df_grp["trial_id"], df_grp["optimal_rate_min"], df_grp["optimal_rate_max"], color=gray, label='Std. Dev.')

    #y should be betwee 0 and 1
    ax.set_ylim(0, 1.05) 

    #labels
    ax.set_xlabel('Trial ID')
    ax.set_ylabel('Optimal Rate')
    ax.set_title('Estimated Cumulative Optimal Rate over Trials', fontsize=10, loc='left')
    ax.legend(fontsize=8)

    plt.show()

# COMMAND ----------

def plot_p_est_simulation(df):
    """
    Given a dataframe output from a simulation, plot the estimated and true save rates for each goalkeeper over time
    """

    #aggregate experiments
    df_grp = (
        df
        .groupby('trial_id')
        .agg(
            Buffon_est_p_mu=('Buffon_est_p', 'mean'),
            Buffon_est_p_std=('Buffon_est_p', 'std'),
            Buffon_p=('Buffon_p', 'max'),
            Casillas_est_p_mu=('Casillas_est_p', 'mean'),
            Casillas_est_p_std=('Casillas_est_p', 'std'),
            Casillas_p=('Casillas_p', 'max'),
            Neuer_est_p_mu=('Neuer_est_p', 'mean'),
            Neuer_est_p_std=('Neuer_est_p', 'std'),
            Neuer_p=('Neuer_p', 'max'),
            )
    )

    #reset index
    df_grp = df_grp.reset_index()

    #calculate range
    df_grp['Buffon_p_max'] = df_grp['Buffon_est_p_mu'] + df_grp['Buffon_est_p_std']
    df_grp['Buffon_p_min'] = df_grp['Buffon_est_p_mu'] - df_grp['Buffon_est_p_std']
    df_grp['Casillas_p_max'] = df_grp['Casillas_est_p_mu'] + df_grp['Casillas_est_p_std']
    df_grp['Casillas_p_min'] = df_grp['Casillas_est_p_mu'] - df_grp['Casillas_est_p_std']
    df_grp['Neuer_p_max'] = df_grp['Neuer_est_p_mu'] + df_grp['Neuer_est_p_std']
    df_grp['Neuer_p_min'] = df_grp['Neuer_est_p_mu'] - df_grp['Neuer_est_p_std']

    #plot data
    fig, ax = plt.subplots(figsize=(6, 4))

    #plot average and target rates
    ax.plot(df_grp["trial_id"], df_grp["Buffon_est_p_mu"], label='Avg. Buffon Estimated Save Rate', color=blue)
    ax.plot(df_grp["trial_id"], df_grp["Buffon_p"], label='Buffon True Save Rate', color=gray, linestyle="-.")

    #shade between standard deviation
    ax.fill_between(df_grp["trial_id"], df_grp["Buffon_p_min"], df_grp["Buffon_p_max"], color=blue_shade, alpha=0.3, label='Buffon Std. Dev.')

    #plot average and target rates
    ax.plot(df_grp["trial_id"], df_grp["Casillas_est_p_mu"], label='Avg. Casillas Estimated Save Rate', color=red)
    ax.plot(df_grp["trial_id"], df_grp["Casillas_p"], label='Casillas True Save Rate', color=gray, linestyle=":")

    #shade between standard deviation
    ax.fill_between(df_grp["trial_id"], df_grp["Casillas_p_min"], df_grp["Casillas_p_max"], color=red_shade, alpha=0.3, label='Casillas Std. Dev.')

    #plot average and target rates
    ax.plot(df_grp["trial_id"], df_grp["Neuer_est_p_mu"], label='Avg. Neuer Estimated Save Rate', color=teal)
    ax.plot(df_grp["trial_id"], df_grp["Neuer_p"], label='Neuer True Save Rate', color=gray, linestyle="--")

    #shade between standard deviation
    ax.fill_between(df_grp["trial_id"], df_grp["Neuer_p_min"], df_grp["Neuer_p_max"], color=teal_shade, alpha=0.3, label='Neuer Std. Dev.')

    #set the y axis as max of upper limit for any goalkeeper
    y_max = df_grp[['Buffon_p_max', 'Casillas_p_max', 'Neuer_p_max']].max().max() +0.05
    ax.set_ylim(0, y_max) 

    #labels
    ax.set_xlabel('Trial ID')
    ax.set_ylabel('Save Rate')
    ax.set_title('Estimated Save Rates over Trials', fontsize=10, loc='left')
    ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.show()

# COMMAND ----------

def plot_top_goalkeeper_simulation(df):
    """
    Given a dataframe output from a simulation, plot the estimated best goalkeeper over time
    """

    # PREPARE DATAFRAME

    #map gk_names to colors
    colors = ['blue', 'red', 'teal']
    color_map = {'Buffon': colors[0], 'Casillas': colors[1], 'Neuer': colors[2]}

    #create df for heatmap. experiments are rows; columns are trials
    experiment_ids = df['experiment_id'].unique()
    trial_ids = df['trial_id'].unique()
    heatmap_data = pd.DataFrame(index=experiment_ids, columns=trial_ids)

    #fill heatmap using 'top_goalkeeper' column mapped to color
    for index, row in df.iterrows():
        goalkeeper = row['top_goalkeeper']
        trial_id = row['trial_id']
        experiment_id = row['experiment_id']
        heatmap_data.loc[experiment_id, trial_id] = color_map[goalkeeper]

    #convert colors to numeric values for heatmap
    color_to_num = {color: i for i, color in enumerate(colors)}
    heatmap_data_numeric = heatmap_data.replace(color_to_num)

    # PLOTTING

    #plot heatmap
    fig, ax = plt.subplots(figsize=(15, 5))
    sns.heatmap(heatmap_data_numeric, cmap=colors, cbar=False, ax=ax)

    #label every 10th experiment id
    ax.set_yticks(range(0, max(experiment_ids), 10))
    ax.set_yticklabels(range(0, max(experiment_ids), 10), rotation=0)
    ax.set_ylabel('Experiment ID')

    #label every 100th trial id
    ax.set_xticks(range(0, max(trial_ids), 100))
    ax.set_xticklabels(range(0, max(trial_ids), 100), rotation=0)
    ax.set_xlabel('Trial ID')

    #set title
    ax.set_title('Top Goalkeeper After Each Trial', fontsize=10, loc='left')
    #add legend
    handles = [plt.Rectangle((0,0),1,1, color=color_map[gk]) for gk in color_map]
    ax.legend(handles, color_map.keys(), title="Goalkeepers", bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.show()

# COMMAND ----------

def plot_top_goalkeeper_rate_simulation(df):
    """
    Given a dataframe output from a simulation, plot the rate that we chose each goalkeeper over time
    """

    #after each trial, count how many times we woulc pick each goalkeeper
    df_grp = (
        df
        .groupby('trial_id')['top_goalkeeper']
        .value_counts()
        .unstack(fill_value=0)
    )

    #count number of experiments
    df_grp['experiments'] = df_grp.sum(axis=1)

    for gk in df_sim['top_goalkeeper'].unique():
        df_grp[gk+'_rate'] = df_grp[gk] / df_grp['experiments']

    df_grp = df_grp.reset_index()

    #plot data
    fig, ax = plt.subplots(figsize=(6, 4))

    #plot average and target rates
    ax.plot(df_grp["trial_id"], df_grp["Buffon_rate"], label='Buffon', color=blue)
    ax.plot(df_grp["trial_id"], df_grp["Casillas_rate"], label='Casillas', color=red)
    ax.plot(df_grp["trial_id"], df_grp["Neuer_rate"], label='Neuer', color=teal)

    #y should be betwee 0 and 1
    ax.set_ylim(0, 1) 

    #labels
    ax.set_xlabel('Trial ID')
    ax.set_ylabel('Rate Selected')
    ax.set_title('Selected Rate over Trials', fontsize=10, loc='left')
    ax.legend(fontsize=8)

    plt.show()

# COMMAND ----------

def plot_simulation_summary(df):
    """
    Given a dataframe output from a simulation, plot all relevant graphs
    """

    #set up grid
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(nrows=3, ncols=2, sharex=True, figsize=(12, 12))

    # EXPLOIT RATE

    #aggregate experiments
    df_grp = (
        df
        .groupby('trial_id')
        .agg(
            exploit_rate_mu=('cumulative_exploit_rate', 'mean'),
            exploit_rate_std=('cumulative_exploit_rate', 'std'),
            target_exploit_rate=('target_exploit_rate', 'max')
            )
    )

    #reset index
    df_grp = df_grp.reset_index()

    #calculate range
    df_grp['exploit_rate_max'] = df_grp['exploit_rate_mu'] + df_grp['exploit_rate_std']
    df_grp['exploit_rate_min'] = df_grp['exploit_rate_mu'] - df_grp['exploit_rate_std']

    #plot average and target rates
    ax1.plot(df_grp["trial_id"], df_grp["exploit_rate_mu"], label='Avg. Exploit Rate', color=black)
    ax1.plot(df_grp["trial_id"], df_grp["target_exploit_rate"], label='Target Exploit Rate', color=orange, linestyle="--")

    # Shade between standard deviation
    ax1.fill_between(df_grp["trial_id"], df_grp["exploit_rate_min"], df_grp["exploit_rate_max"], color=gray, label='Std. Dev.')

    #y should be betwee 0 and 1
    ax1.set_ylim(0, 1) 

    #labels
    # ax1.set_xlabel('Trial ID')
    ax1.set_ylabel('Exploit Rate')
    ax1.set_title('Estimated Cumulative Exploit Rate over Trials', fontsize=10, loc='left')
    ax1.legend(fontsize=8)

    ## SAVE RATE

    #aggregate experiments
    df_grp = (
        df
        .groupby('trial_id')
        .agg(
            save_rate_mu=('cumulative_save_rate', 'mean'),
            save_rate_std=('cumulative_save_rate', 'std'),
            target_save_rate=('target_save_rate', 'max')
            )
    )

    #reset index
    df_grp = df_grp.reset_index()

    #calculate range
    df_grp['save_rate_max'] = df_grp['save_rate_mu'] + df_grp['save_rate_std']
    df_grp['save_rate_min'] = df_grp['save_rate_mu'] - df_grp['save_rate_std']

    #plot average and target rates
    ax2.plot(df_grp["trial_id"], df_grp["save_rate_mu"], label='Avg. Save Rate', color=black)
    ax2.plot(df_grp["trial_id"], df_grp["target_save_rate"], label='Target Save Rate', color=orange, linestyle="--")

    #shade between standard deviation
    ax2.fill_between(df_grp["trial_id"], df_grp["save_rate_min"], df_grp["save_rate_max"], color=gray, label='Std. Dev.')

    #y should be betwee 0 and 1
    ax2.set_ylim(0, 1) 

    #labels
    # ax2.set_xlabel('Trial ID')
    ax2.set_ylabel('Save Rate')
    ax2.set_title('Estimated Cumulative Save Rate over Trials', fontsize=10, loc='left')
    ax2.legend(fontsize=8)

    # OPTIMAL RATE

    #aggregate experiments
    df_grp = (
        df
        .groupby('trial_id')
        .agg(
            optimal_rate_mu=('cumulative_optimal_rate', 'mean'),
            optimal_rate_std=('cumulative_optimal_rate', 'std'),
            target_optimal_rate=('target_optimal_rate', 'max')
            )
    )

    #reset index
    df_grp = df_grp.reset_index()

    #calculate range
    df_grp['optimal_rate_max'] = df_grp['optimal_rate_mu'] + df_grp['optimal_rate_std']
    df_grp['optimal_rate_min'] = df_grp['optimal_rate_mu'] - df_grp['optimal_rate_std']

    #plot average and target rates
    ax3.plot(df_grp["trial_id"], df_grp["optimal_rate_mu"], label='Avg. Optimal Rate', color=black)
    ax3.plot(df_grp["trial_id"], df_grp["target_optimal_rate"], label='Target Optimal Rate', color=orange, linestyle="--")

    #shade between standard deviation
    ax3.fill_between(df_grp["trial_id"], df_grp["optimal_rate_min"], df_grp["optimal_rate_max"], color=gray, label='Std. Dev.')

    #y should be betwee 0 and 1
    ax3.set_ylim(0, 1.05) 

    #labels
    # ax3.set_xlabel('Trial ID')
    ax3.set_ylabel('Optimal Rate')
    ax3.set_title('Estimated Cumulative Optimal Rate over Trials', fontsize=10, loc='left')
    ax3.legend(fontsize=8)

    # P EST

    #aggregate experiments
    df_grp = (
        df
        .groupby('trial_id')
        .agg(
            Buffon_est_p_mu=('Buffon_est_p', 'mean'),
            Buffon_est_p_std=('Buffon_est_p', 'std'),
            Buffon_p=('Buffon_p', 'max'),
            Casillas_est_p_mu=('Casillas_est_p', 'mean'),
            Casillas_est_p_std=('Casillas_est_p', 'std'),
            Casillas_p=('Casillas_p', 'max'),
            Neuer_est_p_mu=('Neuer_est_p', 'mean'),
            Neuer_est_p_std=('Neuer_est_p', 'std'),
            Neuer_p=('Neuer_p', 'max'),
            )
    )

    #reset index
    df_grp = df_grp.reset_index()

    #calculate range
    df_grp['Buffon_p_max'] = df_grp['Buffon_est_p_mu'] + df_grp['Buffon_est_p_std']
    df_grp['Buffon_p_min'] = df_grp['Buffon_est_p_mu'] - df_grp['Buffon_est_p_std']
    df_grp['Casillas_p_max'] = df_grp['Casillas_est_p_mu'] + df_grp['Casillas_est_p_std']
    df_grp['Casillas_p_min'] = df_grp['Casillas_est_p_mu'] - df_grp['Casillas_est_p_std']
    df_grp['Neuer_p_max'] = df_grp['Neuer_est_p_mu'] + df_grp['Neuer_est_p_std']
    df_grp['Neuer_p_min'] = df_grp['Neuer_est_p_mu'] - df_grp['Neuer_est_p_std']

    #plot average and target rates
    ax4.plot(df_grp["trial_id"], df_grp["Buffon_est_p_mu"], label='Avg. Buffon Estimated Save Rate', color=blue)
    ax4.plot(df_grp["trial_id"], df_grp["Buffon_p"], label='Buffon True Save Rate', color=gray, linestyle="-.")

    #shade between standard deviation
    ax4.fill_between(df_grp["trial_id"], df_grp["Buffon_p_min"], df_grp["Buffon_p_max"], color=blue_shade, alpha=0.3, label='Buffon Std. Dev.')

    #plot average and target rates
    ax4.plot(df_grp["trial_id"], df_grp["Casillas_est_p_mu"], label='Avg. Casillas Estimated Save Rate', color=red)
    ax4.plot(df_grp["trial_id"], df_grp["Casillas_p"], label='Casillas True Save Rate', color=gray, linestyle=":")

    #shade between standard deviation
    ax4.fill_between(df_grp["trial_id"], df_grp["Casillas_p_min"], df_grp["Casillas_p_max"], color=red_shade, alpha=0.3, label='Casillas Std. Dev.')

    #plot average and target rates
    ax4.plot(df_grp["trial_id"], df_grp["Neuer_est_p_mu"], label='Avg. Neuer Estimated Save Rate', color=teal)
    ax4.plot(df_grp["trial_id"], df_grp["Neuer_p"], label='Neuer True Save Rate', color=gray, linestyle="--")

    #shade between standard deviation
    ax4.fill_between(df_grp["trial_id"], df_grp["Neuer_p_min"], df_grp["Neuer_p_max"], color=teal_shade, alpha=0.3, label='Neuer Std. Dev.')

    #set the y axis as max of upper limit for any goalkeeper
    y_max = df_grp[['Buffon_p_max', 'Casillas_p_max', 'Neuer_p_max']].max().max() +0.05
    ax4.set_ylim(0, y_max) 

    #labels
    # ax4.set_xlabel('Trial ID')
    ax4.set_ylabel('Save Rate')
    ax4.set_title('Estimated Save Rates over Trials', fontsize=10, loc='left')
    ax4.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc='upper left')

    #TOP GOALKEEPER

    #after each trial, count how many times we woulc pick each goalkeeper
    df_grp = (
        df
        .groupby('trial_id')['top_goalkeeper']
        .value_counts()
        .unstack(fill_value=0)
    )

    #count number of experiments
    df_grp['experiments'] = df_grp.sum(axis=1)

    for gk in df_sim['top_goalkeeper'].unique():
        df_grp[gk+'_rate'] = df_grp[gk] / df_grp['experiments']

    df_grp = df_grp.reset_index()

    #plot average and target rates
    ax5.plot(df_grp["trial_id"], df_grp["Buffon_rate"], label='Buffon', color=blue)
    ax5.plot(df_grp["trial_id"], df_grp["Casillas_rate"], label='Casillas', color=red)
    ax5.plot(df_grp["trial_id"], df_grp["Neuer_rate"], label='Neuer', color=teal)

    #y should be betwee 0 and 1
    ax5.set_ylim(0, 1) 

    #labels
    ax5.set_xlabel('Trial ID')
    ax5.set_ylabel('Rate Selected')
    ax5.set_title('Selected Rate over Trials', fontsize=10, loc='left')
    ax5.legend(fontsize=8)

    # HEATMAP

    # PREPARE DATAFRAME

    #map gk_names to colors
    colors = ['blue', 'red', 'teal']
    color_map = {'Buffon': colors[0], 'Casillas': colors[1], 'Neuer': colors[2]}

    #create df for heatmap. experiments are rows; columns are trials
    experiment_ids = df['experiment_id'].unique()
    trial_ids = df['trial_id'].unique()
    heatmap_data = pd.DataFrame(index=experiment_ids, columns=trial_ids)

    #fill heatmap using 'top_goalkeeper' column mapped to color
    for index, row in df.iterrows():
        goalkeeper = row['top_goalkeeper']
        trial_id = row['trial_id']
        experiment_id = row['experiment_id']
        heatmap_data.loc[experiment_id, trial_id] = color_map[goalkeeper]

    #convert colors to numeric values for heatmap
    color_to_num = {color: i for i, color in enumerate(colors)}
    heatmap_data_numeric = heatmap_data.replace(color_to_num)

    # PLOTTING

    sns.heatmap(heatmap_data_numeric, cmap=colors, cbar=False, ax=ax6)

    #label every 10th experiment id
    ax6.set_yticks(range(0, max(experiment_ids), 10))
    ax6.set_yticklabels(range(0, max(experiment_ids), 10), rotation=0)
    ax6.set_ylabel('Experiment ID')

    #label every 100th trial id
    ax6.set_xticks(range(0, max(trial_ids), 100))
    ax6.set_xticklabels(range(0, max(trial_ids), 100), rotation=0)
    ax6.set_xlabel('Trial ID')

    #set title
    ax6.set_title('Top Goalkeeper After Each Trial', fontsize=10, loc='left')
    #add legend
    handles = [plt.Rectangle((0,0),1,1, color=color_map[gk]) for gk in color_map]
    ax6.legend(handles, color_map.keys(), title="Goalkeepers", bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run Simulation

# COMMAND ----------

#set parameters for the simulation
num_experiments = 100
num_trials = 1000
eps = 0.1
gk_save_rates = [0.2, 0.25, 0.3]
gk_names = ['Buffon', 'Casillas', 'Neuer']

df_sim = simulation(num_experiments,num_trials,eps)

#choose an experiment
df_exp = df_sim[df_sim['experiment_id']==8]

# COMMAND ----------

# MAGIC %md
# MAGIC The analyst runs his simulation with the following parameters:
# MAGIC - We run each experiment for 1000 trials
# MAGIC   - A "trial" is one shot
# MAGIC - We simulate 100 experiments
# MAGIC   - In real life, we will only do 1 experiment. Simulating 100 gives us a sense of how much we can trust the results from doing just one
# MAGIC - Buffon has a save rate of 20%, Casillas 25%, and Neuer 30%.
# MAGIC   - In real life, we won't know these beforehand, but setting them can guide our evaluation of our process
# MAGIC   - We want our method to identify Neuer as the best goalkeeper as quickly as possible
# MAGIC   - We also want our process to accurately predict the save rate of each goalkeeper 

# COMMAND ----------

# MAGIC %md
# MAGIC The coach and the analyst sit down to go over the results. The analyst starts by breaking down what happened in a single experiment.

# COMMAND ----------

# MAGIC %md
# MAGIC This first chart shows the *exploit rate* over time. Remember, 10% of the time we *explore* by selecting a goalkeeper at random to face the shot. The other 90% of the time we *exploit* and have the goalkeeper with the best estimated save rate face the shot.  
# MAGIC
# MAGIC We can see that throughout our experiment, our exploit rate stayed very close to the expected 90%

# COMMAND ----------

plot_exploit_rate_experiment(df_exp)

# COMMAND ----------

# MAGIC %md
# MAGIC Next, we chart our save rate over time. Our *save rate* is the percent of times we have saved the shot _regardless of which goalkeeper faced the shot_. 
# MAGIC
# MAGIC Our target here is 30% - the highest save rate among all the goalkeepers. If Neuer, our best goalkeeper, faced every shot, we would expect a save rate of about 30%. 
# MAGIC
# MAGIC We can see that the longer the experiemnt goes on, the close we get to our target. This suggests that throughout the experiment we get better at identifying who the best goalkeeper is and selecting them to face the next shot when we exploit. We do, however, explore throughout the entire experiment, so with this method we would not expect to ever achieve a 30% save rate.

# COMMAND ----------

plot_save_rate_experiment(df_exp)

# COMMAND ----------

# MAGIC %md
# MAGIC Remember that Neuer is our best goalkeeper. This chart shows the percent of shots that Neuer faces throughout the experiement.
# MAGIC
# MAGIC Ideally, we would want our best goalkeeper to face every shot, so the target here is 100%. 
# MAGIC
# MAGIC We can see that for the first 400 shots, we only choose Neuer about 10% of the time. This suggests we thought a different goalkeeper was superior and when we *exploited* we did not choose Neuer. After about 400 kicks, however, we correctly identified him and began having him face more shots. 

# COMMAND ----------

plot_optimal_rate_experiment(df_exp)

# COMMAND ----------

# MAGIC %md
# MAGIC This chart shows the true save rate and estimated save rate for each goalkeeper over time. The dashed lines are the true rates and the sold lines are our estimates.
# MAGIC
# MAGIC In this experiment, we can see that Buffon made a save early on while the others were scored on. This led us to believe Buffon had the best rate and have him face more shots. 
# MAGIC
# MAGIC Casillas then made a couple saves around 75 shots in. This led us to believe he was the best goalkeeper. He, however, failed to save subsequent shots and his estimate dropped back below Buffon. 
# MAGIC
# MAGIC Since we suspected Buffon and Casillas of better save rates, we actually selected Neuer so few times that he does not save a shot until the 200th shot of the experiment. With more shots, however, we see his estimated save rate climb. By the 400th shot of the experiement, his estimated save rate is above the other two goalkeepers. We continue to have him face more and more shots and his estimated save rate converges to his true 30% mark. 
# MAGIC

# COMMAND ----------

plot_p_est_experiment(df_exp)

# COMMAND ----------

# MAGIC %md
# MAGIC This final chart shows who we thought was the best goalkeeper after each shot. So for example, if we had stopped our experiement after 10 trials, the bar is blue and we would have chosen Buffon as our top goalkeeper. If we had stopped after 320 trials, the bar is red and we would have selected Casillas as our best goalkeeper. After 500 trials, the bar is teal, and we would have selected Neuer.

# COMMAND ----------

plot_top_goalkeeper_experiment(df_exp)

# COMMAND ----------

plot_exploit_rate_simulation(df_sim)

# COMMAND ----------

plot_save_rate_simulation(df_sim)

# COMMAND ----------

plot_optimal_rate_simulation(df_sim)

# COMMAND ----------

plot_p_est_simulation(df_sim)

# COMMAND ----------

plot_top_goalkeeper_simulation(df_sim)

# COMMAND ----------

plot_top_goalkeeper_rate_simulation(df_sim)

# COMMAND ----------

plot_simulation_summary(df_sim)
