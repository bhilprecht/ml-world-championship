
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 15})

#G=Goal, OG=Own Goal, Y=Yellow Card, R=Red Card, SY = Red Card by second yellow, P=Penalty, MP=Missed Penalty, I = Substitution In, O=Substitute Out, IH= In half time?

df_matches=pd.read_csv('data_raw/WorldCupMatches.csv', sep=',')
df_players=pd.read_csv('data_raw/WorldCupPlayers.csv', sep=',')
df_cups=pd.read_csv('data_raw/WorldCups.csv', sep=',')

df_events = pd.read_csv('data_prepared/event.csv', sep=',').replace(np.nan, '', regex=True)


# In[2]:


# at first a few descriptive statistics about the goals made during all the world cups

# * How many Goals per Worldcup?
goals_per_match = df_events.groupby(['MatchID'])[['Year', 'Home Team Goals', 'Away Team Goals']].mean()
goals_per_world_cup = goals_per_match.groupby(['Year'])[['Home Team Goals', 'Away Team Goals']].sum()
print(goals_per_world_cup.sum(1))
goals_per_world_cup.sum(1).plot()
plt.show()


# In[3]:


# Okay seems like more goals occured lately, so lets check the average goals per game for each world cup

goals_per_match_ratio = df_events.groupby(['MatchID'])[['MatchID', 'Year', 'Home Team Goals', 'Away Team Goals']].mean()
goals_per_match_ratio = goals_per_match_ratio.assign(total = lambda x: x['Home Team Goals'] + x['Away Team Goals'])
goals_per_match_ratio = goals_per_match_ratio[['MatchID', 'Year', 'total']]
goals_per_match_ratio = goals_per_match_ratio.groupby(['Year'])[['MatchID', 'total']].agg({'MatchID' : ['size'], 'total' : ['sum']})
goals_per_match_ratio = goals_per_match_ratio.rename(columns={'size':'Number of Games', 'sum' : 'Amount of Goals'})
goals_per_match_ratio = goals_per_match_ratio.assign(average = lambda x: (x['MatchID']['Number of Games'].astype(dtype=float) / x['total']['Amount of Goals'].astype(dtype=float)) )
goals_per_match_ratio = goals_per_match_ratio[['average']].rename(columns={'average':'Average Goals per Game'})
print(goals_per_match_ratio)
goals_per_match_ratio.plot(kind='bar')
plt.show()

# Well it seems that yes, the average number of goals per game increased since 1930
# but lately the average was more or less consistent with some peaks in the 00´s


# In[4]:


# Next up we ask ourself: has the goalkeeper ever shot a goal?

goal_keeper_goal = df_events.loc[(df_events['EventType'] == 'G') & (df_events['Position'] == 'GK')]

print(goal_keeper_goal)
# Unfortunately no goal keeper has ever shot a goal during world cup


# In[5]:


# Next Question is: Does the team from from the country hosting the event perform better than otherwise?

# Therefore first take a look at the host and the finals for each world cup


# In[6]:


host_and_winner = df_events2.loc[(df_events2['Stage'] == 'Final')]
host_and_winner = host_and_winner[['Year', 'Host Team Name', 'Home Team Name', 'Home Team Goals', 'Away Team Name', 'Away Team Goals', 'GoldenGoal', 'Penalty', 'DecisionPenaltyAway', 'DecisionPenaltyHome']]
host_and_winner = host_and_winner.drop_duplicates()
def calcChamp(x):
    if x['Penalty']:
        return x['Home Team Name'] if x['DecisionPenaltyHome'] > x['DecisionPenaltyAway'] else x['Away Team Name']
    else:
        return x['Home Team Name'] if x['Home Team Goals'] > x['Away Team Goals'] else x['Away Team Name']
host_and_winner['Champion'] = host_and_winner.apply(lambda x : calcChamp(x), axis=1)
host_and_winner = host_and_winner[['Year', 'Host Team Name', 'Champion']]
host_and_winner


# In[ ]:


# ok has the host ever won the finals?


# In[ ]:


host_won = host_and_winner.loc[(host_and_winner['Host Team Name'] == host_and_winner['Champion'])]
print(host_won)
num_host_won = len(host_won)
num_world_cups = len(host_and_winner)
print('That makes the Host win ' + repr(len(host_won)) + ' of ' + repr(len(host_and_winner)))
print('Which is a quote of: ' + repr(float(num_host_won)/num_world_cups))

