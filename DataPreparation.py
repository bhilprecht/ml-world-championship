
# coding: utf-8

# # Data Preparation
# 
# Match data is not normalized at the moment, i.e. 
# - several events appear in one row
# - extra time, penalty and golden goal all appear in win conditions
# 
# This notebook reads raw data and saves results to csv

# In[62]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools

df_matches=pd.read_csv('data_raw/WorldCupMatches.csv', sep=',')
df_players=pd.read_csv('data_raw/WorldCupPlayers.csv', sep=',')
df_cups=pd.read_csv('data_raw/WorldCups.csv', sep=',')

df_joined_matches = pd.merge(df_matches, df_players, on=['RoundID', 'MatchID'])


# In[63]:


#G=Goal, OG=Own Goal, Y=Yellow Card, R=Red Card, SY = Red Card by second yellow, P=Penalty, MP=Missed Penalty, I = Substitution In, O=Substitute Out, IH= In half time?
df_joined_matches = df_joined_matches.replace(np.nan, '', regex=True)
# explode rows with multiple events
df_joined_matches_exploded_series = pd.DataFrame(df_joined_matches.Event.str.split(' ').tolist()).stack()
# join
df_joined_matches_exploded = pd.merge(df_joined_matches.reset_index(), df_joined_matches_exploded_series.to_frame().reset_index(), how = 'left', left_on = 'index', right_on = 'level_0')
# drop useless columns
df_joined_matches_exploded.drop(['Event','level_0','level_1','index'], 1, inplace=True)
df_joined_matches_exploded.columns.values[26] = 'Event'
df_joined_matches_exploded


# In[64]:


def filter_event(x):
    return "".join(itertools.takewhile(str.isalpha, x.Event))

def filter_minute(x):
    return ''.join(filter(lambda x: x.isdigit(), x.Event))

# Filter out EventType and Minute
df_joined_matches_exploded['EventType'] = df_joined_matches_exploded.apply(filter_event, axis=1)
df_joined_matches_exploded['EventMinute'] = df_joined_matches_exploded.apply(filter_minute, axis=1)
df_joined_matches_exploded.drop(['Event'], 1,inplace=True)

df_joined_matches_exploded.rename(columns={'Win conditions': 'WinConditions'}, inplace=True)


# In[ ]:


df_joined_matches_exploded = df_joined_matches_exploded.assign(ExtraTime = lambda x: x.WinConditions.str.contains("extra time"))
df_joined_matches_exploded = df_joined_matches_exploded.assign(Penalty = lambda x: x.WinConditions.str.contains("penalties"))
df_joined_matches_exploded = df_joined_matches_exploded.assign(GoldenGoal = lambda x: x.WinConditions.str.contains("Golden Goal"))
# Obviously Golden Goal means extra time
df_joined_matches_exploded.loc[df_joined_matches_exploded.GoldenGoal == True, 'ExtraTime'] = True
df_joined_matches_exploded.loc[df_joined_matches_exploded.Penalty == True, 'ExtraTime'] = True


# In[ ]:


def transform_penalty_away(x):
    if(x.Penalty):
        return int(x.WinConditions[-3])
    else:
        return 0
    
def transform_penalty_home(x):
    if(x.Penalty):
        return int(x.WinConditions[-7])
    else:
        return 0
    
df_joined_matches_exploded['DecisionPenaltyAway'] = df_joined_matches_exploded.apply(transform_penalty_away, axis=1)
df_joined_matches_exploded['DecisionPenaltyHome'] = df_joined_matches_exploded.apply(transform_penalty_home, axis=1)

df_joined_matches_exploded.drop(['WinConditions'], 1,inplace=True)

# extra column indicating whether home team won the match
df_joined_matches_exploded['HomeTeamWins'] = df_joined_matches_exploded.apply(lambda x: (x['Home Team Goals']>x['Away Team Goals'] or x['DecisionPenaltyHome']>x['DecisionPenaltyAway']), axis=1)
df_joined_matches_exploded['AwayTeamWins'] = df_joined_matches_exploded.apply(lambda x: (x['Home Team Goals']<x['Away Team Goals'] or x['DecisionPenaltyHome']<x['DecisionPenaltyAway']), axis=1)

# extra column indicating which team the event belongs to
df_joined_matches_exploded = df_joined_matches_exploded.assign(EventOfHomeTeam = lambda x: x['Team Initials'] == x['Home Team Initials'])
df_joined_matches_exploded['EventOfWinner'] = df_joined_matches_exploded.apply(lambda x: (x['EventOfHomeTeam'] and x['HomeTeamWins']) or (not x['EventOfHomeTeam'] and x['AwayTeamWins']), axis=1)
df_joined_matches_exploded['EventOfLoser'] = df_joined_matches_exploded.apply(lambda x: (x['EventOfHomeTeam'] and x['AwayTeamWins']) or (not x['EventOfHomeTeam'] and x['HomeTeamWins']), axis=1)

# Format datetime
df_joined_matches_exploded = df_joined_matches_exploded.assign(HourGameStart = lambda x: x.Datetime.str[-6:-4])

df_joined_matches_exploded = df_joined_matches_exploded.replace(np.nan, '', regex=True)
df_joined_matches_exploded


# In[ ]:


df_joined_matches_exploded.Stage.unique()


# In[ ]:


def compute_stage(x):
    if("Preliminary" in x['Stage']):
        return 0
    if("Group" in x['Stage']):
        return 1
    if("Round of 16" in x['Stage']):
        return 2
    if("First round" in x['Stage']):
        return 2
    if("Quarter-finals" in x['Stage']):
        return 3
    if("Semi-finals" in x['Stage']):
        return 4
    if("Play-off for third place" in x['Stage']):
        return 4
    if("Match for third place" in x['Stage']):
        return 4
    if("Third place" in x['Stage']):
        return 5
    if("Final" in x['Stage']):
        return 6
    else:
        return 99

df_joined_matches_exploded["StageRank"] = df_joined_matches_exploded.apply(lambda row: compute_stage(row), axis=1)


# In[ ]:


df_joined_matches_exploded.to_csv("data_prepared/event.csv",index=False)

