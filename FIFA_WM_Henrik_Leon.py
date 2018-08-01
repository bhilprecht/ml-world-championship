
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({'font.size': 15})

#G=Goal, OG=Own Goal, Y=Yellow Card, R=Red Card, SY = Red Card by second yellow, P=Penalty, MP=Missed Penalty, I = Substitution In, O=Substitute Out, IH= In half time?

df_matches = pd.read_csv('data_raw/WorldCupMatches.csv', sep=',')
df_players = pd.read_csv('data_raw/WorldCupPlayers.csv', sep=',')
df_cups = pd.read_csv('data_raw/WorldCups.csv', sep=',')

df_events = pd.read_csv('data_prepared/event.csv', sep=',').replace(np.nan, '', regex=True)
df_penalties = pd.read_csv('data_prepared/penalties.csv', encoding='iso-8859-1', sep=',')


# # Wie oft hat sich ein zurückliegendes Team den Sieg geholt?

# In[2]:


ht_wins_after_residual = 0
at_wins_after_residual = 0
halftime_tie = 0
regular_time_tie = 0
leading_team_wins = 0

df_grouped_events = df_events.groupby(["MatchID"]).mean()

for index, row in df_grouped_events.iterrows():
    # ignore penalties and extra time
    if row["Penalty"] == True or row["ExtraTime"] == True:
        continue
    if row["Half-time Home Goals"] == row["Half-time Away Goals"]:
        halftime_tie+=1
        continue
    if row["Home Team Goals"] == row["Away Team Goals"]:
        regular_time_tie +=1
        continue
    if row["Half-time Home Goals"] > row["Half-time Away Goals"] and row["Home Team Goals"] < row["Away Team Goals"]:
        at_wins_after_residual+=1
        continue
    if row["Half-time Home Goals"] < row["Half-time Away Goals"] and row["Home Team Goals"] > row["Away Team Goals"]:
        ht_wins_after_residual+=1
        continue  
    leading_team_wins+=1
    


# In[3]:


decision_changed_after_ht = ht_wins_after_residual + at_wins_after_residual
y = [decision_changed_after_ht, halftime_tie, regular_time_tie, leading_team_wins]
x = ["Decision changed", "Halftime tie", "Regular time tie", "Leading Team Wins"]
plt.bar(x = x, height = y)

plt.show()


# In[4]:


total_matches = decision_changed_after_ht + halftime_tie + regular_time_tie + leading_team_wins


# In[5]:


decision_changed_after_ht


# In[6]:


total_matches


# ## Antwort: Von knapp 780 Spielen werden weniger als 40 Spiele gedreht

# ### Hier wird nicht berücksichtigt: Spiele mit Elfmeterschießen oder Nachspielzeit und es wird nicht zwischen Gruppen- und KO-Phase unterschieden

# # Wann steht der Gewinner fest?

# In[7]:


# data preparation: event data since 1986
df_relevant_events=df_events.loc[(df_events['Year'] >= 1986)]


# In[8]:


df_relevant_events = df_relevant_events.groupby(["MatchID"]).mean()


# In[9]:


first_half_decided_group = 0
second_half_decided_group = 0
first_half_decided = 0
second_half_decided = 0
extra_time_decided = 0
penalty_decided = 0

for index, row in df_relevant_events.iterrows():
    if row["Home Team Goals"] == row["Half-time Home Goals"] and row["Away Team Goals"] == row["Half-time Away Goals"] and row["Penalty"] == False:
        if row["Home Team Goals"] == row["Away Team Goals"]:
            continue
        if row["StageRank"] > 1:
            first_half_decided+=1
        else:
            first_half_decided_group+=1
        continue
    if row["ExtraTime"] == True and row["Penalty"] == False:
        extra_time_decided+=1
        continue
    if row["Penalty"] == True:
        penalty_decided+=1
        continue
    if row["StageRank"] > 1:
        second_half_decided+=1
    else:
        second_half_decided_group+=1


# ## Betrachtung der KO-Phase

# In[10]:


match_decisions = [first_half_decided, second_half_decided, extra_time_decided, penalty_decided]
match_axis = ["First Half", "Second Half", "Extra Time", "Penalty"]
plt.bar(x = match_axis, height = match_decisions)

plt.show()


# ## Betrachtung der Gruppen-Phase

# In[11]:


match_decisions = [first_half_decided_group, second_half_decided_group]
match_axis = ["First Half", "Second Half"]
plt.bar(x = match_axis, height = match_decisions)

plt.show()


# # Vorhersage zu welcher Spielphase sich ein Spiel entscheidet

# In[12]:


# data prep
df_relevant_events # events since 1986
df_relevant_events = df_relevant_events.groupby(["MatchID"]).mean()
# df_relevant_events


# In[13]:


# lambda function
def calculateDecisionPhase(x):
    if x["Home Team Goals"] == x["Half-time Home Goals"] and x["Away Team Goals"] == x["Half-time Away Goals"] and x["Penalty"] == False:
        if x["Home Team Goals"] == x["Away Team Goals"]:
            return 0
        else:
            return 1
    if x["ExtraTime"] == True and x["Penalty"] == False:
        return 3
    if x["Penalty"] == True:
        return 4
    return 2


# In[14]:


def calculateWorldCupStage(x):
    if x["StageRank"] == 1:
        return 0 # group phase
    return 1 # ko phase


# In[15]:


df_relevant_events_truth = df_relevant_events
df_relevant_events_truth["DecisionPhase"] = df_relevant_events_truth.apply(calculateDecisionPhase, axis=1)
df_relevant_events_truth["WorldCupStage"] = df_relevant_events_truth.apply(calculateWorldCupStage, axis=1)

# df_relevant_events_truth


# ## KNN

# In[16]:


# Normalization
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


df_relevant_events_truth[["WorldCupStage", "Home Team Goals", "Half-time Home Goals", "Away Team Goals", "Half-time Away Goals", "ExtraTime", "Penalty"]] = scaler.fit_transform(df_relevant_events_truth[["WorldCupStage", "Home Team Goals", "Half-time Home Goals", "Away Team Goals", "Half-time Away Goals", "ExtraTime", "Penalty"]])
# Values
x = np.array(df_relevant_events_truth[["WorldCupStage", "Home Team Goals", "Half-time Home Goals", "Away Team Goals", "Half-time Away Goals", "ExtraTime", "Penalty"]])

# Labels
y = np.array(df_relevant_events_truth["DecisionPhase"]) 


# In[17]:


# Split data
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=101)


# In[18]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# fit decision tree
clf = KNeighborsClassifier()
clf = clf.fit(X_train, Y_train)

# compute accuracy
y_pred = clf.predict(X_test)
score = clf.score(X_test, Y_test)

train_score = accuracy_score(clf.predict(X_train), Y_train)


# In[19]:


train_score


# In[20]:


y_pred


# In[21]:


score


# In[22]:


# for comparison also compute accuracy for base model (always output zero)
y_base_pred = np.full(shape = (len(Y_test)), fill_value=2) # two as most matches are decided in second half 
base_score = accuracy_score(y_base_pred, Y_test)


# In[23]:


base_score


# ## Ergebnis des KNN: 
# * Knapp 440 Datensätze
# * Insgesamt 7 Features (Merkmale des Datensatzes)
#     * Home Team Goals
#     * Away Team Goals
#     * Half Time Home Goals
#     * Half Time Away Goals
#     * Entra Time?
#     * Penalty?
# * Trainings Accuracy: 96,6%
# * Test Accuracy: 93,7%
# * Base Accuracy: 69,2% (Base = 2. Halbzeit entscheidend)

# ## Decision Tree

# In[24]:


from sklearn import tree
from sklearn.metrics import accuracy_score

# fit decision tree
clf = tree.DecisionTreeClassifier(max_depth = 7)
clf = clf.fit(X_train, Y_train)

# compute accuracy
y_pred = clf.predict(X_test)
score = clf.score(X_test, Y_test)

train_score = accuracy_score(clf.predict(X_train), Y_train)


# In[25]:


y_pred


# In[26]:


score


# In[27]:


train_score


# In[28]:


import graphviz 
dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=["WorldCupStage", "Home Team Goals", "Half-time Home Goals", "Away Team Goals", "Half-time Away Goals", "ExtraTime", "Penalty"],  
                         class_names=["Tie", "HT-Decision", "RG-Decision", "Extra Time", "Penalty"],  
                         filled=True, rounded=True,  
                         special_characters=True) 

graph = graphviz.Source(dot_data)  
graph


# ## Ergebnis des Decision Tree: 
# * Knapp 440 Datensätze
# * Insgesamt 7 Features (Merkmale des Datensatzes)
#     * Home Team Goals
#     * Away Team Goals
#     * Half Time Home Goals
#     * Half Time Away Goals
#     * Entra Time?
#     * Penalty?
# * Trainings Accuracy: 98,5%
# * Test Accuracy: 98,6%
# * Base Accuracy: 69,2% (Base = 2. Halbzeit entscheidend)

# # Erfolgsquote von Strafstößen

# In[29]:


df_penalties_success_rate = df_events[(df_events.EventType.str.contains("P") == True)]


# In[30]:


f = {
     'EventType':'count'
    }

df_penalties_success_rate = df_penalties_success_rate[["EventType"]].groupby(["EventType"]).agg(f)


# In[31]:


df_penalties_success_rate


# In[32]:


df_penalties_MP = df_penalties_success_rate.loc["MP"]


# In[33]:


df_penalties_P = df_penalties_success_rate.loc["P"]


# In[34]:


df_penalties_MP


# In[35]:


df_penalties_P


# In[36]:


df_penalty_quote = print(float(df_penalties_MP) / df_penalties_P)


# ## Antwort: Nur 6,17 % der Strafstöße werden verschossen

# # General information about Football World Cups

# In[37]:


df_cups.set_index('Year').plot(kind='bar', title="General World Cup Information", figsize=(14,7), legend=True, fontsize=12)
plt.show()


# # Attendance per World Cup

# In[38]:


# Data preparation
df_cups_attendance = df_cups[["Year", "Attendance"]]
df_cups_attendance['Attendance'] = df_cups_attendance['Attendance'].str.replace('.', '')
df_cups_attendance['Attendance'] = df_cups_attendance.Attendance.astype(float)
df_cups_attendance.dtypes


# ## Total Attendance per World Cup

# In[39]:


plt.figure(figsize=(12,6))
df_cups_attendance.set_index("Year").plot(style="o-", title="Attendance per World Cup", figsize=(14,7), legend=True, fontsize=12)
plt.grid()
plt.show()


# ## Average Attendance per World Cup

# In[40]:


df_avg_attendance = df_matches.groupby("Year")["Attendance"].mean().reset_index()
df_avg_attendance["Year"] = df_avg_attendance["Year"].astype(int)
df_avg_attendance.set_index("Year").plot(kind='bar', color="b", title="Average attendance by year", figsize=(14,7), legend=True, fontsize=12)
plt.show()


# # Player with the most goals across all World Cups

# In[41]:


# https://www.weltfussball.de/ewige_torjaeger/wm/tore/1/
# data consistency not sufficient for this calculation as player names are not unique


# In[42]:


df_players_goals = df_events[["Player Name", "EventType"]]
df_players_goals = df_players_goals[df_players_goals.EventType.str.contains("G") == True]
df_players_goals = df_players_goals[df_players_goals.EventType.str.contains("P") == False]
df_players_goals = df_players_goals[df_players_goals.EventType.str.contains("I") == False]
df_players_goals = df_players_goals[df_players_goals.EventType.str.contains("Y") == False]
df_players_goals = df_players_goals[df_players_goals.EventType.str.contains("R") == False]
df_players_goals = df_players_goals[df_players_goals.EventType.str.contains("O") == False]

df_players_goals.groupby(["Player Name"])
df_players_goals.dropna()


# ## List of top scoring players

# In[43]:


f = {
     'EventType':'count'
    }

df_test = df_players_goals.groupby(['Player Name']).agg(f)
df_test['EventType'].sort_values(ascending=False)
df_top_players = df_players_goals.groupby(['Player Name']).agg(f)
df_top_players['EventType'].sort_values(ascending=False)
df_top14_players = df_top_players[df_test.EventType > 8]
df_top14_players['EventType'].sort_values(ascending=False).plot(kind='bar', title="Total goals of players", figsize=(14,7), legend=False, fontsize=12)
plt.show()


# # Fun with Penalty data
# 
# The used dataset covers only penalty shoot-outs after regular time. Penalty shoot-outs were introduced in 1972.

# In[44]:


# data preparation
df_penalties["Match Result"] = df_penalties["Match Result"].str.replace("","-")
df_penalties["Final Result"] = df_penalties["Final Result"].str.replace("","-")

df_penalties["Winner Penalty Goals"] = df_penalties["Final Result"].str.split("-").str[0]
df_penalties["Loser Penalty Goals"]  = df_penalties["Final Result"].str.split("-").str[1]


# ## Penalty overview

# In[45]:


df_penalties


# ## How many matches were dicided after penalty shoot-outs?

# In[46]:


df_penalties[['Winner']].count()


# Between 1982 and 2014 took 26 penalty shoot-outs place

# ## How many penalty matches by year?

# In[47]:


df_penalties_by_year = df_penalties['Year'].value_counts().reset_index()
df_penalties_by_year.columns = ['Year', 'Count']
df_penalties_by_year = df_penalties_by_year.sort_values(by='Year',ascending=True)
df_penalties_by_year.set_index("Year").plot(kind='bar',color='bgrcmyk', title="Penalty matches by year", figsize=(14,7), legend=False, fontsize=12)
plt.show()


# ## When decided a penalty shoot-out the world cup winner?

# In[48]:


df_penalties[df_penalties['Round'] == 'Final']


# In 1994 won Brazil against Italy with the endresult of 3:2, and in 2006 won Italy against France in the world cup final 5:3.

# ## Which team is the most successful on penalties?

# In[49]:


df_penalty_winners = df_penalties["Winner"].value_counts().reset_index()
df_penalty_winners.columns = ["Country","Number of wins"]
df_penalty_winners.sort_values(by='Number of wins',ascending=False)


# In[50]:


df_penalty_winners.set_index("Country").plot(kind="bar", color="g", title="Total penalty wins per team", figsize=(14,7), legend=False, fontsize=12)
plt.show()


# ## Which team is the less successful on penalty?

# In[51]:


df_penalty_loosers = df_penalties["Loser"].value_counts().reset_index()
df_penalty_loosers.columns = ["Country","Number of loses"]
df_penalty_loosers.sort_values(by='Number of loses',ascending=False)


# In[52]:


df_penalty_loosers.set_index("Country").plot(kind="bar", color="r", title="Total penalty loses per team", figsize=(14,7), legend=False, fontsize=12)
plt.show()

