
# coding: utf-8

# In[16]:


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


# In[17]:


df_events


# # When is a match winner decided?

# In[18]:


# data preparation: event data since 1986
df_relevant_events=df_events.loc[(df_events['Year'] >= 1986)]


# In[31]:


df_relevant_events


# In[33]:


df_relevant_events = df_relevant_events.groupby(["MatchID"]).mean()


# In[27]:


# df_goals_by_minute = df_relevant_events[(df_relevant_events.EventType.str.contains("G") == True) | (df_relevant_events.EventType.str.contains("OG") == True) | (df_relevant_events.EventType.str.contains("P") == True) & (df_relevant_events.EventType.str.contains("MP") == False)]
# df_goals_by_minute
# (df_relevant_events.Penalty == True)


# In[34]:


df_relevant_events


# In[48]:


first_half_decided_group = 0
second_half_decided_group = 0
first_half_decided = 0
second_half_decided = 0
extra_time_decided = 0
penalty_decided = 0

for index, row in df_relevant_events.iterrows():
    if row["Home Team Goals"] == row["Half-time Home Goals"] and row["Away Team Goals"] == row["Half-time Away Goals"] and row["Penalty"] == False:
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


# ## KO phase

# In[49]:


match_decisions = [first_half_decided, second_half_decided, extra_time_decided, penalty_decided]
match_axis = ["First Half", "Second Half", "Extra Time", "Penalty"]
plt.bar(x = match_axis, height = match_decisions)

plt.show()


# ## Group phase

# In[86]:


match_decisions = [first_half_decided_group, second_half_decided_group]
match_axis = ["First Half", "Second Half"]
plt.bar(x = match_axis, height = match_decisions)

plt.show()


# # Erfolgsquote von Strafstößen

# In[87]:


df_penalties_success_rate = df_events[(df_events.EventType.str.contains("P") == True)]
df_penalties_success_rate


# In[88]:


f = {
     'EventType':'count'
    }

df_penalties_success_rate = df_penalties_success_rate[["EventType"]].groupby(["EventType"]).agg(f)


# In[89]:


df_penalties_success_rate


# In[90]:


df_penalties_MP = df_penalties_success_rate.loc["MP"]
# df_penalties_P = df_penalties_success_rate.loc["P"]


# In[91]:


df_penalties_P = df_penalties_success_rate.loc["P"]


# In[92]:


df_penalties_MP


# In[93]:


df_penalties_P


# In[99]:


df_penalty_quote = print(float(df_penalties_MP) / df_penalties_P)


# ## Antwort: Nur 6,17 % der Strafstöße werden verschossen

# # Wie oft hat sich ein zurückliegendes Team den Sieg geholt?

# In[105]:


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
    


# In[106]:


decision_changed_after_ht = ht_wins_after_residual + at_wins_after_residual
y = [decision_changed_after_ht, halftime_tie, regular_time_tie, leading_team_wins]
x = ["Decision changed", "Halftime tie", "Regular time tie", "Leading Team Wins"]
plt.bar(x = x, height = y)

plt.show()


# In[107]:


total_matches = decision_changed_after_ht + halftime_tie + regular_time_tie + leading_team_wins


# In[109]:


decision_changed_after_ht


# In[108]:


total_matches


# ## Antwort: Von knapp 780 Spielen werden weniger als 40 Spiele gedreht

# ### Hier wird nicht berücksichtigt: Spiele mit Elfmeterschießen oder Nachspielzeit und es wird nicht zwischen Gruppen- und KO-Phase unterschieden

# # Vorhersage, zu welcher Spielphase sich ein Spiel entscheidet

# In[117]:


# data prep
df_relevant_events # events since 1986
df_relevant_events = df_relevant_events.groupby(["MatchID"]).mean()
df_relevant_events


# In[125]:


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


# In[130]:


def calculateWorldCupStage(x):
    if x["StageRank"] == 1:
        return 0 # group phase
    return 1 # ko phase


# In[175]:


df_relevant_events_truth = df_relevant_events
df_relevant_events_truth["DecisionPhase"] = df_relevant_events_truth.apply(calculateDecisionPhase, axis=1)
df_relevant_events_truth["WorldCupStage"] = df_relevant_events_truth.apply(calculateWorldCupStage, axis=1)

df_relevant_events_truth


# ## KNN

# In[177]:


# Normalization
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


df_relevant_events_truth[["WorldCupStage", "Home Team Goals", "Half-time Home Goals", "Away Team Goals", "Half-time Away Goals", "ExtraTime", "Penalty"]] = scaler.fit_transform(df_relevant_events_truth[["WorldCupStage", "Home Team Goals", "Half-time Home Goals", "Away Team Goals", "Half-time Away Goals", "ExtraTime", "Penalty"]])
# Values
x = np.array(df_relevant_events_truth[["WorldCupStage", "Home Team Goals", "Half-time Home Goals", "Away Team Goals", "Half-time Away Goals", "ExtraTime", "Penalty"]])

# Labels
y = np.array(df_relevant_events_truth["DecisionPhase"]) 


# In[178]:


# Split data
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=101)


# In[179]:


X_train, X_test, Y_train, Y_test


# In[180]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# fit decision tree
clf = KNeighborsClassifier()
clf = clf.fit(X_train, Y_train)

# compute accuracy
y_pred = clf.predict(X_test)
score = clf.score(X_test, Y_test)

train_score = accuracy_score(clf.predict(X_train), Y_train)


# In[181]:


train_score


# In[182]:


y_pred


# In[183]:


score


# In[184]:


# for comparison also compute accuracy for base model (always output zero)
y_base_pred = np.full(shape = (len(Y_test)), fill_value=2) # two as most matches are decided in second half 
base_score = accuracy_score(y_base_pred, Y_test)


# In[185]:


base_score


# ## Decision Tree

# In[209]:


from sklearn import tree
from sklearn.metrics import accuracy_score

# fit decision tree
clf = tree.DecisionTreeClassifier(max_depth = 7)
clf = clf.fit(X_train, Y_train)

# compute accuracy
y_pred = clf.predict(X_test)
score = clf.score(X_test, Y_test)

train_score = accuracy_score(clf.predict(X_train), Y_train)


# In[210]:


y_pred


# In[211]:


score


# In[212]:


train_score


# In[213]:


import graphviz 
dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=["WorldCupStage", "Home Team Goals", "Half-time Home Goals", "Away Team Goals", "Half-time Away Goals", "ExtraTime", "Penalty"],  
                         class_names=["Tie", "HT-Decision", "RG-Decision", "Extra Time", "Penalty"],  
                         filled=True, rounded=True,  
                         special_characters=True) 

graph = graphviz.Source(dot_data)  
graph

