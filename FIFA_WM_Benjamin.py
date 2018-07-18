
# coding: utf-8

# In[149]:


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


df_events


# In[3]:


df_cups


# In[4]:


plt.figure(figsize=(14,7))
plt.subplot(121)
df_cups.groupby(['Winner']).size().sort_values(ascending=False).plot.bar()
plt.title("World Cup Wins")
plt.xlabel("")

plt.subplot(122)
df_cups.groupby(['Runners-Up']).size().sort_values(ascending=False).plot.bar()
plt.title("2nd Prize")
plt.xlabel("")

plt.show()


# In[5]:


plt.figure(figsize=(9,6))

df_players.groupby(['Player Name']).size().sort_values(ascending=False).nlargest(n=10).plot.bar()


# In[6]:


# data consistency not sufficient for this calculation as player names are not unique

#df_timespan = df_joined_matches[['Player Name','Year']].groupby(['Player Name']).aggregate(['min','max']).Year
#df_timespan.columns = ['start','end']
#df_timespan.apply(lambda x: (x.end-x.start), axis=1).sort_values(ascending = False)


# # Analysis of Fair Play
# 
# We first need some preliminary work. How many matches were played at all? How many of them were won by one team?

# In[7]:


num_matches_total = len(df_events.groupby('MatchID').mean())
num_matches_decision = len(df_events.loc[(df_events['HomeTeamWins'] == True) | (df_events['AwayTeamWins'] == True)].groupby('MatchID').mean())
num_matches_tie = len(df_events.loc[(df_events['HomeTeamWins'] == False) & (df_events['AwayTeamWins'] == False)].groupby('MatchID').mean())

print("num_matches_total: %g"% num_matches_total)
print("num_matches_decision: %g"% num_matches_decision)
print("num_matches_tie: %g"% num_matches_tie)

print("proportion decision: %.2f"% (num_matches_decision/num_matches_total*100))
print("proportion no decision: %.2f"% (num_matches_tie/num_matches_total*100))


# ## Yellow and Red Cards Statistics
# 
# On average 2.68 yellow cards, 0.14 red cards and 0.06 red cards for second yellow are given during a match.

# In[8]:


f = {
     'Year':'count' # we could do this with any attribute
    }

df_cards = df_events.loc[(df_events["EventType"] == "Y") | (df_events["EventType"] == "R") | (df_events["EventType"] == "RSY")]
df_cards = df_cards.groupby(["EventType"]).agg(f)
df_cards.columns = ['Total']
df_cards.assign(AvgPerMatch = lambda x : x.Total/num_matches_total)


# ## Event minutes of red and yellow cards
# 
# In this section we want to find out when most yellow and red cards are given. As expected, red cards tend to be given later.

# In[74]:


df_events[['EventMinute']] = df_events[['EventMinute']].apply(pd.to_numeric)
#df_events.loc[(df_events['EventType'] == "Y") & (int(df_events['EventMinute']) < 20)]
minutes_yellow = df_events[df_events.EventType == "Y"].EventMinute.values
minutes_red = df_events[df_events.EventType == "R"].EventMinute.values
minutes_red_2nd_yellow = df_events[df_events.EventType == "RSY"].EventMinute.values

plt.figure(figsize=(12,6))

ax = plt.subplot(131)

ax.boxplot(minutes_yellow)
plt.title("Yellow Cards")

ax = plt.subplot(132)

ax.boxplot(minutes_red)
plt.title("Red Cards")

ax = plt.subplot(133)

ax.boxplot(minutes_red_2nd_yellow)
plt.title("Red Cards (By 2nd Yellow)")

plt.show()


# # Fairest team
# 
# In this section we want to find out which team is given the fewest yellow cards per match on average. First we have to find out how many yellow cards a team was awarded. We have to create to dataframes as teams appear either as home or away team.

# In[184]:


df_yellow_cards = df_events[df_events.EventType == "Y"]

df_yellow_cards_home = df_yellow_cards[df_yellow_cards.EventOfHomeTeam == True][["Home Team Name", "EventType"]]
df_yellow_cards_home = df_yellow_cards_home.groupby("Home Team Name").count().reset_index()
df_yellow_cards_home.columns = ['Team', 'YellowCardsHome']

df_yellow_cards_away = df_yellow_cards[df_yellow_cards.EventOfHomeTeam == False][["Away Team Name", "EventType"]]
df_yellow_cards_away = df_yellow_cards_away.groupby("Away Team Name").count().reset_index()
df_yellow_cards_away.columns = ['Team', 'YellowCardsAway']

df_yellow_cards_count = pd.merge(df_yellow_cards_home, df_yellow_cards_away).fillna(0)
df_yellow_cards_count['YellowCardsTotal'] = df_yellow_cards_count.YellowCardsHome+df_yellow_cards_count.YellowCardsAway
df_yellow_cards_count


# Now we need the amount of matches per team to finally compute the average amount of yellow cards per match per team.

# In[192]:


df_home_matches = df_matches[["Home Team Name"]]
df_home_matches["MatchesCount1"] = 1
df_home_matches = df_home_matches.groupby("Home Team Name").count().reset_index()
df_home_matches.columns = ['Team', 'MatchesHome']

df_away_matches = df_matches[["Away Team Name"]]
df_away_matches["MatchesCount2"] = 1
df_away_matches = df_away_matches.groupby("Away Team Name").count().reset_index()
df_away_matches.columns = ['Team', 'MatchesAway']

df_matches_count = pd.merge(df_home_matches, df_away_matches).fillna(0)
df_matches_count['MatchesTotal'] = df_matches_count.MatchesHome+df_matches_count.MatchesAway

df_yellow_cards_teams = pd.merge(df_yellow_cards_count, df_matches_count)
df_yellow_cards_teams['AvgYellowPerMatch'] = df_yellow_cards_teams.YellowCardsTotal/df_yellow_cards_teams.MatchesTotal
#just to get team as index
df_yellow_cards_teams = df_yellow_cards_teams.groupby("Team").mean()
df_yellow_cards_teams


# We see that some teams with only very few matches appear in both lists. These could be statistical outliers.

# In[202]:


plt.figure(figsize=(12,6))

ax = plt.subplot(121)
df_yellow_cards_teams["AvgYellowPerMatch"].sort_values(ascending=False).nlargest(n=10).plot.bar()
plt.title("Teams with most yellow cards")

ax = plt.subplot(122)
df_yellow_cards_teams["AvgYellowPerMatch"].sort_values(ascending=True).nsmallest(n=10).plot.bar()
plt.title("Fairest Teams")

plt.show()


# Based on the previous observation we restrict ourselves to teams with at least 30 matches which are in total 21 teams. Germany is on rank 3.

# In[220]:


df_yellow_cards_reg_teams = df_yellow_cards_teams[df_yellow_cards_teams.MatchesTotal > 30]
print(len(df_yellow_cards_reg_teams))

plt.figure(figsize=(12,6))

df_yellow_cards_reg_teams["AvgYellowPerMatch"].sort_values(ascending=False).plot.bar()
plt.title("Teams with most yellow cards")

plt.show()


# ## Winners play fair!?
# 
# Whether or not a game is a tie does not have an effect on the amount of red or yellow cards. However, if it is not a tie the winner are given less yellow and red cards.

# In[9]:


avg_yellow_of_winner = len(df_events.loc[(df_events['EventOfWinner'] == True) & (df_events['EventType'] == 'Y')])/num_matches_decision
avg_yellow_of_loser = len(df_events.loc[(df_events['EventOfLoser'] == True) & (df_events['EventType'] == 'Y')])/num_matches_decision

avg_red_of_winner = len(df_events.loc[(df_events['EventOfWinner'] == True) & ((df_events['EventType'] == 'R') | (df_events['EventType'] == 'RSY'))])/num_matches_decision
avg_red_of_loser = len(df_events.loc[(df_events['EventOfLoser'] == True) & ((df_events['EventType'] == 'R') | (df_events['EventType'] == 'RSY'))])/num_matches_decision

avg_yellow_decided_match = avg_yellow_of_winner+avg_yellow_of_loser
avg_red_decided_match = avg_red_of_winner+avg_red_of_loser

avg_yellow_tie_match = len(df_events.loc[(df_events['HomeTeamWins'] == False) & (df_events['AwayTeamWins'] == False) & (df_events['EventType'] == 'Y')])/num_matches_tie
avg_red_tie_match = len(df_events.loc[(df_events['HomeTeamWins'] == False) & (df_events['AwayTeamWins'] == False) & ((df_events['EventType'] == 'R') | (df_events['EventType'] == 'RSY'))])/num_matches_tie

print("avg_yellow_of_winner: %.2f"% (avg_yellow_of_winner))
print("avg_yellow_of_loser: %.2f"% (avg_yellow_of_loser))
print("avg_yellow_decided_match: %.2f"% (avg_yellow_decided_match))
print("avg_yellow_tie_match: %.2f"% (avg_yellow_tie_match))

print("avg_red_of_winner: %.2f"% (avg_red_of_winner))
print("avg_red_of_loser: %.2f"% (avg_red_of_loser))
print("avg_red_decided_match: %.2f"% (avg_red_decided_match))
print("avg_red_tie_match: %.2f"% (avg_red_tie_match))

plt.figure(figsize=(12,6))

ind = np.arange(2)
width = 0.35
dist = 0.2

ax = plt.subplot(121)

yellow_cards = (avg_yellow_of_winner, avg_yellow_of_loser)
red_cards = (avg_red_of_winner, avg_red_of_loser)
plt.xticks(ind, ('Winner', 'Loser'))
ax.bar(ind, yellow_cards, width, color='y')
ax.bar(ind + width, red_cards, width, color='r')

ax = plt.subplot(122)

yellow_cards = (avg_yellow_decided_match, avg_yellow_tie_match)
red_cards = (avg_red_decided_match, avg_red_tie_match)
plt.xticks(ind, ('Decided', 'Tie'))
ax.bar(ind, yellow_cards, width, color='y')
ax.bar(ind + width, red_cards, width, color='r')

plt.show()


# # Predict Fouls
# In this section we trained a model to predict the amount of yellow cards given in a match.
# 
# ## Initial Features for Regression
# - Hour game starts, just added for fun. We do not expect a correlation. 
# - The year the match took place
# - The stage (group phase, quarter-finals etc.) We chose to perform a one hot encoding.
# - The total amount of goals
# - The goal difference
# - The goal difference in the half time
# - The change of these differences in the second part of the match
# - Whether there was extra time
# - Whether penalty decided the match
# - The amount of substitutions
# - The amount of substitutions at half time
#      
# ## Explanation of Event Types
# The codes for the event types are: G=Goal, OG=Own Goal, Y=Yellow Card, R=Red Card, SY = Red Card by second yellow, P=Penalty, MP=Missed Penalty, I = Substitution In, O=Substitute Out, IH= In half time?
# 

# First, we need some data preparation. Specifically, we build the columns:
# - One Hot Encoding for the EventType column (needed for groupby later on)
# - One Hot Encoding for the StageRank column (for the actual regression)
# - total goals scored
# - goal difference half time
# - goal difference end
# - delta of the last two values
# 
# Additionally, attendance must be a numeric data type
# 

# In[10]:


df_events_ohe = pd.concat([df_events, pd.get_dummies(df_events['EventType'])], axis=1)
df_events_ohe = pd.concat([df_events_ohe, pd.get_dummies(df_events['StageRank'], prefix="Stage")], axis=1)

df_events_ohe = df_events_ohe.assign(GoalsTotal = lambda x : x['Home Team Goals']+x['Away Team Goals'])
df_events_ohe = df_events_ohe.assign(GoalDifference = lambda x : abs(x['Home Team Goals']-x['Away Team Goals']))
df_events_ohe = df_events_ohe.assign(GoalDifferenceHalfTime = lambda x : abs(x['Half-time Home Goals']-x['Half-time Away Goals']))
df_events_ohe = df_events_ohe.assign(DeltaGoals = lambda x : x['GoalDifference']-x['GoalDifferenceHalfTime'])
df_events_ohe[['Attendance']] = df_events_ohe[['Attendance']].apply(pd.to_numeric)
df_events_ohe


# Perform a group by to get sum of yellow cards

# In[11]:


f = {'HourGameStart':['mean'],
     #'Home Team Goals':['mean'], # not symmetric -> throw out
     #'Away Team Goals':['mean'],
     #'Half-time Home Goals':['mean'],
     #'Half-time Away Goals':['mean'],
     'Year':['mean'],
     'Stage_1':['mean'],
     'Stage_2':['mean'],
     'Stage_3':['mean'],
     'Stage_4':['mean'],
     'Stage_5':['mean'],
     'Stage_6':['mean'],
     'GoalsTotal':['mean'],
     'GoalDifference':['mean'],
     'GoalDifferenceHalfTime':['mean'],
     'DeltaGoals':['mean'],
     'ExtraTime':['mean'],
     'Penalty':['mean'],
     'I':['sum'], #substitutions
     'IH':['sum'], #substitutions half time
     'Y':['sum'],
    }

df_events_grp = df_events_ohe.groupby(['MatchID']).agg(f)
df_events_grp.columns = df_events_grp.columns.get_level_values(0)
df_events_grp


# In[12]:


df_events_grp.columns


# We simply use MinMaxScaler as preprocessing

# In[13]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_events_grp[['HourGameStart','Year','GoalsTotal','GoalDifference','GoalDifferenceHalfTime','DeltaGoals']] = scaler.fit_transform(df_events_grp[['HourGameStart','Year','GoalsTotal','GoalDifference','GoalDifferenceHalfTime','DeltaGoals']])


# Linear regression is used to predict the amount of yellow cards for the test data. The MSE is fairly lower than the one of the baseline model. The coefficient of determination (R squared, amount of explained variance) is 0.5 indicating a moderate model performance. Keep in mind that R squared values depend on the amount of features used (strictly increasing on number of features).

# In[14]:


from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# train test split
# linear regression to predict y 
X = df_events_grp.loc[:, 'HourGameStart':'IH']
#X = pd.concat([df_fouls.loc[:, 'StageRank':'IH'], df_fouls.loc[:, 'ALG':'ZAI']], axis=1)
Y = df_events_grp.loc[:, 'Y']

# transform to numpy array
X = X.as_matrix().astype(np.float)
Y = Y.as_matrix().astype(np.float)

# train/ test split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=40)

# fit regression model
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

# compare RMSE and R value with base model
y_pred = regr.predict(X_test)

# base model would be the average y value
# needed for comparison
y_base_pred = np.zeros((len(y_pred)))
y_base_pred[:,] = y_train.mean()

print("LINEAR REGRESSION: Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
print('LINEAR REGRESSION: Variance score: %.2f' % r2_score(y_test, y_pred))

print("BASE: Mean squared error: %.2f"
      % mean_squared_error(y_test, y_base_pred))
# zero for sure, just added for completeness
print('BASE: Variance score: %.2f' % r2_score(y_test, y_base_pred))


# To statistically analyze the model we also compute several statistics. The most important ones are
# 
# - T values: statistic for t test checking whether coefficient is zero
# - p values: probability that coefficient is zero
# - F statistic: is the group of features significant?
# 
# We can derive that only x2, x14, x15, x16 are significant, i.e. Year, Penalty, I (=Amount of substitutions) and IH (=Amount of half time substitutions). 
# 
# All together F-statistic prob is low enough. Hence, all variables together can be considered significant.

# In[15]:


import statsmodels.api as sm
from scipy import stats

X2 = sm.add_constant(X_train)
est = sm.OLS(y_train, X2)
est2 = est.fit()
print(est2.summary())


# To confirm a correlation between yellow cards and year, penalty, substitutions and half time substitutions we compute the Pearson correlation coefficient. The p-value corresponds to the probability to observe the left correlation coefficient randomly. All correlations are significant.
# 
# The positive correlation with year suggests that nowadays more yellow cards are given. This might be due to stricter rules or less fair play.
# 
# Note that penalty does not include penalties due to red cards but only penalties after extra time to decide the game. A positive correlation could be due to the fact that such matches take longer and are very close games.
# 
# The correlation with substitutions is not obvious. 

# In[16]:


from scipy.stats.stats import pearsonr  

print(pearsonr(df_events_grp['Year'],Y)) # equivalent to: print(pearsonr(X[:,1],Y)) because not sensitive to scaling
print(pearsonr(df_events_grp['Penalty'],Y))
print(pearsonr(df_events_grp['I'],Y)) # substitutions
print(pearsonr(df_events_grp['IH'],Y)) # half time substitutions


# The correlation with year and substitutions can also be observed in a scatter plot. For penalty this does plot does not make sense as it is a binary decision. 

# In[17]:


import numpy as np
import matplotlib.pyplot as plt

plt.figure()
plt.plot(df_events_grp['Year'],Y, "o")
plt.figure()
plt.plot(df_events_grp['I'],Y, "o")


# We try several ways to increase the performance
# 
# 1. introduce regularization to tune our linear model (usually we need cross validation to tune the introduced hyperparameter values. However as we could not improve model performance we did not perform that step)
# 2. try a different ML model to increase accuracy
# 3. introduce team as one hot encoding feature to increase performance
# 
# But first, why not remove the unnecessary features and make the model more robust?

# In[18]:


X2 = df_events_grp.loc[:, ['Year', 'Penalty', 'IH', 'I']]
Y2 = df_events_grp.loc[:, 'Y']

# transform to numpy array
X2 = X2.as_matrix().astype(np.float)
Y2 = Y2.as_matrix().astype(np.float)

# train/ test split
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, Y2, test_size=0.33, random_state=40)

# fit regression model
regr = linear_model.LinearRegression()
regr.fit(X_train2, y_train2)

# compare RMSE and R value with base model
y_pred2 = regr.predict(X_test2)

# base model would be the average y value
# needed for comparison
y_base_pred = np.zeros((len(y_pred)))
y_base_pred[:,] = y_train.mean()

print("LINEAR REGRESSION: Mean squared error: %.2f"
      % mean_squared_error(y_test2, y_pred2))
print('LINEAR REGRESSION: Variance score: %.2f' % r2_score(y_test2, y_pred2))

print("BASE: Mean squared error: %.2f"
      % mean_squared_error(y_test2, y_base_pred))
# zero for sure, just added for completeness
print('BASE: Variance score: %.2f' % r2_score(y_test2, y_base_pred))


# ## Attempt 1: Regularization
# As mentioned above, regularization fails to improve the linear model. In this case we only tried ridge regression. One could also employ lasso regression etc. to regularize which have the effect of feature selection or lower variable scale respectively.

# In[19]:


from sklearn.linear_model import Ridge
    
clf = Ridge(alpha=2)
clf.fit(X_train, y_train) 

y_pred = clf.predict(X_test)

print("REGULARIZATION: Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
print('REGULARIZATION: Variance score: %.2f' % r2_score(y_test, y_pred))


# ## Attempt 2a: Try a decision tree
# Just another regression model. Leafes of the tree contain values for the specific subspace.

# In[20]:


from sklearn.tree import DecisionTreeRegressor

# fit regression model
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=4)
regr_1.fit(X_train, y_train)
regr_2.fit(X_train, y_train)

# compare RMSE and R value with base model
y_pred = regr_1.predict(X_test)
y_pred_2 = regr_2.predict(X_test)

# base model would be the average y value
# needed for comparison
y_base_pred = np.zeros((len(y_pred)))
y_base_pred[:,] = y_train.mean()

print("DECISION TREE 1: Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
print('DECISION TREE 1: Variance score: %.2f' % r2_score(y_test, y_pred))

print("DECISION TREE 2: Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred_2))
print('DECISION TREE 2: Variance score: %.2f' % r2_score(y_test, y_pred_2))


# ## Attempt 2b: Neural Network
# 
# Sounds fancy but actually just a vanilla multilayer perceptron with only two layers.

# In[21]:


from sklearn.neural_network import MLPRegressor

nn = MLPRegressor(
    hidden_layer_sizes=(1,),  activation='relu', solver='adam', alpha=0.001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
    random_state=9, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

nn.fit(X_train, y_train)

y_pred = nn.predict(X_test)

print("NEURAL NETWORK: Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
print('NEURAL NETWORK: Variance score: %.2f' % r2_score(y_test, y_pred))


# ## Attempt 3: Add team as one hot encoding
# We hot encode the teams of the specific match and hope to increase the accuracy

# In[22]:


df_teams_ohe = (pd.get_dummies(df_events['Home Team Initials'])+pd.get_dummies(df_events['Away Team Initials'])).fillna(value=0)
df_teams_ohe = pd.concat([df_teams_ohe,df_events['MatchID']],axis=1).groupby('MatchID').mean()
df_fouls = df_events_grp.join(df_teams_ohe)
df_fouls = df_fouls.reset_index()
df_fouls.drop(['MatchID'], 1,inplace=True)
df_fouls


# In[23]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_fouls[['HourGameStart','Year','GoalsTotal','GoalDifference','GoalDifferenceHalfTime','DeltaGoals']] = scaler.fit_transform(df_fouls[['HourGameStart','Year','GoalsTotal','GoalDifference','GoalDifferenceHalfTime','DeltaGoals']])


# Unfortunately, adding the team did not improve the model performance. Probably due to shortage of data.

# In[24]:


from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# train test split
# linear regression to predict y 
X = pd.concat([df_fouls.loc[:, 'HourGameStart':'IH'], df_fouls.loc[:, 'ALG':'ZAI']], axis=1)
Y = df_events_grp.loc[:, 'Y']

# transform to numpy array
X = X.as_matrix().astype(np.float)
Y = Y.as_matrix().astype(np.float)

# train/ test split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=40)

# base model would be the average y value
# needed for comparison
y_base_pred = np.zeros((len(y_pred)))
y_base_pred[:,] = y_train.mean()

# fit regression model
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

# compare RMSE and R value with base model
y_pred = regr.predict(X_test)
print("LINEAR REGRESSION: Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
print('LINEAR REGRESSION: Variance score: %.2f' % r2_score(y_test, y_pred))

print("BASE: Mean squared error: %.2f"
      % mean_squared_error(y_test, y_base_pred))
# zero for sure, just added for completeness
print('BASE: Variance score: %.2f' % r2_score(y_test, y_base_pred))

