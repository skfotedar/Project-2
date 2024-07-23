import pandas as pd
import datetime
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
import sys
import numpy as np

#pull data from the file
#this file lists all NBA games going back to 1946
file_name = "./csv/game.csv"
output_file = "./output/reg_season.txt"
sys.stdout = open(output_file, "w")
df = pd.read_csv(file_name, index_col='game_id')

#only use games from 2004 to present
#2004 is the first year the NBA is in its current form
cut_off = datetime.datetime(2004,9,1)
df['game_date'] = pd.to_datetime(df['game_date'])
df = df[df['game_date'] > cut_off]

#remove all-star games and pre-season games
df = df[df['season_type'] != 'Pre Season']
df = df[df['season_type'] != 'All-Star']
df = df[df['season_type'] != 'All Star']

#calculate and add the assist to turnover ratio
#this is a metric that captures how well a team takes care of possesions
df = df.assign(a2to_home = lambda x: (x['ast_home'] / x['tov_home']))
df = df.assign(a2to_away = lambda x: (x['ast_away'] / x['tov_away']))

#calculate and add true shooting percentage
#this combines field goal, 3 point shooting and free throws into a single metric
df = df.assign(tsp_home = lambda x: (2*x['fgm_home']+3*x['fg3m_home']+x['ftm_home']) /
            (2*(x['fga_home'] + x['fg3a_home'] + 0.44*x['fta_home'])))
df = df.assign(tsp_away = lambda x: (2*x['fgm_away']+3*x['fg3m_away']+x['ftm_away']) /
            (2*(x['fga_away'] + x['fg3a_away'] + 0.44*x['fta_away'])))

#calculate and add rebound margin
df = df.assign(rebound_margin_home = lambda x: x['reb_home'] - x['reb_away'])
df = df.assign(rebound_margin_away = lambda x: x['reb_away'] - x['reb_home'])

#set W L to 1s and 0s for Logistic Regression
df['wl_home'] = df['wl_home'].astype(str)
for i in df.index:
    if df.loc[i,'wl_home'] == 'W':
        df.loc[i,'wl_home'] = 1
    else:
        df.loc[i,'wl_home'] = 0

#data cleaning
#remove all unecessary data
df = df.drop(columns=['season_id', 'team_id_home', 'team_abbreviation_home', 'team_name_home',
                 'game_date', 'matchup_home', 'video_available_home', 'reb_home', 'reb_away',
                'team_id_away', 'team_abbreviation_away', 'team_name_away', 'matchup_away',
                'wl_away', 'video_available_away', 'min', 'fgm_home', 'fga_home', 'fg3m_home',
                'fg3a_home', 'oreb_home', 'dreb_home', 'ast_home', 'fg_pct_home', 'fg_pct_away',
                'stl_home', 'blk_home', 'tov_home', 'pf_home', 'fgm_away', 'fga_away', 'ftm_home',
                'fg3m_away', 'fg3a_away', 'reb_away', 'ast_away', 'fg3_pct_home', 'fg3_pct_away',
                'stl_away', 'blk_away', 'tov_away', 'pf_away', 'ft_pct_home', 'ft_pct_away', 'fta_home',
                'pts_home', 'pts_away', 'ftm_away', 'fta_away', 'oreb_away', 'dreb_away'])

#create 2 data frames, one for in-season data, the other for playoffs
df_season = df[df['season_type'] == 'Regular Season']
df_season = df_season.drop(columns=['season_type'])
df_season.info()

#Linear Regression - Predict the scoring margin of the home team based on
# true shooting percentage, assist to turnover ration, rebound margin
X = df_season[['tsp_home', 'tsp_away', 'a2to_home', 'a2to_away', 'rebound_margin_home']]
y = df_season[['plus_minus_home']]

reg = linear_model.LinearRegression()
reg.fit(X,y)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = linear_model.LinearRegression()
model.fit(X_train,y_train)

predicted = model.predict(X_test)
mse = mean_squared_error(y_test, predicted)
r2 = r2_score(y_test, predicted)

#print out the formula from the linear regression
formula_df = pd.DataFrame()
formula_df['variables'] = X.columns.tolist()
coef_list = model.coef_[0]
formula_df['coefficients'] = coef_list
print(f"\n***Results from Linear Regression***\n")
print(formula_df)
print(f"\ny-intercept: {model.intercept_[0]}")
print(f"\nmean squared error (MSE): {mse}")
print(f"R-squared (R2): {r2}")

# Call the `score()` method on the model to show the R2 score
model.score(X_test, y_test)
print(f"Model Score: {model.score(X_test, y_test)}")

#Now do it with a Logistic Regression
#The Logistic Regression attemps calculate what drives the chances of the home team winning
#This does analsyis is not concerned with scoring margin
logistic_regression_model = linear_model.LogisticRegression(random_state=25, max_iter=1000)
y = df_season['wl_home'].astype(int)
X = df_season[['tsp_home', 'tsp_away', 'a2to_home', 'a2to_away', 'rebound_margin_home']]
X_train, X_test, y_train, y_test = train_test_split(X, y)
lr_model = logistic_regression_model.fit(X_train, y_train)

# Print Out Results from the Logistic Regression
print(f"\n***Results from Logistic Regression***")
print(f"\nTraining Data Score: {lr_model.score(X_train, y_train)}")
print(f"Testing Data Score: {lr_model.score(X_test, y_test)}")
testing_predictions = lr_model.predict(X_test)
print(f"Accuracy Score: {accuracy_score(y_test, testing_predictions)}")
print("Coefficients:", lr_model.coef_[0])
print("Intercept:", lr_model.intercept_[0])

# Make predictions on the test data
predictions = logistic_regression_model.predict(X)
# Create a confusion matrix
print(confusion_matrix(y, predictions, labels = [1,0]))
print(classification_report(y, predictions, labels = [1, 0]))
print(balanced_accuracy_score(y, predictions))

#do the analysis on playoff data?
#Darryl Morey "Modified Pytagorean Theorem"
#separate files for k_means, pca, playoff and in-season
#show how you checked for null values and content in the data
#proper citation

#Random Forest Classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

#scale the data
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Random Forest model
clf = RandomForestClassifier(random_state=1, n_estimators=500).fit(X_train_scaled, y_train)

# Evaluate the model
print(f'Training Score: {clf.score(X_train_scaled, y_train)}')
print(f'Testing Score: {clf.score(X_test_scaled, y_test)}')

# Get the feature importance array
feature_importances = clf.feature_importances_
# List the top 10 most important features
importances_sorted = sorted(zip(feature_importances, X.columns), reverse=True)
print("\n")
print(f"{importances_sorted[:10]}")

# Plot the feature importances
features = sorted(zip(X.columns, feature_importances), key = lambda x: x[1])
cols = [f[0] for f in features]
width = [f[1] for f in features]

fig, ax = plt.subplots()

fig.set_size_inches(8,6)
plt.margins(y=0.001)

ax.barh(y=cols, width=width)

plt.show()
print(df_season['wl_home'].value_counts())