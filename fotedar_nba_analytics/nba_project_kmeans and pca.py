import pandas as pd
import datetime
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import sys
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np

#pull data from the file
#this file lists all NBA games going back to 1946
file_name = "./csv/game.csv"
df = pd.read_csv(file_name, index_col='game_id')
output_file = "./output/pca_analysis.txt"
sys.stdout = open(output_file, "w")

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

#kmeans analysis
scaler = StandardScaler()
season_wl_list = df_season.index.values.tolist()
normalized_data = scaler.fit_transform(df_season)
normalized_df = pd.DataFrame(normalized_data, columns = df_season.columns)
normalized_df['game_id'] = season_wl_list
normalized_df = normalized_df.set_index('game_id')

k_values = []
for i in range (1,50):
    k_values.append(i)

inertias = []

for k in k_values:
    model = KMeans(n_clusters=k, n_init='auto', random_state=1)
    model.fit(normalized_df)
    inertias.append(model.inertia_)

dictionary = {'x_values':k_values, 'y_values': inertias}
elbow_curve_df = pd.DataFrame(dictionary)

# Plot a line chart with all the inertia values computed with
# the different values of k to visually identify the optimal value for k.
plt.plot(elbow_curve_df['x_values'], elbow_curve_df['y_values'], 'o-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Inertia Values vs. Number of Clusters')
plt.grid(True)
plt.show()

#kmeans clustering
# I tried different ranges for the clustering, 25 seems like a good amount
model = KMeans(n_clusters = 25)
model.fit(normalized_df)
#model.labels_

# Create a copy of the DataFrame
copy_normalized_df = normalized_df

# Add a new column to the DataFrame with the predicted clusters
copy_normalized_df['kmeans_25'] = model.labels_

# Create a scatter plot using Pandas plot by setting
# x = True Shooting Percentage of the Home Team
# y = Assist - to - Turnover for Home Team
# Use "rainbow" for the color to better visualize the data.
plt.scatter(x=copy_normalized_df['tsp_home'],
            y=copy_normalized_df['a2to_home'],
            c=copy_normalized_df['kmeans_25'],
            cmap='rainbow')
plt.xlabel('True Shooting Percentage Home')
plt.ylabel('Assist to Turnover Home')
plt.title('Scatter Plot')
plt.show()

# x = True Shooting Percentage of the Home Team
# y = Rebound Margin of the Home Team
# Use "rainbow" for the color to better visualize the data.
plt.scatter(x=copy_normalized_df['tsp_home'],
            y=copy_normalized_df['rebound_margin_home'],
            c=copy_normalized_df['kmeans_25'],
            cmap='rainbow')
plt.xlabel('True Shooting Percentage Home')
plt.ylabel('Rebound Margin Home')
plt.title('Scatter Plot')
plt.show()

# x = True Shooting Percentage of the Home Team
# y = Rebound Margin of the Home Team
# Use "rainbow" for the color to better visualize the data.
plt.scatter(x=copy_normalized_df['tsp_home'],
            y=copy_normalized_df['tsp_away'],
            c=copy_normalized_df['kmeans_25'],
            cmap='rainbow')
plt.xlabel('True Shooting Percentage Home')
plt.ylabel('True Shooting Percentage Away')
plt.title('Scatter Plot')
plt.show()

#pca analysis
pca = PCA(n_components=6)

pca_list = []
#use lambda for this
for i in range(1,7):
    x = 'PCA'+str(i)
    pca_list.append(x)

normalized_df =normalized_df.drop(columns = ['kmeans_25'])

games = pca.fit_transform(normalized_data)
games_df = pd.DataFrame(games,columns = pca_list)

# Use the columns from the original scaled DataFrame as the index.
print(pca_list)
print(normalized_df.columns)

pca_component_weights = pd.DataFrame(pca.components_.T, columns=pca_list, index=normalized_df.columns)
print(pca_component_weights)



