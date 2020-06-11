#This is a small project that uses K-means clustering on a basketball data
#from the 2018-2019 and 2019-2020 Southside Christian School basketball teams.

import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

#print('hello')


#reads the file
scs_stats = pd.read_csv('averages_2018_2020.csv')


averages = scs_stats.mean()
#creats a pairplot for field goal attempts, assists, and rebounds
sns.pairplot(scs_stats[['FGA', 'AST', 'REB']])
plt.show()

plt.clf()

#creates a heat map for the correlation between field goal attempts, 
#assists, and rebounds
correlation = scs_stats[['FGA', 'AST', 'REB']].corr()
sns.heatmap(correlation, annot = True)
plt.show()

plt.clf()

#setting up and training the kmeans model using the stats
kmeans_model = KMeans(n_clusters = 6, random_state = 1)
#good_columns = scs_stats._get_numeric_data().dropna(axis = 1)
df = pd.DataFrame(scs_stats[['PPG', 'FGM', 'FT%', 'AST', 'REB', 'STL', 'BLK']])
good_columns = np.array(df)
#print(good_columns)


kmeans_model.fit(good_columns)
labels = kmeans_model.labels_



dfprint = pd.DataFrame(scs_stats[['Athlete', 'PPG', 'FGM', 'FT%', 'AST', 'REB', 'STL', 'BLK']])
dfprint['Cluster'] = labels
dfprint.sort_values('Cluster', inplace = True)
print(dfprint)

#player_names = scs_stats['Athlete']
#fga = scs_stats['FGA']
#ast = scs_stats['AST']
#reb = scs_stats['REB']


#players_and_labels = zip(player_names, fga, ast, reb, labels)
#players_and_labels_sorted = sorted(players_and_labels, key = lambda x : x[4])
#for player in players_and_labels_sorted:
#	print(player)


kmeans_model_points_rebounds = KMeans(n_clusters = 2, random_state = 1)
pr = pd.DataFrame(scs_stats[['PPG', 'REB']])
pr_columns = np.array(pr)

kmeans_model_points_rebounds.fit(pr_columns)
pr_labels = kmeans_model_points_rebounds.labels_

prprint = pd.DataFrame(scs_stats[['Athlete', 'PPG', 'REB']])
prprint['Cluster'] = pr_labels
prprint.sort_values('Cluster', inplace = True)
print(prprint)

plt.grid(linestyle = '--')


plt.scatter(pr_columns[pr_labels==0, 0], pr_columns[pr_labels==0, 1], s=100, c='red', label ='Cluster 0')
plt.scatter(pr_columns[pr_labels==1, 0], pr_columns[pr_labels==1, 1], s=100, c='blue', label ='Cluster 1')
plt.scatter(pr_columns[pr_labels==2, 0], pr_columns[pr_labels==2, 1], s=100, c='green', label ='Cluster 2')
plt.scatter(pr_columns[pr_labels==3, 0], pr_columns[pr_labels==3, 1], s=100, c='cyan', label ='Cluster 3')

plt.scatter(kmeans_model_points_rebounds.cluster_centers_[:, 0], kmeans_model_points_rebounds.cluster_centers_[:, 1], s=50, c='yellow', label = 'Centroids')
plt.title('Clusters of Players')
plt.xlabel('Points Per Game')
plt.ylabel('Rebounds Per Game')
plt.legend()

plt.show()

