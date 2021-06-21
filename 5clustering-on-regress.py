
# Create cluster df
clustercoef = res3.params[9:]
clusterp = res3.pvalues[9:]
price = pd.concat([clustercoef[:29],clusterp[:29]], axis=1)
distri = pd.concat([clustercoef[30:60],clusterp[30:60]], axis=1)
volume = pd.concat([clustercoef[60:],clusterp[60:]], axis=1)
df_reg1.info()

psplit = price.index.str.split('[').str[1].str.split(']').str[0]
psplit = psplit.values
df_price = pd.DataFrame(price)
df_price = df_price.set_index(psplit)

dsplit = distri.index.str.split('[').str[1].str.split(']').str[0]
dsplit = dsplit.values
df_distri = pd.DataFrame(distri)
df_distri = df_distri.set_index(dsplit)

vsplit = volume.index.str.split('[').str[1].str.split(']').str[0]
vsplit = vsplit.values
df_volume = pd.DataFrame(volume)
df_volume = df_volume.set_index(vsplit)

df_cluster = df_price.merge(df_distri, how='outer', left_on=df_price.index, right_on=df_distri.index).merge(df_volume, left_on='key_0', right_on=df_volume.index)
df_cluster = pd.DataFrame(df_cluster)
df_cluster.rename(columns={'key_0':'study','0_x':'price','1_x':'price-sig','0_y':'distri','1_y':'distri-sig',0:'volume',1:'volume-sig'}, inplace=True)
df_cluster.set_index('study', inplace=True)

df_cluster_backup = df_cluster
df_cluster = df_cluster_backup

# save cluster df
pkl_df_cluster = "df_cluster.pkl"
with open(pkl_df_cluster, 'wb') as file:
    pickle.dump(df_cluster, file)
with open("df_cluster.pkl", 'rb') as file:
    df_cluster2 = pickle.load(file)


scaler = MinMaxScaler()
df_cluster['pricemm'] = scaler.fit_transform(df_cluster[['price']])
df_cluster['pricesigmm'] = scaler.fit_transform(df_cluster[['price-sig']])
df_cluster['volumemm'] = scaler.fit_transform(df_cluster[['volume']])
df_cluster['volumesigmm'] = scaler.fit_transform(df_cluster[['volume-sig']])
df_cluster['distrimm'] = scaler.fit_transform(df_cluster[['distri']])
df_cluster['distrsigmm'] = scaler.fit_transform(df_cluster[['distri-sig']])

df_cluster.index = df_cluster.index.astype(float)
df_cluster = df_cluster.merge(df_reg1study.frequency, how='outer', left_on=df_cluster.index, right_on=df_reg1study.index)
df_cluster = df_cluster.set_index('key_0')
df_cluster = df_cluster.dropna()


df_cluster.to_csv('df_cluster.csv')
df_cluster = pd.read_csv('df_cluster.csv')

plt.scatter(df_cluster['volume'], df_cluster['distri'])
plt.show()



#freqdumclust = pd.get_dummies(df_cluster['frequency'])
#df_cluster = pd.concat([df_cluster, freqdumclust[[1,2,3]]], axis=1)
#df_cluster = df_cluster.drop(columns=['frequency'])

#cluster_X = df_cluster.iloc[:,6:]
#cluster_x = cluster_X.dropna()
#zsigni_X = df_cluster.iloc[:,[6,8,10,12,13,14]]
#zsigni_X = zsigni_X.dropna()

cluster_X = df_cluster.iloc[:,6:12]
#cluster_x = cluster_X.dropna()
zsigni_X = df_cluster.iloc[:,[6,8,10]]

cluster3 = df_cluster.iloc[:,:6]
from yellowbrick.cluster import KElbowVisualizer

# KMEANS
cluster_X = cluster_X.dropna()
from sklearn.cluster import KMeans

distortions = []
for i in range(1, 20):
    km = KMeans(n_clusters=i, 
                random_state=0,
                n_init=50)
    km.fit(cluster_X)
    distortions.append(km.inertia_)
    
plt.plot(range(1, 20), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()

model= KMeans()
visualizer = KElbowVisualizer(model, k=(2,30), timings= True)
visualizer.fit(zsigni_X)        # Fit data to visualizer
visualizer.show()  
kmeans.n_init
kmeans = KMeans(n_clusters=10, random_state=0)
labels = pd.DataFrame(kmeans.fit_predict(cluster_X))
study = pd.DataFrame(cluster_X.index.values)
kmeansout = pd.concat([study,labels], axis=1)
kmeansout = kmeansout.set_index(kmeansout.iloc[:,0])

kmeans = KMeans(n_clusters=7, random_state=0)
labels = pd.DataFrame(kmeans.fit_predict(zsigni_X))
study = pd.DataFrame(df_cluster.index.values)
kmeansoutzsig = pd.concat([study,labels], axis=1)
kmeansoutzsig = kmeansoutzsig.set_index(kmeansoutzsig.iloc[:,0])

# KMEDIODS
from sklearn_extra.cluster import KMedoids
distortions = []
for i in range(1, 20):
    km = KMedoids(n_clusters=i, 
                random_state=0)
    km.fit(cluster_X)
    distortions.append(km.inertia_)
    
plt.plot(range(1, 20), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()

model = KMedoids()
# k is range of number of clusters.
visualizer = KElbowVisualizer(model, k=(2,30), timings= True)
visualizer.fit(zsigni_X)        # Fit data to visualizer
visualizer.show()   

kmeds = KMedoids(n_clusters=4, random_state=0)
labelsmed = pd.DataFrame(kmeds.fit_predict(zsigni_X))
studymed = pd.DataFrame(zsigni_X.index.values)
kmedsoutzsig = pd.concat([studymed,labelsmed], axis=1)
kmedsoutzsig = kmedsoutzsig.set_index(kmedsoutzsig.iloc[:,0])

kmeds = KMedoids(n_clusters=4, random_state=0)
labelsmed = pd.DataFrame(kmeds.fit_predict(cluster_X))
studymed = pd.DataFrame(cluster_X.index.values)
kmedsout = pd.concat([studymed,labelsmed], axis=1)
kmedsout = kmedsout.set_index(kmedsout.iloc[:,0])

means = pd.DataFrame(kmeansout[0].values, index=kmeansout.index)
means =means.drop(columns=[0])
meds = pd.DataFrame(kmedsout[0].values, index=kmedsout.index)
meds = meds.drop(columns=[0])
zmeans = pd.DataFrame(kmeansoutzsig[0].values, index=kmeansoutzsig.index)
zmeans =zmeans.drop(columns=[0])
zmeds = pd.DataFrame(kmedsoutzsig[0].values, index=kmedsoutzsig.index)
zmeds = zmeds.drop(columns=[0])

doneclusterw = means.merge(meds, how='outer', left_on=means.index, right_on=meds.index)
doneclusterz = zmeans.merge(zmeds, how='outer', left_on=zmeans.index, right_on=zmeds.index)
donecluster = doneclusterw.merge(doneclusterz, how='outer', left_on=doneclusterw.index, right_on=doneclusterz.index)

donecluster = donecluster.drop(columns=['key_0','key_0_y'])
donecluster.rename(columns={'key_0_x':'study','1_x_x':'KMeans+sig','1_y_x':'KMedoids+sig','1_x_y':'KMeans-sig','1_y_y':'KMedoids-sig'}, inplace=True)
donecluster = donecluster.set_index('study')

#donecluster = donecluster.merge(cluster3, how='outer', left_on=donecluster.index, right_on=cluster3.index)
#donecluster.sort_values(by='KMeans')



#get centroids
meanscentroids = kmeans.cluster_centers_
medscentroids = kmeds.cluster_centers_
zmeanscentroids = kmeanszsig.cluster_centers_
zmedscentroids = kmedszsig.cluster_centers_

centroids2 = pd.DataFrame(centroids)
plotdf = cluster_X.merge(donecluster, how='outer',left_on=cluster_X.index, right_on=donecluster.index)
plotdf = plotdf.set_index('key_0')

df2  =cluster3.merge(donecluster, how='outer',left_on=cluster3.index, right_on=donecluster.index)
df2 = df2.set_index('key_0')
df2.sort_values(by=['KMeans+sig','KMedoids+sig'])
df2 = df2.drop(columns=['KMeans-sig','KMedoids-sig'])
df2 = df2.round(3)
df2 = df2.sort_values(by=['KMeans+sig','KMedoids+sig'])
print(tabulate(df2, tablefmt='latex_booktabs',headers=df2.columns))
df2.sort_values(by=['KMedoids+sig'])

df2  =cluster3.merge(means, how='outer',left_on=cluster3.index, right_on=means.index)
df2 = df2.set_index('key_0')
df2.rename(columns={1:'KMeans'}, inplace=True)
df2.sort_values(by=['KMeans','price','distri','volume'], inplace=True)
df2 = df2.round(3)

print(tabulate(df2, tablefmt='latex_booktabs',headers=df2.columns))
df2.sort_values(by=['KMedoids+sig'])

# VISUALIZE CLUSTERS
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

x= plotdf.iloc[:,:6]
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
principalDf = principalDf.set_index(plotdf.index)
#finalDf = pd.concat([principalDf, plotdf.iloc[:,6]], axis = 1)
finalDf = principalDf.merge(plotdf.iloc[:,6], how='outer',left_on=principalDf.index,right_on=plotdf.index)

finalDf['KMeans+sig'].round(0)


fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['KMeans+sig'] == target
    scatter = ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50,
               label= 'Cluster '+ str(int(target+1)))
    scatter2 = ax.scatter(finalDf.loc[finalDf['KMeans+sig']==target].mean()[[1]]
               , finalDf.loc[finalDf['KMeans+sig']==target].mean()[[2]]
               , c = color
               , s = 50*4
               , marker="^",
               label = None)
leg1= ax.legend(scatter.legend_elements(),labels=['Observations','Cluster Means'],loc="lower right", frameon=True,prop={'size': 14})
leg2 = ax.legend(loc='upper right', frameon=True,prop={'size': 14})
ax.add_artist(leg1)
ax.add_artist(leg2)
leg1.get_frame().set_edgecolor('black')
leg2.get_frame().set_edgecolor('black')
leg1.legendHandles[0]._sizes = [50]
leg1.legendHandles[1]._sizes = [50]
ax.tick_params(axis='x', labelsize=13)
ax.tick_params(axis='y', labelsize=13)
ax.grid()
plt.show()


pca.explained_variance_ratio_ #small, only 0.33+.28=.61


# Try hierarchical
cluster_X = cluster_X.dropna()
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


X = cluster_X

# setting distance_threshold=0 ensures we compute the full tree.
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

model = model.fit(X)
plt.title('Hierarchical Clustering Dendrogram')
# plot the top three levels of the dendrogram
plot_dendrogram(model, truncate_mode='level', p=3)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()


7
mod = AgglomerativeClustering(n_clusters=7)
mod.fit(X)







prodcat = ['parfume',
'parfume',
'cigarettes',
'Sausage snacks',
'cigarettes',
'cigarettes',
'cigarettes',
'cigarettes',
'butter',
'butter',
'chips',
'batteries',
'cigarettes',
'drinkyoghurt',
'yoghurt',
'butter',
'batteries',
'butter',
'butter',
'cigarettes',
'cigarettes',
'batteries',
'batteries',
'batteries',
'energydrinks',
'wasmiddel',
'wasmiddel',
'vitaminensupplementen',
'shampoos']

cluster3['prodcat'] = prodcat
decisiondf = cluster3
decisiondf['prodcat'] = decisiondf['prodcat'].astype('category')
