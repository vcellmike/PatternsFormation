# %%
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

print("All dependencies present")

# %%
#load in dataframe
pd.set_option('display.max_columns', None)
seed_val= 42
file_in = "FINAL2/data/unclassified_features_39517"

with open(file_in+'.pkl', 'rb') as f:
  feats_df = pickle.load(f) # deserialize using load()

print(feats_df.shape)
feats_df.head()

# %%
# PCA on feat arrays

# scaling data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

feat_array_scaled = pd.DataFrame(sc.fit_transform(feats_df[feats_df.columns[13:-3]])) # 14th column - third from last (exclusive)
print(feat_array_scaled.shape)

# Applying PCA function on training and testing set of X component
from sklearn.decomposition import PCA

pca = PCA(n_components = 2)
pca_data = pca.fit_transform(feat_array_scaled)

explained_variance = pca.explained_variance_ratio_
print(explained_variance)

feats_df["pc1"] = pca_data[:,0]
feats_df["pc2"] = pca_data[:,1]

# PCA components (loadings)
loadings = pd.DataFrame(pca.components_.T,
                        columns=[f'PC{i+1}' for i in range(len(pca.components_))],
                        index=feat_array_scaled.columns)
print("PCA Loadings:")
print(loadings)

# Find top features for each principal component
top_features = {}
for pc in loadings.columns:
    top_features[pc] = loadings[pc].abs().nlargest(3).index.tolist()  # Top 3 contributing features
print("\nTop Contributing Features per Principal Component:")
print(top_features)

print("PC1:")
for feat in top_features["PC1"]:
    print(f"----{feats_df.columns[feat + 13]}")
print("PC2A:")
for feat in top_features["PC2"]:
    print(f"----{feats_df.columns[feat + 13]}")

# %%
# K_means clustering
from sklearn.cluster import KMeans

#Find optimum number of clusters
sse = [] #SUM OF SQUARED ERROR
for k in range(1,17):
    km = KMeans(n_clusters=k, random_state=seed_val)
    km.fit(feat_array_scaled)
    sse.append(km.inertia_)

fig, ax = plt.subplots()
ax.set(xlabel = "Number of Clusters", ylabel = "SSE", xticks = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])

ax.plot(sse)

plt.show()

# %%
num_clusters = 7
file_out = file_in + "_KM_classes_" + str(num_clusters) + "_s" + str(seed_val) + ".pkl"

# %%
# K_means clustering
from sklearn.cluster import KMeans

cluster_dicts = {}

for c in range(num_clusters):
    print(c)
    working_df = feats_df
    km = KMeans(n_clusters = num_clusters, init='k-means++', n_init=50, max_iter=300,
                tol=0.0001, random_state=seed_val)
    clusters = km.fit(feat_array_scaled)
    working_df["cluster"] = clusters.labels_
    # create entry in dictionary
    cluster_dicts[str(c)] = working_df

# %%
clusters = np.unique(np.array(feats_df["cluster"])) # list of unique clusters
cluster_dfs = [] # list of dfs for each cluster

for i in clusters:
  df = feats_df[feats_df["cluster"] == i]
  df.reset_index(inplace = True, drop = True)
  cluster_dfs.append(df)

# plot the data
fig, ax = plt.subplots()

colors = ["red","green","blue","purple","orange","black","pink", "yellow", "brown", 'gray', 'cyan']

for j in clusters:
  df = cluster_dfs[j]
  ax.scatter(df["pc1"], df["pc2"], c = colors[j], label = "cluster " + str(j))

leg = ax.legend(loc="lower left")
ax.set(xlabel = "PC1", ylabel = "PC2", title = "Clustering of Intuitive Features")

option_print = True
if option_print:
  plt.savefig("FINAL2/output_images/KMeansClustering_of_Intuitive_Features_cl7.png", dpi = 300)

plt.show()

feats_df.head()

# %%
# add pc1, pc2, cluster columns
feats_df.to_pickle(file_out)
print(feats_df.shape)

# %%
# print image or random images for a given column in the dataframe
def show_image(image_num = None, num_random_images = None, df = feats_df):
  if image_num is not None:
    image_path = "FINAL2" + df["dir"][image_num] + df["path"][image_num]
    image = Image.open(str(image_path))

    plt.figure()
    plt.imshow(image, cmap = "YlOrRd", vmin = 0, vmax = 255)
    plt.show()
    print(image_path)

  if num_random_images is not None:
    random_indices = np.random.randint(0, df.shape[0], num_random_images)
    print(random_indices)
    for i in random_indices:
      image_path = "FINAL2"+ df["dir"][i] + df["path"][i]
      image = Image.open(str(image_path))

      # plt.figure()
      # plt.imshow(image, cmap = "YlOrRd", vmin = 0, vmax = 255)
      # plt.show()
      # print(image_path)

def col_to_print_func(col_to_print = None, num_random_images = None, image_num = None):
  groups = feats_df[col_to_print].unique()
  groups.sort()
  groups = groups.tolist()

  for group in groups:
    print(f"Images from {col_to_print} = {group}")
    cluster_df = feats_df[feats_df[col_to_print] == group].reset_index(drop=True)
    show_image(num_random_images = num_random_images, image_num = image_num, df=cluster_df)

col_to_print_func(col_to_print = "cluster", num_random_images=3)

# %%
num_imgs = 20

fig, ax = plt.subplots(num_imgs, len(cluster_dfs))

for i in range(num_imgs):
  for j in range(len(cluster_dfs)):
    df = cluster_dfs[j]
    num = np.random.randint(low = 0, high= df.shape[0], size=1, dtype=int)[0]
    image_path = "FINAL2" + df["dir"][num] + df["path"][num]
    img = np.array(Image.open(image_path))
    ax[i][j].imshow(img, cmap = "YlOrRd", vmin = 0, vmax = 255)
    ax[i][j].set_title(df["path"][num] + " cluster " + str(j) + " index " + str(num), fontsize = 8)
    ax[i][j].get_xaxis().set_visible(False)
    ax[i][j].get_yaxis().set_visible(False)

fig.set_figwidth(5*len(cluster_dfs))
fig.set_figheight(5 * num_imgs)

option_print = True
if option_print:
  plt.savefig("FINAL2/output_images/KMeansClustering_ImagesPerCluster_cl7_.png", dpi = 300)

plt.show()
