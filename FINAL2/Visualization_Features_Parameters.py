# %%
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys

print("Done.")

# %%
option_print = True 

params = ["Ua","Ui","Ga","Gi","Da","Di","Ba"] # feature parameters to use
colors = ["red","green","blue","purple","orange","black","pink"] # colors for each class
num_PCA_components = 2 # number of PCA components to use

pd.set_option('display.max_columns', None)

# %%
file_in = "FINAL2/data/unclassified_features_39517_KM_classes_7_s42"

with open(file_in + ".pkl", 'rb') as f:
  feats_df_large = pickle.load(f) # deserialize using load()

print("\n", file_in, " shape: ", feats_df_large.shape)

# these unclassified images are from the proper parameter range
file_in_narrow = "FINAL2/data/unclassified_features_33770"

with open(file_in_narrow + ".pkl", 'rb') as f:
  df_narrow = pickle.load(f)

print("\n", file_in_narrow, " shape: ", df_narrow.shape)

# %%
# Create output directories
output_dir = "FINAL2/data/output_images/"
os.makedirs(output_dir, exist_ok=True)

output_sub_dir = output_dir + file_in[11:] + "/"
os.makedirs(output_sub_dir, exist_ok=True)

specific_file_dir = output_sub_dir + "Visualization_Features_Parameters/"
os.makedirs(specific_file_dir, exist_ok=True)

# save output to txt file
sys.stdout = open(specific_file_dir +'Visualization_Features_Parameters_output.txt', 'w')  # redirect print() output
sys.stderr = sys.stdout

# %%
feats_df = feats_df_large[feats_df_large["path"].isin(df_narrow["path"].unique())]
feats_df.reset_index(inplace = True, drop = True)

print("\nFiltered feats_df with class and narrow bounds: ", feats_df.shape)
feats_df.head()

# %%
# number of classes in clustering
num_classes = len(feats_df["cluster"].unique()) 
print("number of classes: ", num_classes)

# %%
classes = np.arange(0,num_classes)# list of unique clusters

class_dfs = [] # list of dfs for each cluster

feats_df = feats_df.copy()
for i in ["Ua","Ui","Ga","Gi","Da","Di","Ba"]:
    feats_df[i] = feats_df[i].astype(float)

for i in classes:
    df = feats_df[feats_df["cluster"] == i]
    df.reset_index(inplace = True, drop = True)
    class_dfs.append(df)

    print("Images in class ", i, ": ", class_dfs[i].shape)

# %%
feats_df[feats_df.columns[13:111]].columns

# %%
num_imgs = 10

fig, ax = plt.subplots(num_imgs, len(class_dfs))
fig.suptitle('Classified Images From: ' + file_in, fontsize=36, y=0.97)
plt.subplots_adjust(top=0.95)

for i in range(num_imgs):
  for j in range(len(class_dfs)):
    df = class_dfs[j]
    if df.shape[0] == 0:
      continue
    num = np.random.randint(low = 0, high= df.shape[0], size=1, dtype=int)[0]
    image_path = "FINAL2" + df["dir"][num] + df["path"][num]
    img = np.array(Image.open(image_path))
    ax[i][j].imshow(img, cmap = "YlOrRd", vmin = 0, vmax = 255)
    ax[i][j].set_title(df["path"][num] + " cluster " + str(j) + " index " + str(num), fontsize = 10)
    ax[i][j].get_xaxis().set_visible(False)
    ax[i][j].get_yaxis().set_visible(False)

fig.set_figwidth(5*len(class_dfs))
fig.set_figheight(5 * num_imgs)

# option_print = True
if option_print:
  plt.savefig(specific_file_dir + "ClassifiedImages_RandomImagesPerClass_cl" + str(num_classes) + ".png", dpi = 300)
plt.close()
# plt.show()

# %%
# PCA on feat arrays
print("\n\nPCA on feature arrays:")

# scaling data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

# feat_array_scaled contain only features (no image data)
feat_array_scaled = pd.DataFrame(sc.fit_transform(feats_df[feats_df.columns[13:111]]))
print(feat_array_scaled.shape)

# Applying PCA function on training and testing set of X component
from sklearn.decomposition import PCA

pca = PCA(n_components = num_PCA_components)
pca_data = pca.fit_transform(feat_array_scaled)

explained_variance = pca.explained_variance_ratio_
print(explained_variance)

for j in range(num_PCA_components):
    feats_df["pc"+str(j+1)] = pca_data[:,j]

class_dfs = [df.copy() for df in class_dfs]
for i in range(num_classes):
    class_subset = feats_df[feats_df["cluster"] == i]
    for j in range(num_PCA_components):
        class_dfs[i].loc[:,"pc" + str(j+1)] = class_subset["pc" + str(j+1)].values

# PCA components (loadings)
loadings = pd.DataFrame(pca.components_.T,
                        columns=[f'PC{i+1}' for i in range(len(pca.components_))],
                        index=feat_array_scaled.columns)

print("\nPCA Loadings:")
print(loadings)

# Find top features for each principal component
top_features = {}
for pc in loadings.columns:
    top_features[pc] = loadings[pc].abs().nlargest(3).index.tolist()  # Top 3 contributing features
print("\nTop Contributing Features per Principal Component:")
print(top_features)

for j in range(num_PCA_components):
    print("PC" + str(j+1)+ ":")
    for feat in top_features["PC"+str(j+1)]:
        print(f"----{feats_df.columns[feat + 13]}")

# %%
# will show PCA columns added to feats_df
feats_df.head()

# %%
# plot the model predicted class in feature space
folder_name = "PCA_of_Features/"
os.makedirs(specific_file_dir + folder_name, exist_ok=True)

import plotly.graph_objects as go

feat_1 = "pc1"
feat_2 = "pc2"

alpha = [1,1,1,1,1,1,1]

if num_PCA_components == 2:
  fig, ax = plt.subplots()
  for j in classes:
    df = class_dfs[j]
    ax.scatter(df[feat_1], df[feat_2], c = colors[j], label = "class " + str(j), alpha = alpha[j])

  leg = ax.legend(loc="lower left")
  ax.set(xlabel = "PC1", ylabel = "PC2", title = "")

  # option_print = True
  if option_print:
    plt.savefig(specific_file_dir + folder_name + "PCA_of_Feats_2_Components.png", dpi = 300)
  plt.close()
  # plt.show()

if num_PCA_components == 3:
  feat_3 = "pc3"

  # 3D matplotlib plot
  fig = plt.figure(figsize=(8, 8))
  ax = plt.axes(projection='3d')

  for j in classes:
    df = class_dfs[j]
    ax.scatter3D(df[feat_1], df[feat_2], df[feat_3], c = colors[j], label = "class " + str(j), alpha = alpha[j])

  leg = ax.legend(loc="upper left")
  ax.set(xlabel = feat_1, ylabel = feat_2, zlabel = feat_3)
  ax.view_init(30,45)

  # enhanced 3D plot with plotly
  traces = []

  for i, df in enumerate(class_dfs):
    color = colors[i]
    cur_alpha = alpha[i]
    
    trace = go.Scatter3d(x=df[feat_1],y=df[feat_2],z=df[feat_3], mode='markers',
        marker=dict(
            size=5,
            color=color,
            opacity=cur_alpha
        )
    )
    traces.append(trace) 

  fig = go.Figure(data=traces)
  fig.update_layout(margin=dict(l=5, r=5, b=5, t=5))
  fig.update_layout(scene=dict(
          xaxis_title='PC1',
          yaxis_title='PC2',
          zaxis_title='PC3'
      ))
  
  # option_print = True
  if option_print:
    plt.savefig(specific_file_dir + folder_name + "PCA_of_Feats_3_Components.png", dpi = 300)
    fig.write_html(specific_file_dir + folder_name + "Interactive_Plot_PCA_of_Feats_3_Components.html")
  # # fig.show()

# %% [markdown]
# Exploration of combos of points in 2D and 3D

# %%
# plot the model predicted class in feature space
print("\n\nNow plotting parameter vs parameter combinations: ")

folder_name = "Parameter_vs_Parameter_By_Class/"
os.makedirs(specific_file_dir + folder_name, exist_ok=True)

from itertools import combinations

def plotter(feat_1,feat_2):
  fig, ax = plt.subplots()

  alpha = [1,0,1,0,1,1,0] # opacity of points
  for j in classes:
    df = class_dfs[j]
    ax.scatter(df[feat_1], df[feat_2], c = colors[j], label = "class " + str(j), alpha = alpha[j])

  leg = ax.legend(loc="upper left")
  ax.set(xlabel = feat_1, ylabel = feat_2)

  # option_print = True
  if option_print:
    plt.savefig(specific_file_dir + folder_name + feat_1 + "_vs_" + feat_2 + ".png", dpi = 300)
  plt.close()
  # plt.show()
  return

for p1,p2 in combinations(params, 2):
  plotter(p1,p2)

# %%
# do clusters in 3 parameters in 3D
print("\n\nNow plotting parameter vs parameter vs parameter combinations: ")

folder_name = "Parameter_vs_Parameter_vs_Parameter_By_Class/"
os.makedirs(specific_file_dir + folder_name, exist_ok=True)

def threeD_plotter(feat_1,feat_2,feat_3):
    # 3D matplotlib plot
    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes(projection='3d')

    alpha = [1,1,1,1,1,1,0]
    for j in classes:
      df = class_dfs[j]
      ax.scatter3D(df[feat_1], df[feat_2], df[feat_3], c = colors[j], label = "class " + str(j), alpha = alpha[j])

    leg = ax.legend(loc="upper left")
    ax.set(xlabel = feat_1, ylabel = feat_2, zlabel = feat_3)

    ax.view_init(30, 90)

    # enhanced 3D plot with plotly
    traces = []

    for i in classes:
      df = class_dfs[i]
      color = colors[i]
      cur_alpha = alpha[i]
    
      trace = go.Scatter3d(x=df[feat_1],y=df[feat_2],z=df[feat_3], mode='markers',
          marker=dict(
              size=5,
              color=color,
              opacity=cur_alpha
          ))
      traces.append(trace)

    fig = go.Figure(data=traces)
    fig.update_layout(margin=dict(l=5, r=5, b=5, t=5))
    ## fig.show()
    
    # option_print = True
    if option_print:
      plt.savefig(specific_file_dir + folder_name + feat_1 + "_vs_" + feat_2 + "_vs_" + feat_3 + ".png", dpi = 300)
      fig.write_html(specific_file_dir + folder_name + feat_1 + "_vs_" + feat_2 + "_vs_" + feat_3 + "_Interactive.html")
    plt.close()
    # plt.show()

for p1,p2,p3 in combinations(params, 3):
  threeD_plotter(p1,p2,p3)

# %%
# plot ratios of params
print("\n\nNow plotting ratios of parameters: ")

folder_name = "Ratios_of_Parameters_By_Class/"
os.makedirs(specific_file_dir + folder_name, exist_ok=True)

def ratio_plotter(feat_1,feat_2):
  for p in range(num_classes):
    class_subset = feats_df[feats_df["cluster"] == p]
    class_dfs[p].loc[:, p0 + "/" + p1] = class_subset[p0 + "/" + p1].values
    class_dfs[p].loc[:, p2 + "/" + p3] = class_subset[p2 + "/" + p3].values

  fig, ax = plt.subplots()

  alpha = [1,0,1,1,1,1,0]
  for j in classes:
    df = class_dfs[j]
    ax.scatter(df[feat_1], df[feat_2], c = colors[j], label = "class " + str(j), alpha = alpha[j])

  leg = ax.legend(loc="upper left")
  ax.set(xlabel = feat_1, ylabel = feat_2)

  # option_print = True
  feat_1_name = feat_1.replace("/","-")
  feat_2_name = feat_2.replace("/","-")
  if option_print:
    plt.savefig(specific_file_dir + folder_name + feat_1_name + "_vs_" + feat_2_name + "_Ratios.png", dpi = 300)
  plt.close()
  # plt.show()
  return

plot=1
counter = 0
for i in range(len(params)):
  for j in range(i,len(params)):
    for k in range(len(params)):
      for l in range(k,len(params)):
        p0 = params[i]
        p1 = params[j]
        p2 = params[k]
        p3 = params[l]

        if p0 != p1 and p1 != p2 and p2 != p3 and p1 != p3 and p0 != p3:
          ratio_1 = np.array(feats_df[p0]) / np.array(feats_df[p1])
          feats_df[p0 + "/" + p1] = ratio_1

          ratio_2 = np.array(feats_df[p2]) / np.array(feats_df[p3])
          feats_df[p2 + "/" + p3] = ratio_2
          class_dfs = [df.copy() for df in class_dfs]
          
          if plot:
            ratio_plotter(p0 + "/" + p1, p2 + "/" + p3)
          counter += 1
print(counter)

# %%
# shows parameter ratio columns
feats_df.columns[116:137]

# %%
# print bar plots for each param based on class
print("\n\nNow plotting mean feature and parameter values by class: ")

folder_name = "Mean_Feat_Param_Values_By_Class/"
os.makedirs(specific_file_dir + folder_name, exist_ok=True)

feats = ["Ua","Ui","Ga","Gi","Da","Di","Ba",'Ua/Ui', 'Ua/Ga', 'Ua/Gi', 'Ua/Da', 'Ua/Di', 'Ua/Ba', 'Ga/Gi', 'Ga/Da',
       'Ga/Di', 'Ga/Ba', 'Gi/Da', 'Gi/Di', 'Gi/Ba', 'Da/Di', 'Da/Ba', 'Di/Ba',
       'Ui/Gi', 'Ui/Da', 'Ui/Di', 'Ui/Ba'] + list(feats_df.columns[13:111])

classes = np.arange(0,num_classes)# list of unique clusters
rows = {str(i): [] for i in classes}

for i in range(len(feats)):
  fig, ax = plt.subplots()

  param = feats[i]
  data = []
  
  for i, df in enumerate(class_dfs):
    rows[str(i)] = df[param].values
    data.append(rows[str(i)])

  positions = np.arange(0, num_classes) * 2.5

  # outliers not shown; to show outliers, set showfliers=True
  bp = ax.boxplot(data, widths=0.5, positions=positions, patch_artist=True, showfliers=False,
              medianprops=dict(color='black'))

  for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    
  ax.set(title = "Mean " + param)
  tick_labels = [str(i) for i in range(num_classes)]
  ax.set_xticklabels(tick_labels)

  # option_print = True
  if option_print:
    if "/" in param:
      param = param.replace("/","-")
      param = param + "_Ratio"
    plt.savefig(specific_file_dir + folder_name + "Mean_" + param + ".png", dpi = 300)
  plt.close()
  # plt.show()

# %%
# scale each param to plot them next to each other
from sklearn.preprocessing import MinMaxScaler

params_df = feats_df[params]
for param in  params:
  sc = MinMaxScaler()
  param_arr = np.array(params_df[param]).reshape(-1,1)
  param_arr_scaled = sc.fit_transform(param_arr)
  feats_df[param + "_scaled"] = param_arr_scaled

feats = ['Ua_scaled', 'Ui_scaled', 'Ga_scaled', 'Gi_scaled', 'Da_scaled', 'Di_scaled', 'Ba_scaled']

class_dfs = [df.copy() for df in class_dfs]
for i in range(num_classes):
    class_subset = feats_df[feats_df["cluster"] == i]
    for feat in feats:
        class_dfs[i].loc[:, feat] = class_subset[feat].values

# %%
# print box plots for all 7 main params based on class 
print("\n\nNow plotting mean scaled parameter values by class: ")

feats = ['Ua_scaled', 'Ui_scaled', 'Ga_scaled', 'Gi_scaled', 'Da_scaled', 'Di_scaled', 'Ba_scaled']

classes = np.arange(0,num_classes)# list of unique clusters
fig, ax = plt.subplots(figsize = (12,8))

for i, class_df in enumerate(class_dfs):
  positions = np.arange(len(feats)) * 0.35 + i*3
  data = []

  for feat in feats:
     df = class_df[feat]
     data.append(df)

  # outliers not shown; to show outliers, set showfliers=True     
  bp = ax.boxplot(data, widths=0.2, positions=positions, patch_artist=True, showfliers=False,
              medianprops={'linewidth': 1.5, 'color':'white'})
  
  for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

ax.set(title = "Mean Params By Classes")

leg = ax.legend(['Ua_scaled', 'Ui_scaled', 'Ga_scaled', 'Gi_scaled', 'Da_scaled', 'Di_scaled', 'Ba_scaled'], loc="upper left")
leg = ax.get_legend()

for idx in range(len(colors)): 
    leg.legend_handles[idx].set_color(colors[idx])

xticks = [1+3*i for i in range(num_classes)]
ax.set_xticks(xticks)
tick_labels = [str(i) for i in range(num_classes)]
ax.set_xticklabels(tick_labels)

# option_print = True
if option_print:
  plt.savefig(specific_file_dir + "Mean_All_Params_Scaled_By_Class.png", dpi = 300)
plt.close()
# plt.show()

# %%
from sklearn.linear_model import LinearRegression

feat_1 = np.array(feats_df["num_spots"]).reshape(-1,1)
param_1 = np.array(feats_df["Ua"]).reshape(-1,1)

lin_reg=LinearRegression()
lin_reg.fit(feat_1,param_1)

# %%
model = LinearRegression()

 #define predictor and response variables
feat_1 = np.array(feats_df["num_spots"]).reshape(-1,1)
param_1 = np.array(feats_df["Ua"]).reshape(-1,1)

# fit linear regression model
model.fit(param_1, feat_1)

#calculate R-squared (value indicating how well the data fits the regression model)
r_squared = model.score(param_1, feat_1)
print(r_squared)

# %%
# calculate R-squared value for all features vs all params
all_params = ["Ua","Ui","Ga","Gi","Da","Di","Ba",'Ua/Ui', 'Ua/Ga', 'Ua/Gi', 'Ua/Da', 'Ua/Di', 'Ua/Ba', 'Ga/Gi', 'Ga/Da',
       'Ga/Di', 'Ga/Ba', 'Gi/Da', 'Gi/Di', 'Gi/Ba', 'Da/Di', 'Da/Ba', 'Di/Ba',
       'Ui/Gi', 'Ui/Da', 'Ui/Di', 'Ui/Ba']
feats = list(feats_df.columns[13:111])

p_list = []
f_list = []
r2 = []

for param in all_params:
  for feat in feats:
    model = LinearRegression()

    feat_1 = np.array(feats_df[feat]).reshape(-1,1)
    param_1 = np.array(feats_df[param]).reshape(-1,1)

    model.fit(param_1, feat_1)

    r_squared = model.score(param_1, feat_1)

    r2.append(r_squared)
    p_list.append(param)
    f_list.append(feat)

r2_dic = {"feat": f_list,"param":p_list,"r2":r2}
r2_df = pd.DataFrame(r2_dic)
r2_df.head()

# %%
# prints highest to lowest R-squared values for given feature
r2_sorted = r2_df.sort_values(by=['r2'], ascending = False)

r2_sorted[r2_sorted["feat"] == "Mean"]

# %%
# plot the model predicted class in feature space
print("\n\nNow plotting parameters by class in the feature space: ")

folder_name = "Param_By_Class_in_Feature_Space/"
os.makedirs(specific_file_dir + folder_name, exist_ok=True)

# features from num_spots to solidity_std
image_features = list(feats_df.columns[13:62])

classes = np.arange(0,num_classes)

for feat in image_features:
  for param in params:
    feat_1 = feat
    feat_2 = param

    fig, ax = plt.subplots()

    alpha = [1,1,1,1,1,1,0]
    for j in classes:
      df = class_dfs[j]
      ax.scatter(df[feat_1], df[feat_2], c = colors[j], label = "class " + str(j), alpha = alpha[j])

    leg = ax.legend(loc="upper left")
    ax.set(xlabel = feat_1, ylabel = feat_2)

    # option_print = True
    if "/" in feat_2:
      feat_2 = feat_2.replace("/","-")
    if option_print:
      plt.savefig(specific_file_dir + folder_name + feat_1 + "_vs_" + feat_2 + ".png", dpi = 300)
    plt.close()
    # plt.show()

# %%
# file that contains dataframe of features for each param value when other param values don't change; doesn't contain cluster info
with open("FINAL2/data/single_param_scans.pkl", 'rb') as f:
  single_dic = pickle.load(f) # deserialize using load()

# %%
single_dic["Ui"].head(20)

# %%
# create a new dictionary containing the cluster info by searching for the paths in feats_df_large
# note: this is not on the narrow param domain like before
indices = {"Ua":[],"Ui":[],"Ga":[],"Gi":[],"Da":[],"Di":[],"Ba":[]}

for i in single_dic.keys():
    df = single_dic[i]
    df.reset_index(inplace=True, drop=True)
    for j in range(df.shape[0]):
        path = df["path"][j]
        line = feats_df_large[feats_df_large["path"] == path]
        indices[i].append(line.index[0])

single_classes = {"Ua":None,"Ui":None,"Ga":None,"Gi":None,"Da":None,"Di":None,"Ba":None}

for i in indices.keys():
  ind_list = indices[i]
  df = feats_df_large.iloc[ind_list]
  df.reset_index(inplace = True, drop = True)
  single_classes[i] = df

# %%
# now shows dataframe with cluster info
single_classes["Ua"].head()

# %%
#sort all the dfs
for i in single_classes.keys():
  df = single_classes[i].copy()
  df[params] = df[params].astype(float)
  df = df.sort_values(by = i)
  single_classes[i] = df

# print out random images from clusters
num_imgs = 10

for i in range(num_imgs):
  fig, ax = plt.subplots(1,len(params))
  for j in range(len(params)):
    df = single_classes[params[j]]
    path = "FINAL2" + df["dir"][i] + df["path"][i]
    img = np.array(Image.open(path))
    ax[j].imshow(img, cmap = "YlOrRd", vmin = 0, vmax = 255)
    ax[j].set_title(str(params[j]) + " = "+ str(df[params[j]][i]), fontsize = 8)
    print(str(params[j]) + " = "+ str(df[params[j]][i]), end=", ")
    ax[j].get_xaxis().set_visible(False)
    ax[j].get_yaxis().set_visible(False)
   
  fig.set_figwidth(5*len(params))
  fig.set_figheight(5*num_imgs)

  # option_print = True
  # if option_print:
  #   plt.savefig(specific_file_dir + "Single_Param_Scans_Random_Images_By_Class" + str(i) + ".png", dpi = 300)
  plt.close()
  # plt.show()


# %%
# plot some dependencies
print("\n\nNow plotting features against parameters in single parameter scans: ")

folder_name = "Linear_Scans_Feat_vs_Param/"
os.makedirs(specific_file_dir + folder_name, exist_ok=True)

keys = list(single_classes.keys())
feat = "num_spots"

# iterate through params
for feat in image_features:
  for i in range(len(keys)):
    fig, ax = plt.subplots()
    # iterate through classes
    for j in range(num_classes):
      # pull up the dataframe for the specific class
      df = single_classes[keys[i]]
      df = df[df["cluster"] == j]
      df.reset_index(inplace = True, drop = True)

      x = df[keys[i]].astype(float)
      y = df[feat].astype(float)

      ax.scatter(x,y, color = colors[j])
    ax.legend([0,1,2,3,4,5,6])
    ax.set(title = f"Dependency of {feat} on {keys[i]}", xlabel = keys[i], ylabel = feat)

    # option_print = True
    if option_print:
      plt.savefig(specific_file_dir + folder_name + feat + "_vs_" + keys[i] + "_SingleParamScan.png", dpi = 300)
    plt.close()
    # plt.show()


