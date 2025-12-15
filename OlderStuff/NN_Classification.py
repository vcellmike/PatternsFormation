# %%
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os 
import sys
# run data loader to unzip images

# %%
# set seed needed for reproducibility in training
import random

def set_seed(seed_val):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

# %%
seed_val = 42
set_seed(seed_val)

# %%
# load manually classified data with image dir using Images1, Images2, Images3 folders
option_print = True

file_in = "FINAL2/data/manual_classification_cl7"

with open(file_in+".pkl", 'rb') as f:
  feats_df = pickle.load(f) # deserialize using load()

print("number of manually classified images:", feats_df.shape)

manually_classified_paths = feats_df["path"].unique().tolist()


output_dir = "FINAL2/data/output_images/"
os.makedirs(output_dir, exist_ok=True)

output_sub_dir = output_dir + file_in[11:] + "/"
os.makedirs(output_sub_dir, exist_ok=True)

specific_file_dir = output_sub_dir + "NN_Classification/"
os.makedirs(specific_file_dir, exist_ok=True)

# save output to txt file
sys.stdout = open(specific_file_dir +'NN_Classification_output.txt', 'w')  # redirect print() output
sys.stderr = sys.stdout

# %%
counter = 0

# Creates a list of non-integer values
non_ints = []

# Cuts down the dataframe and isolates "class", and loops through each value.
for i in feats_df["class"]:
  # Try-Except statement to ensure all numbers are integer values
  try:
    a = int(i)
  except:
    non_ints.append(counter)
  counter += 1

print("Images with no manually classified class", non_ints)

# %%
#Find the number of classes in manually classified data

feats_df = feats_df.drop(non_ints, axis = "rows") # drop 3 images with no class value
feats_df.reset_index(inplace = True, drop = True)
feats_df["class"] = np.array(feats_df["class"].astype(int) -1).astype(int)

values_clusters = feats_df["class"].unique()
print("cluster column values: ", values_clusters)

num_classes = values_clusters.astype(int).max() + 1
print("num classes:", num_classes)

file_out = file_in + "_NN_classes_" + str(num_classes) + "_s" + str(seed_val) + ".pkl"

# %%
# number of images per manually classified classes
for i in np.unique(np.array(feats_df["class"])):
  print("class", i, "has", np.sum(np.array(feats_df["class"]) == i), "images")

# %%
cluster_dfs = [] # list of dfs for each cluster

for i in range(0, num_classes):
  df = feats_df[feats_df["class"] == i]
  print("class", i, "shape:", df.shape)
  df.reset_index(inplace = True, drop = True)
  cluster_dfs.append(df)

num_imgs = 10

fig, ax = plt.subplots(num_imgs, len(cluster_dfs))
fig.suptitle('Manually Classified Images', fontsize=36, y=0.97)
plt.subplots_adjust(top=0.95)

for i in range(num_imgs):
  for j in range(len(cluster_dfs)):
    df = cluster_dfs[j]
    if df.shape[0] == 0:
      continue
    num = np.random.randint(low = 0, high= df.shape[0], size=1, dtype=int)[0]
    image_path = "FINAL2/" + df["dir"][num] + df["path"][num]
    img = np.array(Image.open(image_path))
    ax[i][j].imshow(img, cmap = "YlOrRd", vmin = 0, vmax = 255)
    ax[i][j].set_title(df["path"][num] + " cluster " + str(j) + " index " + str(num), fontsize = 10)
    ax[i][j].get_xaxis().set_visible(False)
    ax[i][j].get_yaxis().set_visible(False)

fig.set_figwidth(5*len(cluster_dfs))
fig.set_figheight(5 * num_imgs)

#option_print = True
if option_print:
  plt.savefig(specific_file_dir + "Manual_Classification_ImagesPerCluster_cl7_.png", dpi = 300)

plt.show()

# %%
# change size of sampling data 
frac_train = 0.01

df_train = feats_df.sample(frac = frac_train, random_state = 42)
df_train.reset_index(inplace = True, drop = True)
df_val_test = feats_df.drop(df_train.index)

p_out = 0.25

df_val_test = df_val_test.sample(frac = p_out, random_state = 42)
df_val = df_val_test.sample(frac = 0.5, random_state = 42)
df_test = df_val_test.drop(df_val.index)

df_val.reset_index(inplace = True, drop = True)
df_test.reset_index(inplace = True, drop = True)

print("df_train shape: ", df_train.shape)
print("df_val shape: ", df_val.shape)
print("df_test shape: ", df_test.shape)

# %%
import torch.optim as optim
from torchvision import models
# TRAINING SETTINGS
NUM_EPOCHS = 20

# LEARNING RATE SETTINGS
BASE_LR = 1e-5
DECAY_WEIGHT = 0.1 # factor by which the learning rate is reduced.
EPOCH_DECAY = 15 # number of epochs after which the learning rate is decayed exponentially by DECAY_WEIGHT.

# # DATASET INFO
# NUM_CLASSES = 2 # set the number of classes in your dataset
# # DATA_DIR = 'hymenoptera_data/' # to run with the sample dataset, just set to 'hymenoptera_data'

# DATALOADER PROPERTIES
BATCH_SIZE = 64

# GPU SETTINGS
CUDA_DEVICE = 0 # Enter device ID of your gpu if you want to run on gpu. Otherwise neglect.
GPU_MODE = 0 # set to 1 if want to run on gpu.

# # SETTINGS FOR DISPLAYING ON TENSORBOARD
# USE_TENSORBOARD = 0 #if you want to use tensorboard set this to 1.
# TENSORBOARD_SERVER = "YOUR TENSORBOARD SERVER ADDRESS HERE" # If you set.
# EXP_NAME = "Nathan_Car" # if using tensorboard, enter name of experiment you want it to be displayed as.

use_gpu = GPU_MODE
if use_gpu:
    torch.cuda.set_device(CUDA_DEVICE)
# show transforms and dataset loader, from https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

# %%
def data_transforms(image, mode):
  if mode == "train":
    tnsfrm = transforms.Compose([
            # transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor()
        ])
  elif mode == "val":
    tnsfrm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224,224))
    ])

  img = tnsfrm(image)
  img = torch.cat((img,img,img),0)
  return img

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, transform, mode, target_transform):
      # image dir and file_name
        self.img_dir = annotations_file["dir"]
        self.path = annotations_file["path"]
      # gets the labels from the dataframe provided in the dataloader
        self.img_labels = annotations_file["class"]
      #transform is defined previously
        self.transform = transform
      #no target transforms for this task
        self.target_transform = target_transform
      # training mode
        self.mode = mode

    def __len__(self):
      #gives the size of whatever dataset is being loaded
        return len(self.img_labels)

    def __getitem__(self, idx):
      #get the path to the image
        img_path = "FINAL2/" + self.img_dir[idx] + self.path[idx]
        #loads in the image
        image = Image.open(img_path)
        #gets the corresponding label
        label = self.img_labels.iloc[idx]
        label_list = np.zeros(7)
        label_list[label - 1] = 1
        #does the transforms
        if self.transform:
            image = self.transform(image,self.mode)
        if self.target_transform:
            label = self.target_transform(label)
            
        #returns the transformed image and its label
        return image.to(torch.float32), torch.tensor(label)

dsets = {}
dsets["train"] = CustomImageDataset(df_train, data_transforms, "train", None)
dsets["val"] = CustomImageDataset(df_val, data_transforms, "val", None)

train_dataloader = DataLoader(dsets["train"], batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(dsets["val"], batch_size=BATCH_SIZE, shuffle=True)

dset_loaders = {"train":train_dataloader, "val":val_dataloader}

dset_sizes = {}
dset_sizes['train'] = df_train.shape[0]
dset_sizes['val'] = df_val.shape[0]

print(dset_sizes)

# %%
#The main training block for the model
import time
from torch.autograd import Variable
import copy
def train_model(model, criterion, optimizer, lr_scheduler, num_epochs):
    since = time.time()

    best_model = model
    best_loss = float('inf')

    losses = {'train': [], 'val': []}

    # uncomment all accuracies lines to also return accuracies
    # accuracies = {'train': [], 'val': []}
    
    for epoch in range(num_epochs):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                optimizer = lr_scheduler(optimizer, epoch)
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to eval mode for validation (no need to track gradients)

            running_loss = 0.0
            # running_corrects = 0

            counter=0
            # Iterate over data, getting one batch of inputs (images) and labels each time.
            for data in dset_loaders[phase]:
                inputs, labels = data
                if use_gpu:
                    try:
                        inputs, labels = Variable(inputs.float().cuda()), Variable(labels.long().cuda())
                    except Exception as e:
                        print("ERROR! here are the inputs and labels before we print the full stack trace:")
                        print(inputs, labels)
                        raise e
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # Set gradient to zero to delete history of computations in previous epoch. Track operations so that differentiation can be done automatically.
                optimizer.zero_grad()
                outputs = model(inputs)

                # _, preds = torch.max(outputs, 1) # for accuracies
                # preds = preds.to(torch.float32) # for accuracies
                
                loss = criterion(outputs, labels).to(torch.float32)

                # Print a line every 10 batches so you have something to watch and don't feel like the program isn't running.
                if counter%10==0:
                    print("Reached batch iteration", counter)
                counter+=1

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                try:
                    running_loss += loss.item()
                    #running_corrects += torch.sum(preds == labels.data)
                except:
                    print('unexpected error, could not calculate loss or do a sum.')

            epoch_loss = running_loss / dset_sizes[phase]
            # epoch_acc = running_corrects.item() / float(dset_sizes[phase])
            
            print('Epoch' + str(epoch) + "loss = " + str(epoch_loss))
            # accuracies[phase].append(epoch_acc)
            losses[phase].append(epoch_loss)

            # deep copy the model
            if phase == 'val':
                # tensorboard - visualize training process in real-time
                # if USE_TENSORBOARD:
                #     foo.add_scalar_value('epoch_loss', epoch_loss,step=epoch)
                #     foo.add_scalar_value('epoch_acc', epoch_acc,step=epoch)
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model = copy.deepcopy(model)
                    print('new best loss =', best_loss)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best loss: {:4f}'.format(best_loss))
    print('returning and looping back')

    return best_model, losses # return accuracies as well if needed

# This function changes the learning rate as the model trains.
def exp_lr_scheduler(optimizer, epoch, init_lr=BASE_LR, lr_decay_epoch=EPOCH_DECAY):
    """Decay learning rate by a factor of DECAY_WEIGHT every lr_decay_epoch epochs."""
    lr = init_lr * (DECAY_WEIGHT**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

#start with resnet50
model_ft = models.resnet50(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 7)

#set loss function to binary crossentropy
criterion = nn.CrossEntropyLoss()

if use_gpu:
    criterion.cuda()
    model_ft.cuda()

optimizer_ft = optim.AdamW(model_ft.parameters(), lr=0.0001)

# Run the functions and save the best model in the function model_ft.
model_ft, losses = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=NUM_EPOCHS)

# to also print accuracies by epoch
# model_ft, losses, accuracies = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=NUM_EPOCHS)

for split in ['train', 'val']:
    # print(split, "accuracies by epoch:", accuracies[split])
    print(split, "losses by epoch:", losses[split])

# %%
model_saved = file_in + "_frac_" + str(frac_train)+".pth"
torch.save(model_ft.state_dict(), model_saved)

# %%
#initialize new model with same architecture as the one that was trained
pred_model = models.resnet50(pretrained=False)
num_ftrs = pred_model.fc.in_features
pred_model.fc = nn.Linear(num_ftrs, 7)
#pred_model.cuda()

#apply saved weights to the model
pred_model.load_state_dict(torch.load(model_saved))

#set model mode to eval
pred_model.eval()

# %%
print(df_test.shape)
df_test.head()

# %%
def data_transforms(image, mode):
  if mode == "train":
    tnsfrm = transforms.Compose([
            # transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor()
        ])
  elif mode == "val":
    tnsfrm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224,224))
    ])

  img = tnsfrm(image)
  img = torch.cat((img,img,img),0)
  return img

# make predictions on the image
preds = []
counter = 0

with torch.no_grad():
  for i in range(df_test.shape[0]):
    image_path = "FINAL2/" + df_test["dir"][i] + df_test["path"][i]
    pred_image = np.array(Image.open(image_path))
    
    pred_img_transformed = data_transforms(pred_image,"val")
    
    preds.append((pred_model(pred_img_transformed.unsqueeze(0))))

    if counter % 250 == 0:
      print(counter)

    counter += 1

# %%
# add predicted class to the test df
pred_class = []

for pred in preds:
  p = pred.cpu().numpy()
  argmax = np.argmax(p)
  pred_class.append(argmax)

df_test["pred_class"] = pred_class

df_test.head()

# %%
# confusion matrix
from sklearn import metrics

actual = df_test["class"]
predicted = df_test["pred_class"]

confusion_matrix = metrics.confusion_matrix(actual, predicted)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1 , 2, 3, 4, 5, 6])

cm_display.plot()

#option_print = True
if option_print:
  plt.savefig(specific_file_dir + "ConfusionMatrix_ManClass_and_PredClass.png", dpi = 300)

plt.show()

# %%
num_correct = 0
num_incorrect = 0

for i in range(df_test.shape[0]):
  if int(df_test["class"][i]) == int(df_test["pred_class"][i]):
    num_correct += 1
  else:
    num_incorrect += 1

print("accuracy = " + str(num_correct/(num_correct + num_incorrect)))

# %%
# split into dataframes for each cluster
clusters = np.unique(np.array(feats_df["class"])) # list of unique clusters
print(np.unique(np.array(df_test["pred_class"]), return_counts = True)) 

cluster_dfs = [] # list of dfs for each cluster

for i in clusters:
  df = df_test[df_test["pred_class"] == i]
  df.reset_index(inplace = True, drop = True)
  cluster_dfs.append(df)

# print sizes of each pred_cluster
print("\nnumber of images per predicted class in the test set:")
class_num = 0
for df in cluster_dfs:
  print(str(class_num) + ": " + str(df.shape[0]))
  class_num += 1

# %%
# visualize images from each predicted class:
num_imgs = 10

fig, ax = plt.subplots(num_imgs, len(cluster_dfs))
fig.suptitle('Images from Each Predicted Class - Test Dataset', fontsize=36, y=0.97)
plt.subplots_adjust(top=0.95)

for i in range(num_imgs):
  for j in range(len(cluster_dfs)):
    df = cluster_dfs[j]
    if df.shape[0] == 0:
      continue
    num = np.random.randint(low = 0, high= df.shape[0], size=1, dtype=int)[0]
    image_path = "FINAL2/" + df["dir"][num] + df["path"][num]
    img = np.array(Image.open(image_path))
    ax[i][j].imshow(img, cmap = "YlOrRd", vmin = 0, vmax = 255)
    ax[i][j].set_title(df["path"][num] + " cluster " + str(j) + " index " + str(num), fontsize = 8)
    ax[i][j].get_xaxis().set_visible(False)
    ax[i][j].get_yaxis().set_visible(False)

fig.set_figwidth(5*len(cluster_dfs))
fig.set_figheight(5 * num_imgs)

#option_print = True
if option_print:
  plt.savefig(specific_file_dir + "ImagesPerPredictedClass_TestDataset.png", dpi = 300)

plt.show()

# %%
# extract features from the rest of the unclassified images

# Load the pretrained model
pred_model = models.resnet50(pretrained=False)
num_ftrs = pred_model.fc.in_features
pred_model.fc = nn.Linear(num_ftrs, 7)
#pred_model.cuda()

pred_model.load_state_dict(torch.load(model_saved))

# %%
# load full dataframe with 39517 images and 111 columns
file_to_classify = "FINAL2/data/unclassified_features_39517"
with open(file_to_classify+".pkl", 'rb') as f:
  feats_df = pickle.load(f) # deserialize using load()
feats_df.head()

# %%
# drop already manually classified images from the full dataframe
feats_df = feats_df[~feats_df["path"].isin(manually_classified_paths)]
feats_df.reset_index(inplace = True, drop = True)

print(feats_df.shape)
feats_df.head()

# %%
df_clustering = feats_df

# From ChatGPT
# This list will store the features
features = []

def hook(module, input, output):
    features.append(output)

pred_model.avgpool.register_forward_hook(hook)
pred_model.eval()

counter = 0
# # Load an image
for i in range(df_clustering.shape[0]):
# for i in range(1):
  img_path = "FINAL2/" + df_clustering["dir"][i] + df_clustering["path"][i]
  img = Image.open(img_path)
  img_transformed = data_transforms(img, "val")

  # Pass the image through the model
  with torch.no_grad():
      _ = pred_model(img_transformed.unsqueeze(0))

  if counter % 500 == 0:
    print(counter)
  counter += 1

  # # The features are stored in the list
  # features = features[0]
  # print(features.shape)

# %%
# save 2048 features in a numpy array
feat_array = np.empty((df_clustering.shape[0], 2048))

for i in range(df_clustering.shape[0]):
  feats = np.array(features[i].cpu()).reshape(2048)
  feat_array[i,:] = feats

print(feat_array.shape)

# %%
# save features to a pickle file
file_NN_features = file_to_classify + "_NN_2048.pkl"
print(file_NN_features)

pickle_out = open(file_NN_features,"wb")
pickle.dump(feat_array, pickle_out)
pickle_out.close()

# load features from the pickle file
with open(file_NN_features, 'rb') as f:
  feat_array = pickle.load(f) # deserialize using load()

# %%
# PCA on feat array: 5 components; uses clusters defined from KMeans clustering

# scaling data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

feat_array_scaled = sc.fit_transform(feat_array)
print(feat_array_scaled.shape)

# Applying PCA function on training and testing set of X component
from sklearn.decomposition import PCA

pca = PCA(n_components = 5)
pca_data = pca.fit_transform(feat_array_scaled)

explained_variance = pca.explained_variance_ratio_
print(explained_variance)

# %%
# K_means clustering
from sklearn.cluster import KMeans

#Find optimum number of clusters
sse = [] #SUM OF SQUARED ERROR
for k in range(1,11):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(feat_array_scaled)
    sse.append(km.inertia_)

fig, ax = plt.subplots()
ax.set(xlabel = "num clusters", ylabel = "SSE")

ax.plot(sse)

# %%
# K_means clustering
num_clusters = 7
km = KMeans(n_clusters = num_clusters, init='k-means++', n_init=50, max_iter=300,
            tol=0.0001, random_state=42)

clusters = km.fit(feat_array)

# %%
# add cluster labels and PCA components to the dataframe
df_clustering["cluster"] = clusters.labels_
df_clustering["pc1"] = pca_data[:,0]
df_clustering["pc2"] = pca_data[:,1]

# %%
clusters = np.arange(0,num_clusters)# list of unique clusters

cluster_dfs = [] # list of dfs for each cluster

for i in clusters:
  df = df_clustering[df_clustering["cluster"] == i]
  df.reset_index(inplace = True, drop = True)
  cluster_dfs.append(df)

# %%
# plot the data
fig, ax = plt.subplots()

colors = ["red","green","blue","purple","orange","black","pink"]

for j in clusters:
  df = cluster_dfs[j] 
  ax.scatter(df["pc1"], df["pc2"], c = colors[j], label = "cluster " + str(j))

leg = ax.legend(loc="upper left")
ax.set(xlabel = "PC1", ylabel = "PC2")

#option_print = True
if option_print:
  plt.savefig(specific_file_dir + "Clustering_2048_Features.png", dpi = 300)

plt.show()

# %%
counter = 0
for cluster in cluster_dfs:
  print("cluster " + str(counter) + ": " + str(cluster.shape[0]))
  counter += 1

# %%
# print out random images from KMeans clusters
num_imgs = 10

fig, ax = plt.subplots(num_imgs, len(cluster_dfs))
fig.suptitle('Images Per Cluster from KMeans Clustering', fontsize=36, y=0.97)
plt.subplots_adjust(top=0.95)

for i in range(num_imgs):
  for j in range(len(cluster_dfs)):
    df = cluster_dfs[j]
    num = np.random.randint(low = 0, high= df.shape[0], size=1, dtype=int)[0]
    image_path = "FINAL2/" + df["dir"][num] + df["path"][num]
    img = np.array(Image.open(image_path))
    ax[i][j].imshow(img, cmap = "YlOrRd", vmin = 0, vmax = 255)
    ax[i][j].set_title(df["path"][num] + " cluster " + str(j) + " index " + str(num), fontsize = 8)
    ax[i][j].get_xaxis().set_visible(False)
    ax[i][j].get_yaxis().set_visible(False)

fig.set_figwidth(5*len(cluster_dfs))
fig.set_figheight(5 * num_imgs)

#option_print = True
if option_print:
  plt.savefig(specific_file_dir + "Clustering_RandomImagesPerCluster.png", dpi = 300)

plt.show()

# %%
#initialize new model with same architecture as the one that was trained
pred_model = models.resnet50(pretrained=False)
num_ftrs = pred_model.fc.in_features
pred_model.fc = nn.Linear(num_ftrs, 7)
#pred_model.cuda()

#apply saved weights to the model
pred_model.load_state_dict(torch.load(model_saved))

#set model mode to eval
pred_model.eval()

# %%
df_classes = feats_df

def data_transforms(image, mode):
  if mode == "train":
    tnsfrm = transforms.Compose([
            # transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor()
        ])
  elif mode == "val":
    tnsfrm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224,224))
    ])

  img = tnsfrm(image)
  img = torch.cat((img,img,img),0)
  return img

# make predictions on the image
preds = []
counter = 0

with torch.no_grad():
  for i in range(df_classes.shape[0]):
    image_path = "FINAL2/" + df_classes["dir"][i] + df_classes["path"][i]
    pred_image = np.array(Image.open(image_path))
    pred_img_transformed = data_transforms(pred_image,"val")
    preds.append((pred_model(pred_img_transformed.unsqueeze(0))))

    if counter % 500 == 0:
      print(counter)
    counter += 1

# %%
# add preds to df
pred_class = []

for i in preds:
  p = i.cpu()
  argmax = np.argmax(p)
  pred_class.append(int(argmax))

df_classes["pred_class"] = pred_class

# %%
np.unique(np.array(df_classes["pred_class"]))

num_NN_classes = df_classes["pred_class"].unique().max() + 1
print(num_NN_classes)

# %%
# PCA on feat arrays: 2 components; uses predicted classes from trained model

# scaling data
sc = StandardScaler()

feat_array_scaled = sc.fit_transform(feat_array)
print(feat_array_scaled.shape)

# Applying PCA function on training and testing set of X component

pca = PCA(n_components = 2)
pca_data = pca.fit_transform(feat_array_scaled)

explained_variance = pca.explained_variance_ratio_
print(explained_variance)

df_classes["pc1"] = pca_data[:,0]
df_classes["pc2"] = pca_data[:,1]

# %%
# plot the model predicted class in feature space
classes = np.arange(0,num_NN_classes)# list of unique clusters
class_dfs = [] # list of dfs for each cluster

for i in classes:
  df = df_classes[df_classes["pred_class"] == i]
  df.reset_index(inplace = True, drop = True)
  class_dfs.append(df)

fig, ax = plt.subplots()

colors = ["red","green","blue","purple","orange","black","pink"]
alpha = [1,1,1,1,1,1,0]
for j in classes:
  df = class_dfs[j]
  ax.scatter(df["pc1"], df["pc2"], c = colors[j], label = "class " + str(j), alpha = alpha[j])

leg = ax.legend(loc="upper left")
ax.set(xlabel = "PC1", ylabel = "PC2")

#option_print = True
if option_print:
  plt.savefig(specific_file_dir + "TrainedModel_Clustering_2048_Features.png", dpi = 300)

plt.show()

# %%
# print out random images from clusters
num_imgs = 10

fig, ax = plt.subplots(num_imgs, len(class_dfs))
fig.suptitle('Images Per Predicted Class from Trained Model', fontsize=36, y=0.97)
plt.subplots_adjust(top=0.95)

for i in range(num_imgs):
  for j in range(len(class_dfs)):
    df = class_dfs[j]
    num = np.random.randint(low = 0, high= df.shape[0], size=1, dtype=int)[0]
    image_path = "FINAL2/" + df["dir"][num] + df["path"][num]
    img = np.array(Image.open(image_path))
    ax[i][j].imshow(img, cmap = "YlOrRd", vmin = 0, vmax = 255)
    ax[i][j].set_title(df["path"][num] + " cluster " + str(j) + " index " + str(num), fontsize = 8)
    ax[i][j].get_xaxis().set_visible(False)
    ax[i][j].get_yaxis().set_visible(False)

fig.set_figwidth(5*len(class_dfs))
fig.set_figheight(5 * num_imgs)

#option_print = True
if option_print:
  plt.savefig(specific_file_dir + "TrainedModel_RandomImagesPerClass.png", dpi = 300)

plt.show()

# %%
# save the dataframe with predicted classes and PCA components to a pickle file
file_out = file_in + "_NN_classes_" + str(num_classes) + "_s" + str(seed_val) + ".pkl"
pickle_out = open(file_out,"wb")
pickle.dump(df_classes, pickle_out)
pickle_out.close()

# %%
counter = 0
for cluster in class_dfs:
  print("cluster " + str(counter) + ": " + str(cluster.shape[0]))
  counter += 1


