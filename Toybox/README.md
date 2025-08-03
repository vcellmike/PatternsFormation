# Data and algorithms sources for ML analysis

## Image-Containing Zip Files
 - 05-09-2024_all_images.zip contains 15,133 images in png format, generated using VCell software using the same initial distribution of A protein (noise).
 - Generated Images.zip contains 4,523 images generated using VCell software using 5 different initial distributions of A proteins, called noise3 (598 images), ..., noise6 (1,000 images), noise7 (999 images)
 - Generated Images 7-2-24 contains 19,858 images in png format, generated using VCell software using 5 different initial distributions of A proteins, called noise 3 (4,000 images), noise 4 (3,900 images), noise 5 (3,965 images), noise 6 (3,993 images), noise 7 (4,000 images)
 - real_images_6-29-25 contains 13 photographs of real monkeyflowers. This data is to be used for To Silico conversion.

## Data Files
 - Copy of 2024-08-18_curr_df.pkl contains a serialized Pandas dataframe with 32210 rows × 8 columns. It encodes noise, class, path variables (path, dir), seed, predicted class, pc1, and pc2.
 - Copy of 2024-08-19_feats_df_narrow.pkl contains a serialized dataframe with 33770 rows × 111 columns. It encodes 111 columns (features, file information, parameters etc)
 - 2025-05-24_feats_df_toybox.pkl contains a serialized dataframe with shape 27515 rows × 113 columns. This is the dataframe used by Data Loader, PCA, and Visualizations.
 - 2024-08-15_manual_classification.pkl contains a serialized dataframe with shape 7307 rows × 5 columns. It encodes class, noise, path, seed dir. This is the pickle file used in the NN notebook/python file.

## Algorithms
The algorithm used was a convolutional neural netowrk that classified images into classes by clustering them. A dataframe inside a pickle file for clustering was created that included the filename and class. The algorithm split the data into training, test, and validating data. Finally, a confusion matrix was created to represent the accuracy of the model in classifying an image into the correct class.
