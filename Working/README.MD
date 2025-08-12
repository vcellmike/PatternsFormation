# Todo:
 - Change Dir --> Images
 - Replace in Readme

# Workflow

The folder titled Working contains working copies of all data, notebooks, 
Python files.
- Data_loader 
- PCA 
- Visualizations

Visualizations and PCA are both runable on their own. Run data_Loader to unzip data files (eg. Generated_Images_7-2-24).

## Image-Containing Zip Files
 - 05-09-2024_all_images.zip contains 15,133 images in png format, generated using VCell software using the same initial distribution of A protein (noise).
 - Generated Images.zip contains 4,523 images generated using VCell software using 5 different ireinitial distributions of A proteins, called noise3 (598 images), noise4 (927 images), noise5 (999 images), noise6 (1,000 images), noise7 (999 images)
 - Generated Images 7-2-24 contains 19,858 images in png format, generated using VCell software using 5 different initial distributions of A proteins, called noise 3 (4,000 images), noise 4 (3,900 images), noise 5 (3,965 images), noise 6 (3,993 images), noise 7 (4,000 images)
 - real_images_6-29-25 contains 13 photographs of real monkeyflowers. This data is to be used for To Silico conversion.

## Data Files
 - Copy of 2024-08-18_curr_df.pkl contains a serialized Pandas dataframe with 32210 rows × 8 columns. It encodes noise, class, path variables (path, dir), seed, predicted class, pc1, and pc2.
 - Copy of 2024-08-19_feats_df_narrow.pkl contains a serialized dataframe with 33770 rows × 111 columns. It encodes 111 columns (features, file information, parameters etc)
 - Images_Classified.pkl contains a serialized dataframe with shape 27515 rows × 113 columns. This is the dataframe used by Data Loader, PCA, and Visualizations. Thsis file is the result of a classifier that yielded the classifier_pred_class value.
 - 2024-08-15_manual_classification.pkl contains a serialized dataframe with shape 7307 rows × 5 columns. It encodes class, noise, path, seed dir. This is the pickle file used in the NN notebook/python file.


# Algorithms
## To_Silico_Conversion

**Silico = Generated/synthesized image reduced down to be analyzed 

To Silico Conversion takes an input of a zip file with real images in it. The images are photographs of real monkeyflowers. The python file attempts to convert each file into a silico representation to be analyzed by PCA, visualization, or fed through the NN classifier.

### Dependencies:
 - Specific dependencies include PILLOW, PILLOW.Image, and PILLOW.ImageEnhance ???

### Steps:
1. Unzip "real_images_6-29-25.zip" and extract images to a folder within the content folder. The images should be numbered from 1-N.png.
2. Iterate over all images in the extracted folder and adds the file paths to a dataframe of N rows by 2 columns (path, path_truncated).
3. Iterate over each path in the dataframe, pull it with pull_from_df (see comments in program), and apply a function round_pixels_by_intensity to each (see inputs in python file).
4. The round_pixels_by_intensity takes a threshold and contrast True/False value. Each pixel in the image is either set to (0,0,0) or (255, 255, 255) depending on whether it is greater than or less than the threshold. If the pixel intensity is less than the threshold, intensity is set to 0. If it is greater, than it is set to 255.
5. If enhance is true, then the program uses ImageEnhance to increase the contrast of the image by a contrast value set by "contrast_factor".
6. The image outputs the original image, the enhanced/unenhanced image, and then the final grayscale image. 

### To Run:
In the main part of the program run:
to_silico(number_of_file*, enhance=True, contrast_factor=1.2 , threshold=162)
* number_of_file corresponds with the number in the filename. eg. 7.png
You can adjust the parameters to see if a more effective in_silico conversion can be made.

### Current Status/Results

Currently, the results are not effective enough. The outputed photos are either over or underexposed in the majority of cases. Work is also being done to adopt the To Silico Conversion to the github environment in so far as changes in the filestructure being reflected in the code. 

Below are the current statuses of the images.

Blackwashed = too dark (ie. pigment is overexpressed)

Whitewashed = too light (ie. pigment is underexpressed)

1.png - Satisfactory

2 - Satisfactory

3 - Blackwashed

4 - Blackwashed

5- Whitewashed

6 - Misc.

7 - Satisfactory

8 - Blackwashed

9 - Whitewashed

10 - Blackwashed (detects 1 spot that does not exist)

11 - Blackwashed

12 - Blackwashed

13 - Blackwashed

Complete Count:

    6/13 - Blackwashed
    3/13 - Whitewashed
    3/13 - Satisfactory
    1/13 - Miscellaneous issues

### Next Steps:
 - Filestructure adaptation
 - Ensuring runability from Github. 
 - Using a CNN to segment each image. 

## Toybox 

### Visualization 

### PCA
This file does clustering. It outputs several graphs that cluster all 30,000+ datapoints into 7 clusters based on PCA of columns 13-111. It outputs one 2d graph and one 3d graph that show clustering based on the PC1, PC2, and PC3 axes.

## CNN Classifier
**File**: 7000_NN_classifier_based_on_manual_classification.ipynb/.py
The algorithm takes a pickle file titled manual_classification.pkl containing 7304 rows and 5 columns (class, noise, path, dir, seed) as input. It unzips zip files within content containing images. The algorithm then checks for non-int/missing class values as well as paths that do not exist. Presently, no such paths or class values exist. 

### # of Images Within Each Directory
Images1: 3728 Images
Images2: 837 Images
Images3: 2742 Images
Total: 7307 Images

### Directory Name Change Steps:
1. Change all folder names within directory
2. Change all folder names within python notebooks and files
3. Change all folder names within ALL pickle files

### Schema
Generated_Images_7-2-24 --> Images1.zip

Generated_Images --> Images2.zip

05-09-2024_all_images.zip --> Images3.zip

real_images_6-29-25.zip --> RealImages.zip


