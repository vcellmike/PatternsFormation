# Data and algorithms sources for ML analysis

## Images
 - 05-09-2024_all_images.zip contains 15,133 images in png format, generated using VCell software using the same initial distribution of A protein (noise).
 - Generated Images.zip contains 4,523 images generated using VCell software using 5 different initial distributions of A proteins, called noise3 (598 images), ..., noise6 (1,000 images), noise7 (999 images)
 - Generated Images 7-2-24 contains 19,858 images in png format, generated using VCell software using 5 different initial distributions of A proteins, called noise 3 (4,000 images), noise 4 (3,900 images), noise 5 (3,965 images), noise 6 (3,993 images), noise 7 (4,000 images) 

## Algorithms
The algorithm used was a convolutional neural netowrk that classified images into classes by clustering them. A dataframe inside a pickle file for clustering was created that included the filename and class. The algorithm split the data into training, test, and validating data. Finally, a confusion matrix was created to represent the accuracy of the model in classifying an image into the correct class.
