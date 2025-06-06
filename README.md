# PatternsFormation

### Questions I still have
 - What is a feret?
 - What is integrational density? How do you calculate it?
 - What is aspect ratio? isn't the aspect ratio the same if we use the smae mesh?
 - What is solidity?
 - What inversion function was used to convert from value to value_inverse (eg. mean to mean_inverted)?
        - my hypothosis that it was literally an inverse function (ie. reflect across y = x)
 - For inverted mean values, do we first find the mean and then invert it or do we first invert all values and find the mean of those. 
 - **NOTE: In order to better understand the relationships between to variables I plotted their values against each other (eg. mean and mean_inverted)** 
 - What is the point of inversion? How does that contribute to the project?

## TODO (for Tim)
 - Download ImageJ to undertand the actual parameters (eg. Feret_mean).
 - Use analyze menu
 - Do visualizations from PowerPoint


### Average Coverage in Each Cluster 
Cluster 0 = Mean 0

Cluster 1 = Mean 255

Cluster 2 = Mean .006 - 189

Cluster 3 = Mean 59.351 - 141.68

Cluster 4 = 11.892 - 254.987

Cluster 5 = 110.332 - 252.66

Cluster 6 = 12.667 - 252.998

### All column names and meanings
**Ua** - Degradation rate for NEGAN

**Ui** - Degradation rate for RTO

**Ga** - Self-activation of A by A

**Gi** - Activation of I by A

**Ba** - Inhibition of A by I

**Da** - Diffusion rate of A

**Di** - Diffusion rate of I

**pattern** - Classified manually. Either 1 or 0.

**noise** - 0 - 3693 with every integer in between being represented. 

**path** - Path to the file

**seed** - ???. Ranges from 3-23352.

**dir** - Directory where the file is found

**feat_bool** - Feature boolean. ???. 

**num_spots** - Number of spots

**Mean** - Mean coverage in the figure. Can range from 0 (completely blank) to 255 (completely filled)

**Median** - Median coverage in the figure. Can range from 0 (completely blank) to 255 (completely filled)

**Area_mean** - Average area of detected objects (maybe spots). Ranges from 1.0 to 40000.0. Max area is completely filled in. Grid is 200 by 200.

**Area_std** - Standard Deviation of area

**X_mean** - Average X coordinate of spots (?)

**X_std** - Standard deviation of X values

**Y_mean** - Average Y coordinates of spots (?)

**Y_std** - Standard deviation of Y values

**Perim._mean** - Mean of perimenter of spots

**Perim._std** - Standard deviation of perimeter of spots

**BX_mean** - ???

**BX_std** - ???

**BY_mean** - ???

**BY_std** - ???

**Width_mean** - Mean of spot width

**Width_std** - Spot width standard deviation

**Height_mean** - Mean height of spot

**Height_std** - Standard deviation of spot height

**Major_mean** - ???

**Major_std** - ???

**Minor_mean** - ???

**Minor_std** - ???

**Angle_mean** - Mean of an agle (what angle??)

**Angle_std** - Standard deviation of the mean ^^

**Circ._mean** - Mean of (???) circumference 

**Circ._std** - Standard deviation of circumference

**Feret_mean** - Mean of feret diameter (distance between two parallel lines tangent to its boundary)

**Feret_std** - Standard deviation of feret diameter

**IntDen_mean** - Average integrated density (sum of pixel values) in object

**IntDen_std** - Standard deviation of integrated density

**%Area_mean** - ???

**%Area_std** - ???

**RawIntDen_mean** - Average raw intensity 

**RawIntDen_std** - Raw intensity standard deviation. (Raw intensity = sum of all pixel areaas)

**FeretX_mean** - Mean X Coordinate of the feret

**FeretX_std** - Standard deviation of the X coordinate of the feret

**FeretY_mean** - Mean Y coordinate of the feret

**FeretY_std** - Standard deviation of the Y coordinate of the feret

**FeretAngle_mean** - Mean feret (?) angle

**FeretAngle_std** - Standard deviation of the feret angle

**MinFeret_mean** - Average minimum feret diameter

**MinFeret_std** - Minimum feret diameter standard deviation

**AR_mean** - Average aspect ratio

**AR_std** - Aspect ratio standard deviation (?)

**Round_mean** - Average roundness 

**Roundness = 4 · Area / (π · (Major axis)²))**

**Round_std** - Roundness standard deviation

**Solidity_mean** - Average solidity (what is solidity?). Rnages from 0.4074 to 1.0.

**Solidity_std** - Solidity standard variation.

**num_spots_inverted** - Num_spots is inverted through some kind of function similair to a / (x + b) where a and b are constants.

**Mean_inverted** - Mean coverage inverted through an equation (see num_spots_inverted)

**Median_inverted** - Median coverage inverted (see above).

**Area_inverted_mean** - Average area of detected objects inverted. Ranges from 1.0 to 40,000 as Area_mean.

**Area_inverted_std** - Area inverted standard deviation.

**X_inverted_mean** - ???

**X_inverted_std** - ???

**Y_inverted_mean** - ???

**Y_inverted_std** - ???

**Perim._inverted_mean** - Mean of perimeter inverted.

**Perim._inverted_std** - Standard deviation of perimeter inverted.

**BX_inverted_mean** - Inverted BX - value

**BX_inverted_std** - ?? I cannot determine what BX_inverted is if I don't know what BX means.

**BY_inverted_mean** - See above.

**BY_inverted_std** - See above.

**Width_inverted_mean** - Inverted width values. Could be y = x because the graph . 1.667 to 34.

**Width_inverted_std** - Standard deviation values of widths.

**Height_inverted_mean** - Mean of inverted height-of-spots values. 

**Height_inverted_std** - Standard deviation within the set of inverted height-of-spots values.

**Major_inverted_mean** - Inverted values of "major". Meaning of "Major_mean" is unkown, therefore, Major_inverted_mean cannot be fully realized.

**Major_inverted_std** - See above.

**Minor_inverted_mean** - See above

**Minor_inverted_std** - See above.

**Angle_inverted_mean** - See above.

**Angle_inverted_std** - See above.

**Circ._inverted_mean** - Mean of inverted circumference (of spots?) values

**Circ._inverted_std** - Standard deviation of inverted circumference (of spots?) values

**Feret_inverted_mean** - The inverted values of the mean of the feret diameter (distance between two parallel lines tangent to its boundary)

**Feret_inverted_std** - The inverted values of the standard deviation of the feret diameter (distance between two parallel lines tangent to its boundary)

**IntDen_inverted_mean** - Average integrated density 

**IntDen_inverted_std** - Standard deviation of integrated density.

**%Area_inverted_mean** - Cannot be determined.

**%Area_inverted_std** - Cannot be determined.

**RawIntDen_inverted_mean** - The mean of the inverted raw integrated density values in the imamge.

**RawIntDen_inverted_std** - The standard deviation of the raw integrated density values in the image.

**FeretX_inverted_mean** - Mean of inverted feret x-coordinates.

**FeretX_inverted_std** - Standard deviation of of inverted feret x-coordinates.

**FeretY_inverted_mean** - Mean of inverted feret y-coordinates.

**FeretY_inverted_std** - Standard deviation of inverted feret y-coordinates.

**FeretAngle_inverted_mean** - Cannot be fully determined as I do not know what FeretAngle is.

**FeretAngle_inverted_std** - See above.

**MinFeret_inverted_mean** - Mean of the inverted values of the minimum feret diameters in the image.

**MinFeret_inverted_std** - Standard deviation of the inverted values of feret diameters in the image.

**AR_inverted_mean** - Mean of inverted aspect ratio values.

**AR_inverted_std** - Standard dev. of inverted aspect ratio values.

**Round_inverted_mean** - Mean of inverted roundness values. 

**Round_inverted_std** - Standard deviation of inverted roundness values.

**Solidity_inverted_mean** - Mean of inverted solidity values.

**Solidity_inverted_std** - Standard deviation of inverted solidity values.

**full_path** - The full path to the file

**classifier_pred_class** - Manually classified through a classifier. How the classifier works is unkown. 
