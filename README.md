# Age prediction from front and side images
 Our goal was to predict the age from different angels photos, and check if the predictions can be improved if we use two images at the same time. We achived


# Dataset description

For training and testing we used the Kaggle IDOC (Illinois Department of Corrections), which uses mugshots from Illionis Individual in Custody Search, and tabular data with rich annotations from Illionis Prison Population Data Sets. The dataset contains individuals who were in prison at the end of 2018, and we extended the dataset with individuals who were in prison in July of 2024, and filter them to people whose admission date is between 2015 and 2018, or between 2021 and 2024 to make sure that the data do net contains images with inaccurate age. The dataset (Prisoner dataset hereafter) contains two photographic portraits of each subject from the shoulders up with plain background (i.e., “mug shots”), one front-view photo, and one side-view photo. The photos were taken after the person was placed under arrest. The dataset consists of high-quality face images from 54,295 individuals, of whom 50,713 are male, 3,582 are female. The distribution of ethnicity was: 66 of Amer Indian, 161 of Asian, 119 of Bi-racial, 29,423 of Black, 6,012 of Hispanic, 18,454 of White, and 60 Unknown ethnicity (Fig. 1b). The youngest age was 17 years and the oldest age was 83 years.

# Training the models

We split the data into training, validation, and testing sets by using 0.6, 0.2, and 0.2 ratio (n = 32,577, 10,859, and 10,859), respectively. All of the images have been cropped using Retinaface face detector followed by padding to a square and resized to 224x244 pixels. Then we trained a Res-Net 50 model on the training dataset by modifying the final layer to a single-layer neuron. We trained separate models using only front face photos (Front model), only side view face photos (Side model), and both front and side view face photos pictures (Front + Side model).

 # Models

 We have three models, one using frontal images, side images, and the third using both images.



|Model name|Model file|
|:--:|:--:|:--:|
|Front|front.pt|

|Side|side.pt|

|Front_Side|dual.pt|



