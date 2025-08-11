# Improving face age prediction by using multiple angles photos 
 Our goal was to predict the age from different angels photos, and check if the predictions can be improved if we use two images at the same time.


## Dataset description

For training and testing we used the [Kaggle IDOC dataset](https://www.kaggle.com/datasets/davidjfisher/illinois-doc-labeled-faces-dataset), which uses mugshots from [Illionis Department Of Corrections](https://idoc.illinois.gov), and tabular data with rich annotations from [Illionis Prison Population Data Sets](https://idoc.illinois.gov/reportsandstatistics/populationdatasets.html). The full dataset contains more than 50000 individual. The youngest age was 17 years and the oldest age was 83 years.

## Training the models

We split the data into training, validation, and testing sets by using 0.6, 0.2, and 0.2 ratio respectively, and the removing those pictures when multiple pictures are taken of one person. After this the train set contains 32577 images, the validation set contains 9685 images, the test set contains 9189 images. All of the images have been cropped using [Retinaface](https://github.com/serengil/retinaface) face detector followed by padding to a square and resized to 224x244 pixels. Then we trained a Res-Net 50 model on the training dataset by modifying the final layer to a single-layer neuron. We trained separate models using only front face photos (Front model), only side view face photos (Side model), and both front and side view face photos pictures (Front + Side model).

 ## Models

 We have three models, one using frontal images, side images, and the third using both images.

|Model name|Model file|
|-----|-----|
|Front|[front_model.pt](https://drive.google.com/file/d/15mcCJbpumZR5yNyNnVAUV2o6DTb9POSZ/view?usp=sharing)|
|Side|[side_model.pt](https://drive.google.com/file/d/1BY-ulTGRKQighzvIsxqHGqYCH8SXWx5j/view?usp=sharing)|
|Front_Side|[dual_model.pt](https://drive.google.com/file/d/1UFaZDD4xH-SclFQUjdz368XHEb4CmdjC/view?usp=sharing)|

### To use the models make sure you intall the packages from `requiments.txt`, and put the files in the directory strucure shown below:
```bash
├── base_folder/
│   ├── predict_one_image.py
│   ├── predict_from_csv.py
│   ├── models/
│   │   ├── front_model.pt
│   │   ├── side_model.pt
│   │   ├── dual_model.pt 
│ 
```
### Predict one image

To predict one image use `predict_one_image.py`, it will print out the predicted age.:
```
python3 predict_one_image.py direction image_path
```
* Replace `<direction>` with the desired model type: front,side, or dual

* Replace `<image_path>` with the path of your image file.

### Predict images from `.csv` file
To predict batches of images you can use `predict_from_csv.py` , put the path of the images into the ``path`` column:

```
python3 predict_image_csv.py direction csv_file
```
* Replace `<direction>` with the desired model type: front,side, or dual

* Replace `<csv_file>` with the path of your '.csv' file.
* Add front image paths `<front_path>` column in the '.csv' file.
* Add side image paths `<side_path>` column in the '.csv' file.
* The predictions will be saved into `<csv_file>_predict.csv`:



## Authors

* **Botond Bárdos Deák** :  [bdbotond@sztaki.hu](mailto:bdbotond@sztaki.hu)
* **Csaba Kerepesi** : [kerepesi@sztaki.hu](mailto:kerepesi@sztaki.hu)

## Contact

For questions or feedback, please contact Botond Bárdos Deák at [bdbotond@sztaki.hu](mailto:bdbotond@sztaki.hu)
