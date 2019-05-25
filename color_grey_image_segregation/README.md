## Getting Started

install the python libraries mentioned in requirements.txt with python 3.x (prefer 3.7)

### Data

Raw images are taken from kaggles's 256_categories dataset. 
Training and testing dataset is being prepared using 'data_prep_kmean_cluster.ipynb'. Before executing the data_prep_kmean_cluster.pynb, create '/training_data/color', '/training_data/gray', '/testing_data/color', and '/testing_data/gray' directories.

### Kmean_clustering

for checking if the data are labeled correctly or not.

### Start the Training of The Model

The model uses tensorflow toolkit with vast opportunity to hypertuning.
To start the training  `train.py`

Once your model is trained, it will save the model files in the current folder.
In order to predict using the trained model, we can run the predict.py code by passing it our test image.
python predict.py testing_data/gray/1110.jpg

