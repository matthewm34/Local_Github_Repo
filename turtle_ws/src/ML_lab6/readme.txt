# CNN Classifier. 

run_cnn.py - This script runs a pretrained CNN model on a specified test set of data. 

There are three ways to run this code:

1) Without visualization of fed images:
python run_cnn.py img_dir

2) With visualization of fed images:
python run_cnn.py img_dir -v

3) With visualization of only wrongly classified images:
python run_cnn.py img_dir -d

A final total accuracy and confusion matrix is printed out as well.
NOTE: img_dir must be a string to the directory containing test images, located in the same folder as the script. No / are required in the name.
ex) python run_cnn.py 2024Simgs