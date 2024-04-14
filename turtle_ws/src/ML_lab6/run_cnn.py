from keras.models import load_model
from image_utils import *

imgDir = '2024Simgs/'

model = load_model('CNN_model.h5')
test_data = get_test_data(imgDir)

# test on test_data
test_model(test_data, model, visualize=True)
