from keras.models import load_model
from image_utils import *

imgDir = '2024Simgs/'
grayscale= False

model = load_model('CNN_model.h5')
test_data, lines, ext = get_test_data(imgDir, grayscale=grayscale)

# test on test_data
test_model(lines, imgDir, ext, test_data, model, visualize=True, only_false=True)
