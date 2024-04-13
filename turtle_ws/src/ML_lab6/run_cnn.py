from image_utils import *

imageDirs = ['2024Simgs/','2023Fimgs/','S2023_imgs/','2022Fimgs/','2022Fheldout/']
# imageDirs = ['2023Fimgs/']
             
test_data = None
train_data = None
val_data = None

for imdir in imageDirs:
    test_trials, val_trials, train_trials = get_dataset(imdir)
    
    if test_data is None:
        test_data = test_trials
        val_data = val_trials
        train_data = train_trials
    else:
        test_data = np.concatenate((test_data, test_trials))
        val_data = np.concatenate((val_data, val_trials))
        train_data = np.concatenate((train_data, train_trials))
    
print(f'test: {len(test_data)}')
print(f'validation: {len(val_data)}')
print(f'train: {len(train_data)}')

num_classes = len(np.unique(train_data[:,1]))
print(f'classes: {num_classes}')
print(test_data[0][0].shape)

# train CNN
model = train_cnn(train_data, val_data)

# test on test_data
test_model(test_data, model)
