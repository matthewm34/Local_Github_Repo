import argparse

from image_utils import *


def main():
    imageDirs = ['2023Fimgs','S2023_imgs','2022Fimgs','2022Fheldout']
    # imageDirs = ['2024Simgs/']
    grayscale = False
                
    train_data = None

    for imdir in imageDirs:
        train_trials = get_train_data(imdir, val_split=False, grayscale=grayscale)

        if train_data is None:
            train_data = train_trials
        else:
            train_data = np.concatenate((train_data, train_trials))
        
    print(f'train dataset: {len(train_data)}')

    num_classes = len(np.unique(train_data[:,1]))
    print(f'classes: {num_classes}')
    print(f'image (input) shape: {train_data[0][0].shape}')

    # train CNN
    model = train_cnn(train_data, patience=3, grayscale=grayscale)


if __name__ == '__main__':
    main()