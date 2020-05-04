import csv
from matplotlib import pyplot as plt
import numpy as np
import sklearn
import os


def get_filename(path):
   '''
   Extract the image filename from the path
   ''' 
   return os.path.basename(path)


def generator(samples, images_path, batch_size=32, steering_correction=0.2):
    batch_labels = []
    batch_images = []
    num_samples = len(samples)
    while True:
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            for row in batch_samples:
                center_img = plt.imread(images_path + get_filename(row[0]))
                center_label = float(row[3])
                batch_images.append(center_img)
                batch_labels.append(center_label)
                
                left_img = plt.imread(images_path + get_filename(row[1]))
                left_label = center_label + steering_correction
                batch_images.append(left_img)
                batch_labels.append(left_label)
                
                right_img = plt.imread(images_path + get_filename(row[2]))
                right_label = center_label - steering_correction
                batch_images.append(right_img)
                batch_labels.append(right_label)
                
                aug_img = np.fliplr(center_img)
                aug_label = -center_label
                batch_images.append(aug_img)
                batch_labels.append(aug_label)

            batch_X = np.array(batch_images)
            batch_Y = np.array(batch_labels)
            
            yield sklearn.utils.shuffle(batch_X, batch_Y)


from keras.models import Sequential
from keras.layers import Lambda, Cropping2D, Conv2D, MaxPooling2D, Flatten, Dense


def setup_model(input_shape):
    model = Sequential()
    
    # preprocessing cropping and normalizing images
    model.add(Cropping2D(cropping=((75, 25), (0, 0)), input_shape=input_shape))
    model.add(Lambda(lambda x: (x / 255.0) - 1.0))
    
    model.add(Conv2D(6, (5, 5), activation="relu"))
    model.add(MaxPooling2D())
    model.add(Conv2D(16, (5, 5), activation="relu"))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120, activation="relu"))
    model.add(Dense(84, activation="relu"))
    model.add(Dense(1))
    
    return model


from sklearn.model_selection import train_test_split

dataset_path = "./driving_data/"
images_path = dataset_path + "IMG/"

# Load the samples from CSV file
samples = []
with open(dataset_path + "driving_log.csv") as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        samples.append(row)

print("Number of rows in CSV file: ", len(samples))

train_samples, validation_samples = train_test_split(samples, test_size=0.2)
print("Number of training rows (augmentation and left-right not included): ", len(train_samples))
print("Number of training samples (augmentation and left-right not included):", len(train_samples)*4)


from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

input_shape = (160, 320, 3)
model = setup_model(input_shape)
model.compile(loss="mse", optimizer="adam")

train_generator = generator(train_samples, images_path)
validation_generator = generator(validation_samples, images_path)

batch_size = 256
epochs = 1

# There len(samples) rows in CSV. Each row has 3 images (center, left, right).
# I also add an augmented image. 
num_train_samples = len(train_samples) * 4
num_validation_samples = len(validation_samples) * 4

model_checkpoint = ModelCheckpoint("./model/checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5", 
                                   verbose=1, save_best_only=True)
early_stopping = EarlyStopping()
csvlogger = CSVLogger("./model/training.log", separator=",", append=False)

history = model.fit_generator(train_generator, 
                    steps_per_epoch=np.ceil(num_train_samples / batch_size), 
                    validation_data=validation_generator, 
                    validation_steps=np.ceil(num_validation_samples / batch_size), 
                    epochs=epochs, verbose=1, 
                    callbacks=[model_checkpoint, early_stopping, csvlogger])


model.save("./model/trained/model.h5")

