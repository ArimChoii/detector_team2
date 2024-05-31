from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
import fnmatch
import os
import librosa
import numpy as np
import keras

# Constants
INPUT_DIR = "archive/"
SAMPLE_RATE = 16000
MAX_SOUND_CLIP_DURATION = 12
CLASSES = ['artifact', 'murmur', 'normal']
NB_CLASSES = len(CLASSES)
label_to_int = {k: v for v, k in enumerate(CLASSES)}
int_to_label = {v: k for k, v in label_to_int.items()}
best_model_file = "./best_model_trained2.hdf5"
MAX_PATIENT = 12
seed = 1

# Function to load file data
def load_file_data(folder, file_names, duration=12, sr=16000):
    input_length = sr * duration
    data = []
    for file_name in file_names:
        try:
            sound_file = folder + file_name
            X, sr = librosa.load(sound_file, sr=sr, duration=duration, res_type='kaiser_fast')
            dur = librosa.get_duration(y=X, sr=sr)
            if round(dur) < duration:
                X = librosa.util.fix_length(X, input_length)
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40).T, axis=0)
        except Exception as e:
            print(f"Error encountered while parsing file: {file_name}")
        data.append(np.array(mfccs).reshape([-1, 1]))
    return data

# Load dataset A
A_folder = INPUT_DIR + '/set_a/'
A_artifact_files = fnmatch.filter(os.listdir(A_folder), 'artifact*.wav')
A_normal_files = fnmatch.filter(os.listdir(A_folder), 'normal*.wav')
A_extrahls_files = fnmatch.filter(os.listdir(A_folder), 'extrahls*.wav')
A_murmur_files = fnmatch.filter(os.listdir(A_folder), 'murmur*.wav')
A_unlabelledtest_files = fnmatch.filter(os.listdir(A_folder), 'Aunlabelledtest*.wav')

A_artifact_sounds = load_file_data(A_folder, A_artifact_files, MAX_SOUND_CLIP_DURATION)
A_normal_sounds = load_file_data(A_folder, A_normal_files, MAX_SOUND_CLIP_DURATION)
A_extrahls_sounds = load_file_data(A_folder, A_extrahls_files, MAX_SOUND_CLIP_DURATION)
A_murmur_sounds = load_file_data(A_folder, A_murmur_files, MAX_SOUND_CLIP_DURATION)
A_unlabelledtest_sounds = load_file_data(A_folder, A_unlabelledtest_files, MAX_SOUND_CLIP_DURATION)

A_artifact_labels = [0 for _ in A_artifact_sounds]
A_normal_labels = [2 for _ in A_normal_sounds]
A_extrahls_labels = [1 for _ in A_extrahls_sounds]
A_murmur_labels = [1 for _ in A_murmur_sounds]
A_unlabelledtest_labels = [-1 for _ in A_unlabelledtest_sounds]

# Load dataset B
B_folder = INPUT_DIR + '/set_b/'
B_normal_files = fnmatch.filter(os.listdir(B_folder), 'normal*.wav')
B_murmur_files = fnmatch.filter(os.listdir(B_folder), 'murmur*.wav')
B_extrastole_files = fnmatch.filter(os.listdir(B_folder), 'extrastole*.wav')
B_unlabelledtest_files = fnmatch.filter(os.listdir(B_folder), 'Bunlabelledtest*.wav')

B_normal_sounds = load_file_data(B_folder, B_normal_files, MAX_SOUND_CLIP_DURATION)
B_murmur_sounds = load_file_data(B_folder, B_murmur_files, MAX_SOUND_CLIP_DURATION)
B_extrastole_sounds = load_file_data(B_folder, B_extrastole_files, MAX_SOUND_CLIP_DURATION)
B_unlabelledtest_sounds = load_file_data(B_folder, B_unlabelledtest_files, MAX_SOUND_CLIP_DURATION)

B_normal_labels = [2 for _ in B_normal_sounds]
B_murmur_labels = [1 for _ in B_murmur_sounds]
B_extrastole_labels = [1 for _ in B_extrastole_sounds]
B_unlabelledtest_labels = [-1 for _ in B_unlabelledtest_sounds]

# Combine datasets
x_data = np.concatenate((A_artifact_sounds, A_normal_sounds, A_extrahls_sounds, A_murmur_sounds,
                         B_normal_sounds, B_murmur_sounds, B_extrastole_sounds))
y_data = np.concatenate((A_artifact_labels, A_normal_labels, A_extrahls_labels, A_murmur_labels,
                         B_normal_labels, B_murmur_labels, B_extrastole_labels))

test_x = np.concatenate((A_unlabelledtest_sounds, B_unlabelledtest_sounds))
test_y = np.concatenate((A_unlabelledtest_labels, B_unlabelledtest_labels))

# Split data into Train, Validation, and Test sets
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.9, random_state=seed, shuffle=True)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.9, random_state=seed, shuffle=True)

# One-Hot encoding for classes
y_train = np.array(keras.utils.to_categorical(y_train, NB_CLASSES))
y_test = np.array(keras.utils.to_categorical(y_test, NB_CLASSES))
y_val = np.array(keras.utils.to_categorical(y_val, NB_CLASSES))
test_y = np.array(keras.utils.to_categorical(test_y, NB_CLASSES))

# Build LSTM RNN model
model = Sequential()
model.add(LSTM(units=64, dropout=0.05, recurrent_dropout=0.20, return_sequences=True, input_shape=(40, 1)))
model.add(LSTM(units=32, dropout=0.05, recurrent_dropout=0.20, return_sequences=False))
model.add(Dense(NB_CLASSES, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='Adamax', metrics=['accuracy'])
model.summary()

# Callbacks
callbacks = [
    ReduceLROnPlateau(patience=MAX_PATIENT, verbose=1),
    ModelCheckpoint(filepath=best_model_file, monitor='val_loss', verbose=1, save_best_only=True)
]

# Evaluate model
print(f"Model train data score: {round(model.evaluate(x_train, y_train, verbose=0)[1] * 100)}%")
print(f"Model test data score: {round(model.evaluate(x_test, y_test, verbose=0)[1] * 100)}%")
print(f"Model validation data score: {round(model.evaluate(x_val, y_val, verbose=0)[1] * 100)}%")
print(f"Model unlabeled data score: {round(model.evaluate(test_x, test_y, verbose=0)[1] * 100)}%")

# Predict and plot results
predicted_classes_label = np.argmax(model.predict(x_test), axis=1)

for i in range(len(x_test)):
    predicted_classes = int_to_label[predicted_classes_label[i]]
    print(f"Sample {i+1}: Predicted Class = " + predicted_classes)

# Reload the best model and evaluate
model.load_weights(best_model_file)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(f"Model evaluation accuracy after loading best model: {round(model.evaluate(x_test, y_test, verbose=0)[1] * 100)}%")


