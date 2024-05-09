import matplotlib.pyplot as plt
import IPython.display as ipd
import wave
from scipy.io import wavfile
import os
import fnmatch
import pandas as pd
import librosa
import librosa.display
import glob
import numpy as np

import tensorflow as tf
print('tensorflow version: ', tf.__version__)

import keras
print('keras version: ', keras.__version__)


# [Important] Version info #
# python = 3.6
# tensorflow = 1.12.0
# numpy	= 1.16.6
# librosa = 0.6.3
# h5py = 2.10.0


# simple encoding of categories, limited to 3 types
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import warnings                        # To ignore any warnings
warnings.filterwarnings("ignore")

INPUT_DIR = "archive/"
SAMPLE_RATE = 16000
MAX_SOUND_CLIP_DURATION = 12

# # Check Dataset # #
set_a = pd.read_csv(INPUT_DIR+"set_a.csv")
print(set_a.head())     # .head() : 초반의 몇 개의 행만 확인

set_a_timing = pd.read_csv(INPUT_DIR+"/set_a_timing.csv")
print(set_a_timing.head())

set_b = pd.read_csv(INPUT_DIR+"/set_b.csv")
print(set_b.head())

# merge both set-a and set-b
frames = [set_a, set_b]
train_ab = pd.concat(frames)
print(train_ab.describe())

# get all unique labels (find classes)
nb_classes = train_ab.label.unique()    # label 행을 불러와서 대표 label들을 불러옴 = classes
print("Number of training examples=", train_ab.shape[0], "  Number of classes=", len(train_ab.label.unique()))
print(nb_classes)

# visualize data distribution by category
category_group = train_ab.groupby(['label', 'dataset']).count()
plot = category_group.unstack().reindex(category_group.unstack().sum(axis=1).sort_values().index).plot(kind='bar', stacked=True, title="Number of Audio Samples per Category", figsize=(16, 10))
plot.set_xlabel("Category")
plot.set_ylabel("Samples Count")
# plt.show()

print('Min samples per category = ', min(train_ab.label.value_counts()))
print('Max samples per category = ', max(train_ab.label.value_counts()))


# Normal case #
# Healthy heart sounds
# These may contain noise in the final second of the recording as the device is removed from body.
# They may contain a variety of background noises (from traffic to radios).
#  "lub dub, lub dub" pattern, with the time from "lub" to "dub" shorter than the time from "dub" to the next "lub"

normal_file = INPUT_DIR + "/set_a/normal__201106111136.wav"

# heart it
ipd.Audio(normal_file)

# Load use wave
wav = wave.open(normal_file)
print("Sampling (frame) rate = ", wav.getframerate())
print("Total samples (frames) = ", wav.getnframes())
print("Duration = ", wav.getnframes()/wav.getframerate())

# Load use scipy
rate, data = wavfile.read(normal_file)
print("Sampling (frame) rate = ", rate)
print("Total samples (frames) = ", data.shape)
print(data)

# plot wave by audio frames
plt.figure(figsize=(16, 10))
plt.plot(data, '-', )
plt.plot()

# Load using Librosa
y, sr = librosa.load(normal_file, duration=5)   # default sampling rate is 22 HZ
dur = librosa.get_duration(y)
print("duration:", dur)
print(y.shape, sr)

# librosa plot
plt.figure(figsize=(16, 10))
librosa.display.waveplot(y, sr=sr)


# Murmur #
#

murmur_file = INPUT_DIR+"/set_a/murmur__201108222231.wav"
y2, sr2 = librosa.load(murmur_file, duration=5)
dur = librosa.get_duration(y)
print("duration:", dur)
print(y2.shape, sr2)

# heart it
ipd.Audio(murmur_file)

# show it
plt.figure(figsize=(16, 10))
librosa.display.waveplot(y2, sr=sr2)


# Extrasystole #
# "lub-lub dub" or a "lub dub-dub"
# It may not be a sign of disease, however, in some situations extrasystoles can be caused by heart diseases.

extrastole_file = INPUT_DIR + "/set_b/extrastole__127_1306764300147_C2.wav"
y3, sr3 = librosa.load(extrastole_file, duration=5)
dur = librosa.get_duration(y)
print("duration:", dur)
print(y3.shape, sr3)

# heart it
ipd.Audio(extrastole_file)

# show it
plt.figure(figsize=(16, 10))
librosa.display.waveplot(y3, sr=sr3)


# Artifact #
# A wide range of different sounds, including feedback squeals and echoes, speech, music and noise

artifact_file = INPUT_DIR + "/set_a/artifact__201012172012.wav"
y4, sr4 = librosa.load(artifact_file, duration=5)
dur = librosa.get_duration(y)
print("duration:", dur)
print(y4.shape, sr4)

# heart it
ipd.Audio(artifact_file)

# show it
plt.figure(figsize=(16, 10))
librosa.display.waveplot(y4, sr=sr4)


# Extra Heart Sound #
extrahls_file = INPUT_DIR+"/set_a/extrahls__201101070953.wav"
y5, sr5 = librosa.load(extrahls_file, duration=5)
dur = librosa.get_duration(y)
print("duration:", dur)
print(y5.shape, sr5)

# heart it
ipd.Audio(extrahls_file)

# show it
plt.figure(figsize=(16, 10))
librosa.display.waveplot(y5, sr=sr5)


# Sound Feature: MFCC (Mel Frequency Cepstral Coefficient) #
# mfccs = librosa.feature.mfcc(y=y, sr=sr)
# print(mfccs)

# Use a pre-computed log-power Mel spectrogram
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
log_S = librosa.feature.mfcc(S=librosa.power_to_db(S))
print(log_S)

# Get more components
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
print(mfccs)

# Visualize the MFCC series
# Mel-frequency cepstral coefficients (MFCCs)
plt.figure(figsize=(12, 3))
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('Mel-frequency cepstral coefficients (MFCCs)')
plt.tight_layout()

# Compare different DCT bases
m_slaney = librosa.feature.mfcc(y=y, sr=sr, dct_type=2)

# m_dct1 = librosa.feature.mfcc(y=y, sr=sr, dct_type=1)
plt.figure(figsize=(12, 6))
# plt.subplot(3, 1, 1)
# librosa.display.specshow(m_dct1, x_axis='time')
# plt.title('Discrete cosine transform (dct_type=1)')
# plt.colorbar()
m_htk = librosa.feature.mfcc(y=y, sr=sr, dct_type=3)
plt.subplot(3, 1, 2)
librosa.display.specshow(m_slaney, x_axis='time')
plt.title('RASTAMAT / Auditory toolbox (dct_type=2)')
plt.colorbar()
plt.subplot(3, 1, 3)
librosa.display.specshow(m_htk, x_axis='time')
plt.title('HTK-style (dct_type=3)')
plt.colorbar()
plt.tight_layout()


# Sound Feature: Onset
# Picking peaks in an onset strength envelope

# Get onset times from a signal
onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
librosa.frames_to_time(onset_frames, sr=sr)

# use a pre-computed onset envelope
o_env = librosa.onset.onset_strength(y, sr=sr)
times = librosa.frames_to_time(np.arange(len(o_env)), sr=sr)
onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr)

# visualize it
D = np.abs(librosa.stft(y))
plt.figure(figsize=(16, 6))
ax1 = plt.subplot(2, 1, 1)
librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),x_axis='time', y_axis='log')
plt.title('Power spectrogram')
plt.subplot(2, 1, 2, sharex=ax1)

plt.plot(times, o_env, label='Onset strength')
plt.vlines(times[onset_frames], 0, o_env.max(), color='r', alpha=0.9,linestyle='--', label='Onsets')
plt.axis('tight')
plt.legend(frameon=True, framealpha=0.75)

# Onset Backtrack
# Backtrack detected onset events to the nearest preceding local minimum of an energy function.

oenv = librosa.onset.onset_strength(y=y, sr=sr)

# Detect events without backtracking
onset_raw = librosa.onset.onset_detect(onset_envelope=oenv, backtrack=False)

# Backtrack the events using the onset envelope
onset_bt = librosa.onset.onset_backtrack(onset_raw, oenv)

# Backtrack the events using the RMS values
rms = librosa.feature.rms(S=np.abs(librosa.stft(y=y)))
onset_bt_rms = librosa.onset.onset_backtrack(onset_raw, rms[0])

# Plot the results
plt.figure(figsize=(16, 6))
plt.subplot(2, 1, 1)
plt.plot(oenv, label='Onset strength')
plt.vlines(onset_raw, 0, oenv.max(), label='Raw onsets')
plt.vlines(onset_bt, 0, oenv.max(), label='Backtracked', color='r')
plt.legend(frameon=True, framealpha=0.75)
plt.subplot(2, 1, 2)
plt.plot(rms[0], label='RMS')
plt.vlines(onset_bt_rms, 0, rms.max(), label='Backtracked (RMS)', color='r')
plt.legend(frameon=True, framealpha=0.75)

# Onset Strength
# Compute a spectral flux onset strength envelope

D = np.abs(librosa.stft(y))
times = librosa.frames_to_time(np.arange(D.shape[1]))

plt.figure(figsize=(16, 6))
# ax1 = plt.subplot(2, 1, 1)
# librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),y_axis='log', x_axis='time')
# plt.title('Power spectrogram')

# Construct a standard onset function
onset_env = librosa.onset.onset_strength(y=y, sr=sr)
plt.subplot(2, 1, 1, sharex=ax1)
plt.plot(times, 2 + onset_env / onset_env.max(), alpha=0.8,label='Mean (mel)')

# median
onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median,fmax=8000, n_mels=256)
plt.plot(times, 1 + (onset_env/onset_env.max()), alpha=0.8,label='Median (custom mel)')

# Constant-Q spectrogram instead of Mel
onset_env = librosa.onset.onset_strength(y=y, sr=sr, feature=librosa.cqt)
plt.plot(times, onset_env / onset_env.max(), alpha=0.8, label='Mean (CQT)')
plt.legend(frameon=True, framealpha=0.75)
plt.ylabel('Normalized strength')
plt.yticks([])
plt.axis('tight')
plt.tight_layout()

onset_subbands = librosa.onset.onset_strength_multi(y=y, sr=sr, channels=[0, 32, 64, 96, 128])
# plt.figure(figsize=(16, 6))
plt.subplot(2, 1, 2)
librosa.display.specshow(onset_subbands, x_axis='time')
plt.ylabel('Sub-bands')
plt.title('Sub-band onset strength')

# plt.show()



# # Load Dataset # #
print("Number of training examples=", train_ab.shape[0], "  Number of classes=", len(train_ab.label.unique()))

def audio_norm(data):
    max_data = np.max(data)
    min_data = np.min(data)
    data = (data - min_data) / (max_data - min_data + 0.0001)
    return data - 0.5

# get audio data without padding highest qualify audio (3s)
def load_file_data_wo_change(folder, file_names, duration=3, sr=16000):
    input_length = sr * duration
    # file_names = glob.glob(os.path.join(folder, '*.wav'))
    data = []

    for file_name in file_names:
        try:
            sound_file = folder + file_name
            print("load file  ", sound_file)
            X, sr = librosa.load(sound_file, sr=sr, duration=duration, res_type='kaiser_fast')  # for faster extraction
            dur = librosa.get_duration(y=X, sr=sr)

            if round(dur) < duration:
                print("fixing audio length :  ", file_name)
                y = librosa.util.fix_length(X, input_length)

            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40).T, axis=0)

        except Exception as e:
            print("Error encountered while parsing file : ", file_name)

        feature = np.array(mfccs).reshape([-1, 1])
        data.append(feature)

    return data

# get audio data with a fix padding may also chop off some file (12s)
def load_file_data(folder, file_names, duration=12, sr=16000):
    input_length = sr*duration
    # function to load files and extract features
    # file_names = glob.glob(os.path.join(folder, '*.wav'))
    data = []

    for file_name in file_names:
        try:
            sound_file = folder + file_name
            print("load file  ", sound_file)
            # use kaiser_fast technique for faster extraction
            X, sr = librosa.load(sound_file, sr=sr, duration=duration, res_type='kaiser_fast')
            dur = librosa.get_duration(y=X, sr=sr)

            # pad audio file same duration
            if round(dur) < duration:
                print("fixing audio length : ", file_name)
                y = librosa.util.fix_length(X, input_length)

            # normalized raw audio
            # y = audio_norm(y)
            # extract normalized mfcc feature from data

            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40).T, axis=0)

        except Exception as e:
            print("Error encountered while parsing file: ", file_name)
        feature = np.array(mfccs).reshape([-1,1])
        data.append(feature)

    return data


# Map label text to integer
CLASSES = ['artifact', 'murmur', 'normal']
# {'artifact': 0, 'murmur': 1, 'normal': 3}
NB_CLASSES = len(CLASSES)

# Map integer value to text labels
label_to_int = {k: v for v, k in enumerate(CLASSES)}
print(label_to_int)
print(" ")

# map integer to label text
int_to_label = {v: k for k, v in label_to_int.items()}
print(int_to_label)

# load dataset-a, keep them separate for testing purpose
A_folder = INPUT_DIR + '/set_a/'

# set-a
A_artifact_files = fnmatch.filter(os.listdir(INPUT_DIR+'/set_a'), 'artifact*.wav')
A_artifact_sounds = load_file_data(folder=A_folder, file_names=A_artifact_files, duration=MAX_SOUND_CLIP_DURATION)
A_artifact_labels = [0 for items in A_artifact_files]

A_normal_files = fnmatch.filter(os.listdir(INPUT_DIR+'/set_a'), 'normal*.wav')
A_normal_sounds = load_file_data(folder=A_folder, file_names=A_normal_files, duration=MAX_SOUND_CLIP_DURATION)
A_normal_labels = [2 for items in A_normal_sounds]

A_extrahls_files = fnmatch.filter(os.listdir(INPUT_DIR+'/set_a'), 'extrahls*.wav')
A_extrahls_sounds = load_file_data(folder=A_folder, file_names=A_extrahls_files, duration=MAX_SOUND_CLIP_DURATION)
A_extrahls_labels = [1 for items in A_extrahls_sounds]

A_murmur_files = fnmatch.filter(os.listdir(INPUT_DIR+'/set_a'), 'murmur*.wav')
A_murmur_sounds = load_file_data(folder=A_folder, file_names=A_murmur_files, duration=MAX_SOUND_CLIP_DURATION)
A_murmur_labels = [1 for items in A_murmur_files]

# test files
A_unlabelledtest_files = fnmatch.filter(os.listdir(INPUT_DIR+'/set_a'), 'Aunlabelledtest*.wav')
A_unlabelledtest_sounds = load_file_data(folder=A_folder, file_names=A_unlabelledtest_files, duration=MAX_SOUND_CLIP_DURATION)
A_unlabelledtest_labels = [-1 for items in A_unlabelledtest_sounds]

print("loaded dataset-a")

# load dataset-b, keep them separate for testing purpose
B_folder = INPUT_DIR+'/set_b/'

# set-b
B_normal_files = fnmatch.filter(os.listdir(INPUT_DIR+'/set_b'), 'normal*.wav')  # include noisy files
B_normal_sounds = load_file_data(folder=B_folder, file_names=B_normal_files, duration=MAX_SOUND_CLIP_DURATION)
B_normal_labels = [2 for items in B_normal_sounds]

B_murmur_files = fnmatch.filter(os.listdir(INPUT_DIR+'/set_b'), 'murmur*.wav')  # include noisy files
B_murmur_sounds = load_file_data(folder=B_folder, file_names=B_murmur_files, duration=MAX_SOUND_CLIP_DURATION)
B_murmur_labels = [1 for items in B_murmur_files]

B_extrastole_files = fnmatch.filter(os.listdir(INPUT_DIR+'/set_b'), 'extrastole*.wav')
B_extrastole_sounds = load_file_data(folder=B_folder, file_names=B_extrastole_files, duration=MAX_SOUND_CLIP_DURATION)
B_extrastole_labels = [1 for items in B_extrastole_files]

# test files
B_unlabelledtest_files = fnmatch.filter(os.listdir(INPUT_DIR+'/set_b'), 'Bunlabelledtest*.wav')
B_unlabelledtest_sounds = load_file_data(folder=B_folder, file_names=B_unlabelledtest_files, duration=MAX_SOUND_CLIP_DURATION)
B_unlabelledtest_labels = [-1 for items in B_unlabelledtest_sounds]

print("loaded dataset-b")

# combine set-a and set-b
x_data = np.concatenate((A_artifact_sounds, A_normal_sounds, A_extrahls_sounds, A_murmur_sounds,
                         B_normal_sounds, B_murmur_sounds, B_extrastole_sounds))

y_data = np.concatenate((A_artifact_labels, A_normal_labels, A_extrahls_labels, A_murmur_labels,
                         B_normal_labels, B_murmur_labels, B_extrastole_labels))

test_x = np.concatenate((A_unlabelledtest_sounds, B_unlabelledtest_sounds))
test_y = np.concatenate((A_unlabelledtest_labels, B_unlabelledtest_labels))

print("combined training data record: ", len(y_data), len(test_y))

# shuffle - whether or not to shuffle the data before splitting. If shuffle=False then stratify must be None.
# random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random.

seed = 1000     # 항상 동일한 결과가 나오도록 함
# split data into Train, Validation and Test
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.9, random_state=seed, shuffle=True)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.9, random_state=seed, shuffle=True)

# One-Hot encoding for classes
y_train = np.array(keras.utils.to_categorical(y_train, len(CLASSES)))
y_test = np.array(keras.utils.to_categorical(y_test, len(CLASSES)))
y_val = np.array(keras.utils.to_categorical(y_val, len(CLASSES)))
test_y = np.array(keras.utils.to_categorical(test_y, len(CLASSES)))

print("label shape: ", y_data.shape)
print("data size of the array: : %s" % y_data.size)
print("length of one array element in bytes: ", y_data.itemsize)
print("total bytes consumed by the elements of the array: ", y_data.nbytes)
print(y_data[1])
print("")
print("audio data shape: ", x_data.shape)
print("data size of the array: : %s" % x_data.size)
print("length of one array element in bytes: ", x_data.itemsize)
print("total bytes consumed by the elements of the array: ", x_data.nbytes)
# print(x_data[1])
print("")
print("training data shape: ", x_train.shape)
print("training label shape: ", y_train.shape)
print("")
print("validation data shape: ", x_val.shape)
print("validation label shape: ", y_val.shape)
print("")
print("test data shape: ", x_test.shape)
print("test label shape: ", y_test.shape)


# # Deep Learning RNN - LSTM # #
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard, ProgbarLogger
from keras.utils import np_utils
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import itertools

print('Build LSTM RNN model ...')
model = Sequential()
model.add(LSTM(units=64, dropout=0.05, recurrent_dropout=0.20, return_sequences=True,input_shape = (40,1)))
model.add(LSTM(units=32, dropout=0.05, recurrent_dropout=0.20, return_sequences=False))
model.add(Dense(len(CLASSES), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='Adamax', metrics=['acc','mse', 'mae', 'mape', 'cosine'])
model.summary()

# 설정값
best_model_file = "./best_model_trained.hdf5"
MAX_PATIENT = 12
MAX_EPOCHS = 100
MAX_BATCH = 32

# 콜백 설정
callbacks = [
    ReduceLROnPlateau(patience=MAX_PATIENT, verbose=1),  # 학습률을 동적으로 조정 : 모델의 성능이 개선되지 않을 때 학습률을 감소시켜 학습을 안정화함
    ModelCheckpoint(filepath=best_model_file, monitor='loss', verbose=1, save_best_only=True)  # 모델의 손실이 가장 낮을 때 모델의 가중치 저장
]

print("training started... please wait.")

# 모델 훈련
history = model.fit(
    x_train, y_train,
    batch_size=MAX_BATCH,
    epochs=MAX_EPOCHS,
    verbose=0,
    validation_data=(x_val, y_val),
    callbacks=callbacks
)

print("training finished!")

# Model Evaluation
score = model.evaluate(x_train, y_train, verbose=0)
print("model train data score       : ", round(score[1]*100), "%")

score = model.evaluate(x_test, y_test, verbose=0)
print("model test data score        : ", round(score[1]*100), "%")

score = model.evaluate(x_val, y_val, verbose=0)
print("model validation data score  : ", round(score[1]*100), "%")

score = model.evaluate(test_x, test_y, verbose=0)
print("model unlabeled data score   : ", round(score[1]*100), "%")

def plot_history(history):
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
    if len(loss_list) == 0:
        print('Loss is missing in history')
        return

    plt.figure(figsize=(22, 10))

    # Accuracy plot
    plt.subplot(221, title='Accuracy')
    epochs = range(1, len(history.history[loss_list[0]]) + 1)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b',
                 label='Training accuracy (' + str(format(history.history[l][-1], '.5f')) + ')')
    for l in val_acc_list:
        plt.plot(epochs, history.history[l], 'g',
                 label='Validation accuracy (' + str(format(history.history[l][-1], '.5f')) + ')')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(222, title='Loss')
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b',
                 label='Training loss (' + str(str(format(history.history[l][-1], '.5f')) + ')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g',
                 label='Validation loss (' + str(str(format(history.history[l][-1], '.5f')) + ')'))
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

# plot history
plot_history(history)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = 'Normalized confusion matrix'
    else:
        title = 'Confusion matrix'

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# Prediction Test

# prediction class
y_pred = model.predict_classes(x_test, batch_size=32)
print("prediction test return :", y_pred[1], "-", int_to_label[y_pred[1]])

plt.figure(1, figsize=(20, 10))
# plot Classification Metrics: Accuracy
plt.subplot(221, title='Prediction')
plt.plot(y_pred)
plt.show()

# Loading a saved training model
print(best_model_file)

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# create model
print('Build LSTM RNN model ...')
model = Sequential()
model.add(LSTM(units=64, dropout=0.05, recurrent_dropout=0.35, return_sequences=True,input_shape = (40,1)))
model.add(LSTM(units=32, dropout=0.05, recurrent_dropout=0.35, return_sequences=False))
model.add(Dense(len(CLASSES), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc','mse', 'mae', 'mape', 'cosine'])
model.summary()

# load weights
model.load_weights(best_model_file)

# Compile model (required to make predictions)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print("Created model and loaded weights from file")


# Test loaded model
# check scores
scores = model.evaluate(x_test, y_test, verbose=0)
print("Model evaluation accuracy: ", round(scores[1]*100), "%")

