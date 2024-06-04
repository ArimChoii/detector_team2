import sys
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects
import numpy as np
import librosa
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QProgressBar, \
    QHBoxLayout
from PyQt5.QtCore import Qt, QThread, pyqtSignal


def cosine_similarity(y_true, y_pred):
    return tf.keras.losses.cosine_similarity(y_true, y_pred)


# 사용자 정의 객체 등록
get_custom_objects().update({'cosine_similarity': cosine_similarity})


def load_custom_model(model_path):
    # 모델 로드
    model = load_model(model_path, custom_objects={'cosine': cosine_similarity})
    return model


def preprocess_wav(file_path, duration=12, sr=16000):
    # WAV 파일 로드
    data = []
    input_length = sr * duration

    X, sr = librosa.load(file_path, sr=sr)
    dur = librosa.get_duration(y=X, sr=sr)

    if round(dur < duration):
        X = librosa.util.fix_length(X, size=input_length)
    mfcc = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40).T, axis=0)

    # (40, 1) 형식으로 변환
    mfcc = np.expand_dims(np.mean(mfcc.T, axis=0), axis=-1)

    return mfcc


class ClassificationThread(QThread):
    progress = pyqtSignal(int)
    result = pyqtSignal(str)

    def __init__(self, model, wav_file):
        super().__init__()
        self.model = model
        self.wav_file = wav_file

    def run(self):
        self.progress.emit(10)  # 예시로 10% 진행
        data = preprocess_wav(self.wav_file)
        self.progress.emit(50)  # 예시로 50% 진행
        prediction = self.model.predict(np.array([data]))
        self.progress.emit(90)  # 예시로 90% 진행
        class_idx = np.argmax(prediction)
        class_idx_mapping = {0: 'Artifact', 1: 'Murmur', 2: 'Normal'}
        class_name = class_idx_mapping.get(class_idx, 'Unknown')
        self.result.emit(f"Predicted class: {class_name}")
        self.progress.emit(100)  # 예시로 100% 진행


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'WAV File Classifier'
        self.wav_file_path = None
        self.model_file_path = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(100, 100, 400, 200)

        layout = QVBoxLayout()

        self.label = QLabel('Select a WAV file and model file to classify', self)
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        self.button_select_wav = QPushButton('Select WAV File', self)
        self.button_select_wav.clicked.connect(self.open_wav_file)
        layout.addWidget(self.button_select_wav)

        self.button_select_model = QPushButton('Select Model File', self)
        self.button_select_model.clicked.connect(self.open_model_file)
        layout.addWidget(self.button_select_model)

        self.progress = QProgressBar(self)
        layout.addWidget(self.progress)

        self.result_label = QLabel('Result will be shown here', self)
        self.result_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.result_label)

        self.button_confirm = QPushButton('Confirm', self)
        self.button_confirm.clicked.connect(self.reset)
        layout.addWidget(self.button_confirm)

        self.button_exit = QPushButton('Exit', self)
        self.button_exit.clicked.connect(self.close)
        layout.addWidget(self.button_exit)

        self.setLayout(layout)

        self.button_confirm.setFixedWidth(100)
        self.button_exit.setFixedWidth(100)

    def open_wav_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_path, _ = QFileDialog.getOpenFileName(self, "Select WAV File", "", "WAV Files (*.wav);;All Files (*)",
                                                   options=options)
        if file_path:
            self.wav_file_path = file_path
            self.label.setText(f'Selected WAV file: {file_path}')

    def open_model_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", "HDF5 Files (*.hdf5);;All Files (*)",
                                                   options=options)
        if file_path:
            self.model_file_path = file_path
            self.label.setText(f'Selected Model file: {file_path}')
            self.run_classification()

    def run_classification(self):
        if hasattr(self, 'wav_file_path') and hasattr(self, 'model_file_path'):
            self.model = load_custom_model(self.model_file_path)
            self.classification_thread = ClassificationThread(self.model, self.wav_file_path)
            self.classification_thread.progress.connect(self.update_progress)
            self.classification_thread.result.connect(self.show_result)
            self.classification_thread.start()

    def update_progress(self, value):
        self.progress.setValue(value)

    def show_result(self, result):
        self.result_label.setText(result)

    def reset(self):
        # 초기 상태로 되돌리기
        self.label.setText('Select a WAV file and model file to classify')
        self.progress.setValue(0)
        self.result_label.setText('Result will be shown here')
        self.wav_file_path = None
        self.model_file_path = None

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())
