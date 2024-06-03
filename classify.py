import sys
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects
import numpy as np
import librosa
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QProgressBar
from PyQt5.QtCore import Qt, QThread, pyqtSignal

def cosine_similarity(y_true, y_pred):
    return tf.keras.losses.cosine_similarity(y_true, y_pred)

# 사용자 정의 객체 등록
get_custom_objects().update({'cosine_similarity': cosine_similarity})

def load_custom_model(model_path):
    # 모델 로드
    model = load_model(model_path)
    return model

def preprocess_wav(file_path):
    # WAV 파일 로드
    y, sr = librosa.load(file_path, sr=16000)
    
    # MFCC 특징 추출
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    
    # 입력 데이터의 형식을 (None, 40, 1)로 변환
    mfcc = np.expand_dims(mfcc.T, axis=-1)
    
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
        prediction = self.model.predict(data)
        self.progress.emit(90)  # 예시로 90% 진행
        class_idx = np.argmax(prediction)
        self.result.emit(f"Predicted class: {class_idx}")
        self.progress.emit(100)  # 예시로 100% 진행

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'WAV File Classifier'
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

        self.setLayout(layout)

    def open_wav_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_path, _ = QFileDialog.getOpenFileName(self, "Select WAV File", "", "WAV Files (*.wav);;All Files (*)", options=options)
        if file_path:
            self.wav_file_path = file_path
            self.label.setText(f'Selected WAV file: {file_path}')

    def open_model_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", "HDF5 Files (*.hdf5);;All Files (*)", options=options)
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

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())