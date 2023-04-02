import os
import copy
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing as p
from tensorflow.keras import layers, models
from tensorflow.math import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn.utils import class_weight

data_path = r"C:\Users\Sharath C R\Documents\GitHub\Shotseeker\Shot-Seeker\gunshot-audio-dataset"

list_folder = os.listdir(data_path)

hop_length = 128 #the default spacing between frames
nfft = 255 #number of samples
n_classes = 9

print("Reading Audio Data")

raw_x = []
raw_y = []
for i in list_folder:
    list_file_wave = os.listdir(os.path.join(data_path,i))
    for j in list_file_wave:
        filename = os.path.join(data_path, i, j)
        data, sr = librosa.load(filename,sr=22050)
        
        raw_x.append(data)
        raw_y.append(i)

x = []
y = []

print("Generating Spectogram")
for j in range(0,3,1):
  for i in range(len(raw_x)):
      data = raw_x[i]
      if len(data) == 44100:
          #spec, t, f, im = plt.specgram(data,NFFT=nfft,Fs=sr)
          #norm_data = p.normalize(10*np.log10(spec))
          stft = np.abs(librosa.stft(data, n_fft=255, hop_length = hop_length))
          stft_norm = p.normalize(10*np.log10(stft))
          #MFCCs = librosa.feature.mfcc(y=data, n_fft=nfft, hop_length=hop_length,n_mfcc=128)
          #MFCCs_abs = np.abs(MFCCs)
          #MFCCs_norm = p.normalize(10*np.log10(MFCCs_abs))

          #image = np.dstack((stft_norm,MFCCs))
          #image = np.dstack((image,MFCCs_norm))
          #image = np.dstack((stft_norm,MFCCs_norm))
          #image = np.dstack((image,MFCCs_abs))
          #x.append(image)
          #x.append(norm_data)
          x.append(stft_norm)
          y.append(raw_y[i])

y_copy = copy.copy(y)
le = p.LabelEncoder()
le.fit(y_copy)
y_copy_new = le.transform(y_copy)
y_copy_new =  np_utils.to_categorical(y_copy_new, n_classes)
x_copy = np.array(x)

X_train, X_test, y_train, y_test = train_test_split(x_copy, y_copy_new, test_size=0.25, random_state=123, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=123)

print("Building Model")
model = models.Sequential()


#model.add(layers.LSTM(128, input_shape=(128,346), return_sequences=True))
#model.add(layers.Reshape(target_shape=(128,128,1)))

#model.add(layers.Conv2D(40, (3, 3), activation='relu', input_shape=(128, 346, 1), padding="same"))
model.add(layers.Conv2D(40, (3, 3), activation='relu', input_shape=(128, 345, 1), padding="same"))
model.add(layers.MaxPooling2D((3, 3),strides=(2,2),padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.2))

model.add(layers.Conv2D(80, (3, 3), activation='relu', padding="same"))
model.add(layers.MaxPooling2D((3, 3),strides=(2,2),padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.1))

model.add(layers.Conv2D(160, (3, 3), activation='relu', padding="same"))
model.add(layers.MaxPooling2D((3, 3),strides=(2,2),padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.1))

model.add(layers.Conv2D(160, (3, 3), activation='relu', padding="same"))
model.add(layers.MaxPooling2D((1, 44),strides=(1,1),padding="same"))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.1))

model.add(layers.TimeDistributed(layers.Flatten()))
model.add(layers.LSTM(128))

#model.add(layers.Flatten())
#model.add(layers.Dense(1024,activation='relu'))
#model.add(layers.Dropout(0.2))
#model.add(layers.Dense(512,activation='relu'))
#model.add(layers.Dropout(0.2))
#model.add(layers.Dense(128,activation='relu'))
#model.add(layers.Dropout(0.2))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(9,activation='softmax'))

model.summary()
sgd = SGD(learning_rate=0.001)
model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])
print("Training Model")
history = model.fit(X_train, y_train, epochs=25, validation_split=0.25,batch_size=16)

#Testing
results = model.evaluate(X_train, y_train, batch_size=16)
print(results)

predictions = model.predict(X_test)

print(confusion_matrix(np.argmax(y_train,axis=1), np.argmax(predictions,axis=1)))