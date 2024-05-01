### Format Train Data Labels (y)
import librosa
import numpy as np
import csv
from sklearn.preprocessing import StandardScaler


#format train labels
with open('train.csv', 'r') as file:
    # Create a CSV reader object
    reader = csv.reader(file)
    # Iterate over each row in the CSV file
    y = []
    for row in reader:
        # Process each row
        #print(row)  # Example: Print each row
        y.append(row[1])
    y.pop(0)

print('Shape of y: ' + str(np.shape(y)))

### Split Labels
from sklearn import preprocessing
y_train = y[:800]
#y_test = y[600:800]

### Find min length of features
min_length = None

### Find min length of mfcc coefficient
for i in range (0,400): #checking 400 should be enough
  if i < 10:
    file_name = 'train' + '00' + str(i) + '.wav'
  elif i < 100:
    file_name = 'train' + '0' + str(i) + '.wav'
  else:
    file_name = 'train' + str(i) + '.wav'
  data, samplerate = librosa.load(file_name)
  mfcc = librosa.feature.mfcc(y=data, sr=samplerate)
  delta_mfcc = librosa.feature.delta(mfcc)
  delta2_mfcc = librosa.feature.delta(mfcc, order=2)
  spec_con = librosa.feature.spectral_contrast(y=data, sr =samplerate)
  spec_cen = librosa.feature.spectral_centroid(y=data, sr=samplerate)
  chroma = librosa.feature.chroma_stft(y=data, sr=samplerate)
  mfcc_d = np.concatenate((mfcc, delta_mfcc, delta2_mfcc,spec_con,spec_cen, spec_roll, chroma))
  #print('Shape of mfcc: ' + str(np.shape(mfcc)))
  if min_length is None or mfcc_d.shape[1] < min_length:
    min_length = mfcc_d.shape[1]

print(min_length)

### Code to create smaller audio chunks, unused
#import librosa

#def audio_chunks(file_path, chunk_duration=10):
    # Load the audio file
    #audio, sr = librosa.load(file_path, sr=None)  # sr=None to preserve the original sampling rate

    # Calculate the number of samples per 10-second chunk
    #samples_per_chunk = sr * chunk_duration

    # Calculate total number of chunks possible
    #total_chunks = int(len(audio) / samples_per_chunk)

    # Split the audio into chunks
    #chunks = [audio[i * samples_per_chunk:(i + 1) * samples_per_chunk] for i in range(total_chunks)]

    #return chunks, sr

# Example usage:
#file_path = 'train234.wav'
#chunks1, sample_rate = audio_chunks(file_path, chunk_duration=7)
#print(np.shape(chunks1[0]))

### Extract MFCC coefficients of all .wav files
import librosa

X = []
for i in range(800):
  if i < 10:
    file_name = 'train' + '00' + str(i) + '.wav'
  elif i < 100:
    file_name = 'train' + '0' + str(i) + '.wav'
  else:
    file_name = 'train' + str(i) + '.wav'
  data, samplerate = librosa.load(file_name)

  mfcc = librosa.feature.mfcc(y=data, sr=samplerate)
  delta_mfcc = librosa.feature.delta(mfcc)
  delta2_mfcc = librosa.feature.delta(mfcc, order=2)
  spec_con = librosa.feature.spectral_contrast(y=data, sr =samplerate)
  spec_cen = librosa.feature.spectral_centroid(y=data, sr=samplerate)
  chroma = librosa.feature.chroma_stft(y=data, sr=samplerate)
  mfcc_d = np.concatenate((mfcc, delta_mfcc, delta2_mfcc,spec_con,spec_cen, chroma)) #misleading label, contains all features
    #print('Shape of mfcc: ' + str(np.shape(mfcc)))
  X.append(mfcc_d[:,:min_length])

### Format features matrix
X = np.array(X)
X_flat = X.reshape(X.shape[0], -1)
print(np.shape(X))
print(np.shape(X_flat))
X_mfccd_train = X_flat[:800,:]
#X_mfccd_test = X_flat[600:800,:]

#print('Shape of X_flat: ' + str(np.shape(X_flat)))
print('Shape of X_mfccd_train: ' + str(np.shape(X_mfccd_train)))
print('Shape of X_mfccd_test: ' + str(np.shape(X_mfccd_test)))

### Format Test Data

import librosa

X = []
for i in range(200):
  if i < 10:
    file_name = 'test' + '00' + str(i) + '.wav'
  elif i < 100:
    file_name = 'test' + '0' + str(i) + '.wav'
  else:
    file_name = 'test' + str(i) + '.wav'
  data, samplerate = librosa.load(file_name)

  #for chunk in chunks:
  mfcc = librosa.feature.mfcc(y=data, sr=samplerate)
  delta_mfcc = librosa.feature.delta(mfcc)
  delta2_mfcc = librosa.feature.delta(mfcc, order=2)
  spec_con = librosa.feature.spectral_contrast(y=data, sr =samplerate)
  spec_cen = librosa.feature.spectral_centroid(y=data, sr=samplerate)
  spec_roll = librosa.feature.spectral_rolloff(y=data, sr=samplerate)
  chroma = librosa.feature.chroma_stft(y=data, sr=samplerate)
  zcr = librosa.feature.zero_crossing_rate(y=data)
  mfcc_d = np.concatenate((mfcc, delta_mfcc, delta2_mfcc,spec_con,spec_cen, spec_roll, chroma, zcr))
    #print('Shape of mfcc: ' + str(np.shape(mfcc)))
  X.append(mfcc_d[:,:min_length])

print('Shape of X: ' + str(np.shape(X)))

X = np.array(X)
X_flat = X.reshape(X.shape[0], -1) #flatten so each data point has 1 long vector

minmax = preprocessing.MinMaxScaler()
X_flat_scale = minmax.fit_transform(X_flat)
X_mfccd_test = X_flat_scale
print(np.shape(X_mfccd_test))

### Hyperparameter Tuning

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Create the pipeline
pipeline = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=50))

param_grid = {
    'randomforestclassifier__n_estimators': [50, 100, 200, 500, 1000],
    'randomforestclassifier__max_depth': [None, 10, 20, 30, 40],
    'randomforestclassifier__min_samples_split': [2, 5, 10],
    'randomforestclassifier__min_samples_leaf': [1, 2, 4]
}

# Setup RandomizedSearchCV
random_search = RandomizedSearchCV(pipeline, param_distributions=param_grid, n_iter=100, cv=3, verbose=3, random_state=50, n_jobs=-1)
random_search.fit(X_mfccd_train, y_train)

print("Best parameters found (Random Search): ", random_search.best_params_)
best_random_model = random_search.best_estimator_

### Random Forest Classifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Initialize a pipeline with data scaling and Random Forest classifier
# Scaling is not strictly necessary for Random Forests, but it's a good practice
clf_hyper = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=200, max_depth=30, min_samples_split=5, min_samples_leaf=4, random_state=50, n_jobs=-1)) #maybe remove random_state
# Train the Random Forest classifier
clf_hyper.fit(X_mfccd_train, y_train)

y_pred = clf_hyper.predict(X_mfccd_test)

#print(classification_report(y_test, y_pred))

### Features Importance

# Get feature importances
feature_importances = rf_model.feature_importances_

# Create labels for the features
features = [f'Feature {i+1}' for i in range(len(feature_importances))]

# Sorting the features by importance
indices = np.argsort(rf_model.feature_importances_)[::-1]
top_indices = indices[:500]  # Select the indices of the top 10 features
sorted_features = [features[i] for i in top_indices]
sorted_importances = feature_importances[top_indices]

# Plotting
plt.figure(figsize=(10, 7))
plt.scatter(sorted_features, sorted_importances, color='#9932CC', s=10)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importances in RandomForest Classifier')
plt.xticks(rotation=90)  # Rotate feature labels for better readability
plt.tight_layout()  # Adjusts plot to ensure everything fits without overlap
plt.gca().set_xticks(sorted_features[::10])  # Show only every 10th label

plt.tight_layout()  # Adjusts plot to ensure everything fits without overlap

plt.show()

### Extract Test Data labels
file_id = []
for i in range(10):
  file_id.append('test' + '00' + str(i) + '.wav')

for i in range(10,100):
  file_id.append('test' + '0' + str(i) + '.wav')
for i in range(100,200):
  file_id.append('test' + str(i) + '.wav')
print(file_id)
print(np.shape(file_id))

### Turn to CSV
import csv
from google.colab import files

with open('RF3_submission.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    field = ["ID", "Genre"]

    writer.writerow(field)

    for i in range(200):
      writer.writerow([file_id[i],y_pred[i]])

files.download('RF3_submission.csv')
