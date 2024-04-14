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
y_train = y[:800]
#y_test = y[600:800] #for validation

print('Shape of y: ' + str(np.shape(y)))
print('Shape of y_train: ' + str(np.shape(y_train)))
#print('Shape of y_test: ' + str(np.shape(y_test)))

### MFCC Coefficients w/ deltas

X = []
min_length = None

### Find min length of mfcc coefficient
for i in range (100,400): #checking 400 should be enough
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
  mfcc_d = np.concatenate((mfcc, delta_mfcc, delta2_mfcc))
  #print('Shape of mfcc: ' + str(np.shape(mfcc)))
  if min_length is None or mfcc_d.shape[1] < min_length:
    min_length = mfcc_d.shape[1]

### Extract MFCC coefficients of all .wav files
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
  mfcc_d = np.concatenate((mfcc, delta_mfcc, delta2_mfcc))
  #print('Shape of mfcc: ' + str(np.shape(mfcc)))
  X.append(mfcc_d[:,:min_length])


print('Shape of X: ' + str(np.shape(X)))
X = np.array(X)
X_flat = X.reshape(X.shape[0], -1) #flatten so each data point has 1 long vector

X_mfccd_train = X_flat[:800,:]
#X_mfccd_test = X_flat[600:800,:]

print('Shape of X_mfccd_train: ' + str(np.shape(X_mfccd_train)))
#print('Shape of X_mfccd_test: ' + str(np.shape(X_mfccd_test)))

### Random Forest Classifier
from sklearn.feature_selection import SelectKBest, f_classif

#from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

### Grid Search to find Optimal Hyperparameters

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

# Define the parameter grid
param_grid = { # ran one by one because all took too long
    'randomforestclassifier__n_estimators': [50, 100, 200, 300], #100 (mfccd)
    #'randomforestclassifier__max_depth': [None, 10, 20, 30], #10 (mfccd)
    #'randomforestclassifier__min_samples_split': [2, 5, 10], #10 (mfccd)
    #'randomforestclassifier__min_samples_leaf': [1, 2, 4, 6], #2 (mfccd)
}

# Create a pipeline
pipeline = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=50))

# Initialize GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=3, verbose=3, n_jobs=-1)

# Fit GridSearchCV
grid_search.fit(X_mfccd_train, y_train)

# Best parameters and estimator
print("Best parameters found: ", grid_search.best_params_)
best_model = grid_search.best_estimator_

### Random Forest Classifier

# Initialize a pipeline with data scaling and Random Forest classifier
# Scaling is not strictly necessary for Random Forests, but it's a good practice
clf_hyper = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=10, min_samples_leaf=2, random_state=50)) #maybe remove random_state

# Train the Random Forest classifier
clf_hyper.fit(X_mfccd_train, y_train)

### Format Test Data

X = []
min_length = 1290 #from previous finding

### Extract MFCC coefficients of all .wav files
for i in range(200):
  if i < 10:
    file_name = 'test' + '00' + str(i) + '.wav'
  elif i < 100:
    file_name = 'test' + '0' + str(i) + '.wav'
  else:
    file_name = 'test' + str(i) + '.wav'
  data, samplerate = librosa.load(file_name)
  mfcc = librosa.feature.mfcc(y=data, sr=samplerate)
  delta_mfcc = librosa.feature.delta(mfcc)
  delta2_mfcc = librosa.feature.delta(mfcc, order=2)
  chroma = librosa.feature.chroma_stft(y=data, sr=samplerate)
  submit = np.concatenate((mfcc, delta_mfcc, delta2_mfcc, chroma))
  #print('Shape of mfcc: ' + str(np.shape(mfcc)))
  X.append(submit[:,:min_length])


print('Shape of X: ' + str(np.shape(X)))
X = np.array(X)
X_flat = X.reshape(X.shape[0], -1)

X_test = X_flat

print('Shape of X_test: ' + str(np.shape(X_test)))

### Put Test Data into Model

# Predict on the test set
y_test_pred = clf_hyper.predict(X_test)
print(y_test_pred)
print(np.shape(y_test_pred))

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

with open('RFs2_submission.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    field = ["ID", "Genre"]

    writer.writerow(field)

    for i in range(200):
      writer.writerow([file_id[i],y_test_pred[i]])

files.download('RFs2_submission.csv')