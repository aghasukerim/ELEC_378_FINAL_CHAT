def chroma_features(file_name):
  audio_file = file_name
  y, sr = librosa.load(audio_file)
  chroma = librosa.feature.chroma_stft(y=y, sr=sr)
  chroma_vector = chroma.flatten()
  return chroma_vector
