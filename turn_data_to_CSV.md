#example#
data = [['ID', 'Genre']]
for i in range(10):
  data.append(['test' + '00' + str(i) + '.wav', clf.predict([Xtest[i]])[0]])
for i in range(10, 100):
  data.append(['test' + '0' + str(i) + '.wav', clf.predict([Xtest[i]])[0]])
for i in range(100, 200):
  data.append(['test' + str(i) + '.wav', clf.predict([Xtest[i]])[0]])

  file_name = 'SVC_classifier.csv'

with open(file_name, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)
