from utils import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# step 1
path = 'myData'
data = importDataInfo(path)

# step 2
data = balanceData(data, display=False)

# step 3
imagesPath, steerings = loadData(path, data)
#print(imagesPath[0], steerings[0])

# step 4
xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, steerings, test_size=0.2, random_state=5)
print("Training_set", len(xTrain))
print("Validation_set", len(xVal))

# step 5

# step 6

# step 7

# step 8
model = createModel()
model.summary()

# step 9
history = model.fit(batchGen(xTrain, yTrain, 100, 1), steps_per_epoch=300, epochs=10,
          validation_data=batchGen(xVal, yVal, 100, 0), validation_steps=200)

# step 10
model.save('model.h5')
print("model saved")

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.ylim([0, 1])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()
