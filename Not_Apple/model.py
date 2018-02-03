from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, Flatten, Conv2D, Dropout
from keras.preprocessing.image import ImageDataGenerator

model = Sequential()
model.add(Conv2D(32, 3, 3, input_shape=(64,64,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(output_dim=128, activation='relu'))
model.add(Dense(output_dim=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

train_gen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory('Data/Training', target_size=(64, 64), batch_size=32, class_mode='binary')
test_data = test_gen.flow_from_directory('Data/Validation', target_size=(64, 64), batch_size=32, class_mode='binary')

model.fit_generator(train_data, steps_per_epoch=493, epochs=2, validation_data=test_data, validation_steps=165)
