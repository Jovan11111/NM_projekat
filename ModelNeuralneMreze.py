import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#Napomena: Ja ovo nisam pokrenuo jer nema potrebe da na moj računar skidam ni tensorflow ni onaj dataset, već sam
# samo gledao šta treba


train_datagen = ImageDataGenerator(rescale=1./255)
test_val_datagen = ImageDataGenerator(rescale=1./255)

# Ovde može da se radi augmentacija al ne mora, svj

train_generator = train_datagen.flow_from_directory(
        'path_to_root_dir/train',# na ova mesta ti stavi path do train DIRa
        target_size=(0, 0),# ovde treba velicina slike nakon augmentacije, ili samo velicina slike
        batch_size=32,# stavio sam batch mode da bude lakše po grafičku mlako
        class_mode='categorical')

validation_generator = test_val_datagen.flow_from_directory(
        'path_to_root_dir/val',# na ova mesta ti stavi path do val DIRa
        target_size=(0, 0),# ovde treba velicina slike nakon augmentacije, ili samo velicina slike
        batch_size=32,
        class_mode='categorical')

test_generator = test_val_datagen.flow_from_directory(
        'path_to_root_dir/test',# na ova mesta ti stavi path do test DIRa
        target_size=(0, 0),# ovde treba velicina slike nakon augmentacije, ili samo velicina slike
        batch_size=32,
        class_mode='categorical',
        shuffle=False)  # Ovo ne znam šta je ali kada sam ubacio kod u GPT, on je rekao da mi ovo fali


model = Sequential([
    Input(shape=(0, 0, 3)),  # Ovo je velicina posle augmentacije
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')  # 4 classes, 4 klase je objašnjeno zašto u pdfu
])


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy']) #Ovo sam našao na netu da trebaju ovi parametri


history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=50  #Lupio vrednost, kako god više odgovara grafičkoj
)


test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)

print(test_acc)

# Ovo čuva model, nzm dal mi treba da predamo već istreniran
model.save('path_to_save_model/my_model.h5')


predictions = model.predict(test_generator, steps=test_generator.samples // test_generator.batch_size)
