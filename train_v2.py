import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Dense, Dropout, Flatten, 
                                      GlobalAveragePooling2D,
                                      BatchNormalization, Input)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

IMG_SIZE = 48
BATCH_SIZE = 64
EPOCHS = 50
DATASET_PATH = "dataset"

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    f"{DATASET_PATH}/train",
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="rgb",        # VGG16 needs RGB
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

test_gen = test_datagen.flow_from_directory(
    f"{DATASET_PATH}/test",
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

for layer in base_model.layers[:-4]:
    layer.trainable = False

inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(7, activation='softmax')(x)

model = Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

callbacks = [
    ModelCheckpoint(
        "emotion_model_v2.h5",
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.000001,
        verbose=1
    )
]

print("\n Training VGG16 model...")
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=test_gen,
    callbacks=callbacks,
    verbose=1
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history.history['accuracy'], color='cyan', label='Train')
ax1.plot(history.history['val_accuracy'], color='orange', label='Val')
ax1.set_title('VGG16 Accuracy')
ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.plot(history.history['loss'], color='cyan', label='Train')
ax2.plot(history.history['val_loss'], color='orange', label='Val')
ax2.set_title('VGG16 Loss')
ax2.legend(); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_v2_results.png")
plt.show()

# ─── Evaluate ─────────────────────────────────────
loss, acc = model.evaluate(test_gen, verbose=0)
print(f"\n✅ Training Complete!")
print(f"📊 Test Accuracy: {acc*100:.2f}%")
print(f"💾 Model saved: emotion_model_v2.h5")
