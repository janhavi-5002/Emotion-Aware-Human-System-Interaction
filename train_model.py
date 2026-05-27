import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dense,
                                      Dropout, Flatten, BatchNormalization,
                                      GlobalAveragePooling2D)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# ─── Config ───────────────────────────────────────
IMG_SIZE    = 48
BATCH_SIZE  = 64   # smaller = more gradient updates = better learning
EPOCHS      = 50   # increased from 20
DATASET_PATH = "dataset"

# ─── Data Augmentation (improved) ────────────────
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    horizontal_flip=True,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    f"{DATASET_PATH}/train",
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

test_gen = test_datagen.flow_from_directory(
    f"{DATASET_PATH}/test",
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

print(f"Classes: {train_gen.class_indices}")

# ─── Deeper Model ────────────────────────────────
model = Sequential([
    # Block 1
    Conv2D(64, (3,3), padding='same', activation='relu', input_shape=(48,48,1)),
    BatchNormalization(),
    Conv2D(64, (3,3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),

    # Block 2
    Conv2D(128, (3,3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(128, (3,3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),

    # Block 3
    Conv2D(256, (3,3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(256, (3,3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),

    # Block 4
    Conv2D(512, (3,3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),

    # Classifier
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(7, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
print(f"\nTotal params: {model.count_params():,}")

# ─── Callbacks ────────────────────────────────────
callbacks = [
    ModelCheckpoint(
        "emotion_model.h5",
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=10,          # more patience
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        verbose=1
    )
]

# ─── Train ────────────────────────────────────────
print("\nTraining started... (50 epochs, ~1-2 hrs on CPU)")
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=test_gen,
    callbacks=callbacks,
    verbose=1
)

# ─── Plot ─────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(history.history['accuracy'],     color='cyan',   label='Train', linewidth=2)
ax1.plot(history.history['val_accuracy'], color='orange', label='Val',   linewidth=2)
ax1.set_title('Accuracy'); ax1.legend(); ax1.grid(True, alpha=0.3)
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy')

ax2.plot(history.history['loss'],     color='cyan',   label='Train', linewidth=2)
ax2.plot(history.history['val_loss'], color='orange', label='Val',   linewidth=2)
ax2.set_title('Loss'); ax2.legend(); ax2.grid(True, alpha=0.3)
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss')

plt.suptitle('Emotion Model Training Results', fontsize=14)
plt.tight_layout()
plt.savefig("training_results.png", dpi=150)
plt.show()

# ─── Evaluate ─────────────────────────────────────
loss, acc = model.evaluate(test_gen, verbose=0)
print(f"\nTraining Complete!")
print(f"Test Accuracy: {acc*100:.2f}%")
print(f"Model saved: emotion_model.h5")
print(f"\nExpected accuracy range: 60-65% (FER2013 is hard!)")
