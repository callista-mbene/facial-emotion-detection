import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os

# ============================================
# CONFIGURATION
# ============================================
# IMPORTANT: Change this to your dataset path
DATASET_PATH = 'C:/Users/chiam/Downloads/fer2013'  # Update this to where you extracted the dataset

# Image settings
IMG_SIZE = 48  # FER2013 images are 48x48 pixels
BATCH_SIZE = 64
EPOCHS = 30  # Can adjust based on your computer speed

# Emotion labels (must match folder names in dataset)
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

print("=" * 50)
print("FACIAL EMOTION DETECTION - MODEL TRAINING")
print("=" * 50)

# ============================================
# STEP 1: DATA PREPARATION
# ============================================
print("\n[STEP 1] Loading and preparing data...")

# Data augmentation for training (creates variations of images to prevent overfitting)
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values to 0-1
    rotation_range=10,  # Randomly rotate images
    width_shift_range=0.1,  # Randomly shift horizontally
    height_shift_range=0.1,  # Randomly shift vertically
    horizontal_flip=True,  # Randomly flip images
    zoom_range=0.1  # Randomly zoom
)

# Only rescaling for validation/test data (no augmentation)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load training data
train_generator = train_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, 'train'),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode='grayscale',  # FER2013 is grayscale
    class_mode='categorical',
    shuffle=True
)

# Load test data
test_generator = test_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, 'test'),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=False
)

print(f"✓ Training samples: {train_generator.samples}")
print(f"✓ Test samples: {test_generator.samples}")
print(f"✓ Emotion classes: {list(train_generator.class_indices.keys())}")

# ============================================
# STEP 2: BUILD THE CNN MODEL
# ============================================
print("\n[STEP 2] Building the neural network...")

model = Sequential([
    # First convolutional block
    Conv2D(64, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1), padding='same'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    # Second convolutional block
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    # Third convolutional block
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    # Fourth convolutional block
    Conv2D(512, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(512, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    # Flatten and dense layers
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(len(EMOTIONS), activation='softmax')  # Output layer
])

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✓ Model architecture built successfully!")
print(f"\nModel Summary:")
model.summary()

# ============================================
# STEP 3: TRAIN THE MODEL
# ============================================
print("\n[STEP 3] Training the model...")
print("This may take 30-60 minutes depending on your computer.")
print("You'll see progress bars for each epoch.\n")

# Callbacks to improve training
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=0.00001,
    verbose=1
)

# Train the model
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# ============================================
# STEP 4: EVALUATE THE MODEL
# ============================================
print("\n[STEP 4] Evaluating model performance...")

test_loss, test_accuracy = model.evaluate(test_generator, verbose=0)
print(f"\n{'='*50}")
print(f"FINAL RESULTS:")
print(f"{'='*50}")
print(f"Test Accuracy: {test_accuracy*100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# ============================================
# STEP 5: SAVE THE MODEL
# ============================================
print("\n[STEP 5] Saving the trained model...")

model.save('face_emotionModel.h5')
print("✓ Model saved as 'face_emotionModel.h5'")

print("\n" + "="*50)
print("TRAINING COMPLETE!")
print("="*50)
print("\nYou can now use this model in your Flask app.")
print("The file 'face_emotionModel.h5' contains the trained AI brain.")