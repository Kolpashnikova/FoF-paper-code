import os
import gc
import tempfile
import pandas as pd
import numpy as np
from PIL import Image
from scipy import signal
from scipy.signal import spectrogram
from scipy.stats import ttest_ind
from scipy.interpolate import interp1d
import pywt
import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score
)
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from keras.applications.vgg16 import preprocess_input

df = pd.read_parquet("data/model_data_final_full.parquet")

feature_cols = [
    'filtered_acc_x', 'filtered_acc_y', 'filtered_acc_z'
]

max_length = 300
step_size = 300
fs = 100
num_classes = 2

def create_multi_representation_image(window_signals):
    if window_signals.shape[0] != max_length:
        if window_signals.shape[0] < max_length:
            pad_len = max_length - window_signals.shape[0]
            window_signals = np.pad(window_signals, ((0, pad_len), (0, 0)), mode='constant')
        else:
            window_signals = window_signals[:max_length, :]

    signals_T = window_signals.T

    windowed_signals = signals_T * signal.windows.hann(max_length)
    fft_result = np.fft.fft(windowed_signals, axis=1)
    fft_magnitude = np.abs(fft_result)

    meaningful_freqs = max_length // 4
    fft_focused = fft_magnitude[:, :meaningful_freqs]

    acc_magnitude = np.sqrt(np.sum(signals_T**2, axis=0))

    acc_x, acc_y, acc_z = signals_T[0], signals_T[1], signals_T[2]
    tilt_x = np.arctan2(acc_y, np.sqrt(acc_x**2 + acc_z**2))
    tilt_y = np.arctan2(-acc_x, np.sqrt(acc_y**2 + acc_z**2))

    acc_features = np.array([
        acc_magnitude,
        tilt_x,
        tilt_y
    ])

    acc_fft = np.abs(np.fft.fft(acc_features, axis=1))
    acc_fft_focused = acc_fft[:, :meaningful_freqs]

    spectrograms = []
    for i in range(3):
        f, t, Sxx = spectrogram(signals_T[i], fs=fs, nperseg=32, noverlap=16)
        Sxx_log = np.log1p(Sxx)
        spectrograms.append(Sxx_log)

    spectro_stack = np.array(spectrograms)
    spectro_flat = spectro_stack.reshape(spectro_stack.shape[0], -1)

    statistical_features = []
    for i in range(3):
        sig = signals_T[i]
        window_size = 30
        rolling_mean = np.convolve(sig, np.ones(window_size)/window_size, mode='same')
        rolling_std = np.array([np.std(sig[max(0, j-window_size//2):j+window_size//2])
                               for j in range(len(sig))])
        rolling_energy = np.convolve(sig**2, np.ones(window_size)/window_size, mode='same')

        downsample_factor = 4
        features = np.column_stack([
            rolling_mean[::downsample_factor],
            rolling_std[::downsample_factor],
            rolling_energy[::downsample_factor]
        ])
        statistical_features.append(features)

    stat_stack = np.array(statistical_features)
    stat_flat = stat_stack.reshape(stat_stack.shape[0], -1)

    wavelet_features = []
    for i in range(3):
        scales = np.arange(1, 32)
        coefficients, _ = pywt.cwt(signals_T[i], scales, 'morl')
        coeff_mag = np.abs(coefficients)
        coeff_compressed = np.mean(coeff_mag, axis=1)
        wavelet_features.append(coeff_compressed)

    wavelet_stack = np.array(wavelet_features)

    jerk_features = []
    for i in range(3):
        jerk = np.diff(signals_T[i], prepend=signals_T[i, 0])
        jerk_fft = np.abs(np.fft.fft(jerk))
        jerk_focused = jerk_fft[:meaningful_freqs]
        jerk_features.append(jerk_focused)

    jerk_stack = np.array(jerk_features)

    target_width = 75

    if spectro_flat.shape[1] > target_width:
        step = spectro_flat.shape[1] // target_width
        spectro_resized = spectro_flat[:, ::step][:, :target_width]
    else:
        spectro_resized = np.pad(spectro_flat, ((0, 0), (0, target_width - spectro_flat.shape[1])), mode='edge')

    if stat_flat.shape[1] > target_width:
        step = stat_flat.shape[1] // target_width
        stat_resized = stat_flat[:, ::step][:, :target_width]
    else:
        stat_resized = np.pad(stat_flat, ((0, 0), (0, target_width - stat_flat.shape[1])), mode='edge')

    wavelet_resized = np.pad(wavelet_stack, ((0, 0), (0, target_width - wavelet_stack.shape[1])), mode='edge')

    combined_representation = np.vstack([
        fft_focused,
        acc_fft_focused,
        spectro_resized,
        stat_resized,
        wavelet_resized,
        jerk_stack
    ])

    return combined_representation

def create_enhanced_image(combined_repr):
    scaler = RobustScaler()
    combined_scaled = scaler.fit_transform(combined_repr.T).T

    min_val = np.min(combined_scaled)
    if min_val < 0:
        combined_shifted = combined_scaled - min_val + 1e-8
    else:
        combined_shifted = combined_scaled + 1e-8

    combined_log = np.log1p(combined_shifted)

    max_val = np.max(combined_log)
    if max_val > 0:
        combined_norm = combined_log / max_val
    else:
        combined_norm = combined_log

    rows, cols = combined_norm.shape
    target_min_dim = 48

    if rows < target_min_dim:
        tile_factor = (target_min_dim // rows) + 1
        combined_norm = np.tile(combined_norm, (tile_factor, 1))
        combined_norm = combined_norm[:target_min_dim, :]

    image_rgb = np.repeat(combined_norm[:, :, np.newaxis], 3, axis=2)

    image_pil = Image.fromarray((image_rgb * 255).astype('uint8'))
    image_resized = image_pil.resize((224, 224), Image.BILINEAR)

    image_array = np.array(image_resized).astype('float32')

    image_preprocessed = preprocess_input(image_array)

    return image_preprocessed

data_records = []
minority_count = 0
majority_count = 0

for (pid, fid), group_data in df.groupby(['participant_full_id', 'file_id']):
    group_data = group_data.sort_values('elapsed_time')

    if hasattr(group_data[feature_cols], 'to_pandas'):
        signals = group_data[feature_cols].to_pandas().values
        label_values = group_data['Potential_FoF'].to_pandas().values
    else:
        signals = group_data[feature_cols].values
        label_values = group_data['Potential_FoF'].values

    scaler = StandardScaler()
    signals = scaler.fit_transform(signals)

    T = signals.shape[0]

    start_positions = range(0, T, step_size)

    for start in start_positions:
        end = start + max_length

        if end > T:
            window_signals = signals[start:T]
            window_labels = label_values[start:T]

            if window_signals.shape[0] < max_length:
                pad_len = max_length - window_signals.shape[0]
                window_signals = np.pad(window_signals, ((0, pad_len), (0, 0)),
                                      mode='constant', constant_values=0)
                window_labels = np.pad(window_labels, (0, pad_len),
                                     mode='constant', constant_values=0)
        else:
            window_signals = signals[start:end]
            window_labels = label_values[start:end]

        minority_samples = np.sum(window_labels == 1)
        total_samples = len(window_labels)
        minority_ratio = minority_samples / total_samples

        if minority_ratio > 0.3:
            window_label = 1
            minority_count += 1

            base_repr = create_multi_representation_image(window_signals)
            base_image = create_enhanced_image(base_repr)
            data_records.append((base_image, window_label))

            for noise_level in [0.01, 0.02, 0.03]:
                noise = np.random.normal(0, noise_level, window_signals.shape)
                noisy_signals = window_signals + noise
                noisy_repr = create_multi_representation_image(noisy_signals)
                noisy_image = create_enhanced_image(noisy_repr)
                data_records.append((noisy_image, window_label))
                minority_count += 1

        else:
            window_label = 0
            majority_count += 1

            combined_repr = create_multi_representation_image(window_signals)
            enhanced_image = create_enhanced_image(combined_repr)
            data_records.append((enhanced_image, window_label))

X_images = np.array([rec[0] for rec in data_records])
y_labels = np.array([rec[1] for rec in data_records])
y_onehot = tf.keras.utils.to_categorical(y_labels, num_classes=num_classes)



gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        pass

if gpus:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)

def clear_memory():
    gc.collect()
    if gpus:
        tf.keras.backend.clear_session()

TEMP_DIR = "./ultra_temp_data/"
os.makedirs(TEMP_DIR, exist_ok=True)

def save_data_to_disk_in_chunks(X_images, y_onehot, chunk_size=500):
    y_labels = tf.argmax(y_onehot, axis=1).numpy()

    indices = np.arange(len(X_images))
    train_indices, val_indices = train_test_split(
        indices, test_size=0.2, stratify=y_labels, random_state=42
    )

    train_chunks = []
    minority_count = 0
    majority_count = 0

    for i in range(0, len(train_indices), chunk_size):
        chunk_indices = train_indices[i:i+chunk_size]

        X_chunk = X_images[chunk_indices]
        y_chunk = y_onehot[chunk_indices]
        y_chunk_labels = y_labels[chunk_indices]

        majority_mask = y_chunk_labels == 0
        minority_mask = y_chunk_labels == 1

        majority_indices = np.where(majority_mask)[0]
        minority_indices = np.where(minority_mask)[0]

        if len(minority_indices) > 0 and len(majority_indices) > 0:
            max_minority = min(len(majority_indices), len(minority_indices) * 3)
            if max_minority > len(minority_indices):
                minority_oversampled = np.random.choice(
                    minority_indices, size=max_minority, replace=True
                )
            else:
                minority_oversampled = minority_indices

            balanced_indices = np.concatenate([majority_indices, minority_oversampled])
            np.random.shuffle(balanced_indices)

            X_balanced = X_chunk[balanced_indices]
            y_balanced = y_chunk[balanced_indices]

            y_balanced_labels = tf.argmax(y_balanced, axis=1).numpy()
            minority_count += np.sum(y_balanced_labels == 1)
            majority_count += np.sum(y_balanced_labels == 0)

        else:
            X_balanced = X_chunk
            y_balanced = y_chunk
            y_balanced_labels = y_chunk_labels
            minority_count += np.sum(y_balanced_labels == 1)
            majority_count += np.sum(y_balanced_labels == 0)

        chunk_file = f"{TEMP_DIR}train_chunk_{len(train_chunks)}.npz"
        np.savez_compressed(chunk_file, X=X_balanced, y=y_balanced)
        train_chunks.append(chunk_file)

        del X_chunk, y_chunk, X_balanced, y_balanced
        clear_memory()

    val_chunks = []
    for i in range(0, len(val_indices), chunk_size):
        chunk_indices = val_indices[i:i+chunk_size]

        X_val_chunk = X_images[chunk_indices]
        y_val_chunk = y_onehot[chunk_indices]

        chunk_file = f"{TEMP_DIR}val_chunk_{len(val_chunks)}.npz"
        np.savez_compressed(chunk_file, X=X_val_chunk, y=y_val_chunk)
        val_chunks.append(chunk_file)

        del X_val_chunk, y_val_chunk
        clear_memory()

    return train_chunks, val_chunks, val_indices

class UltraMemoryEfficientGenerator(tf.keras.utils.Sequence):
    def __init__(self, chunk_files, batch_size=8, shuffle=True, augment=False):
        self.chunk_files = chunk_files
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.current_chunk_data = None
        self.current_chunk_idx = 0
        self.sample_indices = []
        self._load_chunk(0)
        self.on_epoch_end()

    def _load_chunk(self, chunk_idx):
        if self.current_chunk_data is not None:
            del self.current_chunk_data
            clear_memory()

        chunk_file = self.chunk_files[chunk_idx % len(self.chunk_files)]
        self.current_chunk_data = np.load(chunk_file)
        self.current_chunk_idx = chunk_idx

    def __len__(self):
        total_samples = 0
        for chunk_file in self.chunk_files:
            chunk = np.load(chunk_file)
            total_samples += len(chunk['X'])
            del chunk
            clear_memory()
        return int(np.ceil(total_samples / self.batch_size))

    def __getitem__(self, idx):
        chunk_idx = idx % len(self.chunk_files)

        if chunk_idx != self.current_chunk_idx:
            self._load_chunk(chunk_idx)

        chunk_size = len(self.current_chunk_data['X'])
        batch_start = (idx * self.batch_size) % chunk_size
        batch_end = min(batch_start + self.batch_size, chunk_size)

        X_batch = self.current_chunk_data['X'][batch_start:batch_end]
        y_batch = self.current_chunk_data['y'][batch_start:batch_end]

        if self.augment:
            X_batch = self._apply_augmentation(X_batch, y_batch)

        return X_batch, y_batch

    def _apply_augmentation(self, X_batch, y_batch):
        augmented = []
        for i, (img, label) in enumerate(zip(X_batch, y_batch)):
            if np.argmax(label) == 1 and np.random.random() < 0.5:
                if np.random.random() < 0.5:
                    img = np.fliplr(img)
                if np.random.random() < 0.3:
                    img = np.clip(img + np.random.uniform(-10, 10), 0, 255)
            augmented.append(img)
        return np.array(augmented)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.chunk_files)

train_chunks, val_chunks, val_indices = save_data_to_disk_in_chunks(X_images, y_onehot, chunk_size=300)

del X_images, y_onehot
clear_memory()

train_generator = UltraMemoryEfficientGenerator(
    train_chunks, batch_size=8, shuffle=True, augment=True
)
val_generator = UltraMemoryEfficientGenerator(
    val_chunks, batch_size=8, shuffle=False, augment=False
)

def create_ultra_simple_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))

    for layer in base_model.layers[:-2]:
        layer.trainable = False
    for layer in base_model.layers[-2:]:
        layer.trainable = True

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    if gpus:
        outputs = layers.Dense(2, activation='softmax', dtype='float32')(x)
    else:
        outputs = layers.Dense(2, activation='softmax')(x)

    return models.Model(inputs=base_model.input, outputs=outputs)

with tf.device('/GPU:0' if gpus else '/CPU:0'):
    model = create_ultra_simple_model()

    def ultra_simple_loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        loss = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=1)
        weights = tf.reduce_sum(y_true * [1.0, 3.0], axis=1)
        return tf.reduce_mean(loss * weights)

    class UltraSimpleCallback(tf.keras.callbacks.Callback):
        def __init__(self, val_generator, val_chunks):
            self.val_generator = val_generator
            self.val_chunks = val_chunks
            self.best_loss = float('inf')
            self.best_minority_f1 = 0
            self.best_roc_auc = 0
            self.patience = 0
            self.max_patience = 5
            self.epoch_count = 0

        def on_epoch_end(self, epoch, logs=None):
            self.epoch_count += 1
            current_loss = logs.get('val_loss', float('inf'))

            minority_f1, roc_auc = self._calculate_metrics()

            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.patience = 0
            else:
                self.patience += 1

            if minority_f1 > self.best_minority_f1:
                self.best_minority_f1 = minority_f1

            if roc_auc > self.best_roc_auc:
                self.best_roc_auc = roc_auc

            if self.patience >= self.max_patience:
                self.model.stop_training = True

        def _calculate_metrics(self):
            all_predictions = []
            all_probabilities = []
            all_true_labels = []

            for i, val_chunk_file in enumerate(self.val_chunks):
                chunk = np.load(val_chunk_file)
                X_val_chunk = chunk['X']
                y_val_chunk = chunk['y']

                chunk_predictions = self.model.predict(X_val_chunk, batch_size=4, verbose=0)
                chunk_pred_classes = np.argmax(chunk_predictions, axis=1)
                chunk_pred_proba = chunk_predictions[:, 1]
                chunk_true_classes = np.argmax(y_val_chunk, axis=1)

                all_predictions.extend(chunk_pred_classes)
                all_probabilities.extend(chunk_pred_proba)
                all_true_labels.extend(chunk_true_classes)

                from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score
                chunk_minority_f1 = f1_score(chunk_true_classes, chunk_pred_classes, pos_label=1, zero_division=0)

                if len(np.unique(chunk_true_classes)) > 1:
                    chunk_roc_auc = roc_auc_score(chunk_true_classes, chunk_pred_proba)
                else:
                    chunk_roc_auc = 0.5

                del chunk, X_val_chunk, y_val_chunk, chunk_predictions
                clear_memory()

            overall_minority_f1 = f1_score(all_true_labels, all_predictions, pos_label=1, zero_division=0)

            if len(np.unique(all_true_labels)) > 1:
                overall_roc_auc = roc_auc_score(all_true_labels, all_probabilities)
            else:
                overall_roc_auc = 0.5

            return overall_minority_f1, overall_roc_auc

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss=ultra_simple_loss,
        metrics=['accuracy']
    )

    ultra_callback = UltraSimpleCallback(val_generator, val_chunks)

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=15,
        callbacks=[ultra_callback],
        verbose=1
    )

    final_predictions = []
    final_probabilities = []
    final_true_labels = []

    for i, val_chunk_file in enumerate(val_chunks):
        chunk = np.load(val_chunk_file)
        X_val_chunk = chunk['X']
        y_val_chunk = chunk['y']

        chunk_predictions = model.predict(X_val_chunk, batch_size=4, verbose=0)
        chunk_pred_classes = np.argmax(chunk_predictions, axis=1)
        chunk_pred_proba = chunk_predictions[:, 1]
        chunk_true_classes = np.argmax(y_val_chunk, axis=1)

        final_predictions.extend(chunk_pred_classes)
        final_probabilities.extend(chunk_pred_proba)
        final_true_labels.extend(chunk_true_classes)

        from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score
        chunk_minority_f1 = f1_score(chunk_true_classes, chunk_pred_classes, pos_label=1, zero_division=0)

        if len(np.unique(chunk_true_classes)) > 1:
            chunk_roc_auc = roc_auc_score(chunk_true_classes, chunk_pred_proba)
        else:
            chunk_roc_auc = 0.5

        del chunk, X_val_chunk, y_val_chunk, chunk_predictions
        clear_memory()

    from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score

    final_minority_f1 = f1_score(final_true_labels, final_predictions, pos_label=1, zero_division=0)

    if len(np.unique(final_true_labels)) > 1:
        final_roc_auc = roc_auc_score(final_true_labels, final_probabilities)
    else:
        final_roc_auc = 0.5

    final_cm = confusion_matrix(final_true_labels, final_predictions)

    y_pred_proba = np.array(final_probabilities)
    y_true_classes = np.array(final_true_labels)

    best_f1 = 0
    best_threshold = 0.5

    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        y_pred_thresh = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_true_classes, y_pred_thresh, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

for chunk_file in train_chunks + val_chunks:
    if os.path.exists(chunk_file):
        os.remove(chunk_file)

if os.path.exists(TEMP_DIR):
    os.rmdir(TEMP_DIR)

del model, train_generator, val_generator
clear_memory()


train_generator = UltraMemoryEfficientGenerator(
    train_chunks, batch_size=8, shuffle=True, augment=True
)
val_generator = UltraMemoryEfficientGenerator(
    val_chunks, batch_size=8, shuffle=False, augment=False
)


def create_ultra_simple_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))

    trainable_from = len(base_model.layers) - 10

    for i, layer in enumerate(base_model.layers):
        if i < trainable_from:
            layer.trainable = False
        else:
            layer.trainable = True

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    if gpus:
        outputs = layers.Dense(2, activation='softmax', dtype='float32')(x)
    else:
        outputs = layers.Dense(2, activation='softmax')(x)

    return models.Model(inputs=base_model.input, outputs=outputs)

with tf.device('/GPU:0' if gpus else '/CPU:0'):
    model = create_ultra_simple_model()

    def ultra_simple_loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        loss = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=1)
        weights = tf.reduce_sum(y_true * [1.0, 3.0], axis=1)
        return tf.reduce_mean(loss * weights)

    class UltraSimpleCallback(tf.keras.callbacks.Callback):
        def __init__(self, val_generator, val_chunks):
            self.val_generator = val_generator
            self.val_chunks = val_chunks
            self.best_loss = float('inf')
            self.best_minority_f1 = 0
            self.patience = 0
            self.max_patience = 5
            self.epoch_count = 0

        def on_epoch_end(self, epoch, logs=None):
            self.epoch_count += 1
            current_loss = logs.get('val_loss', float('inf'))

            minority_f1 = self._calculate_minority_f1()

            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.patience = 0
            else:
                self.patience += 1

            if minority_f1 > self.best_minority_f1:
                self.best_minority_f1 = minority_f1

            if self.patience >= self.max_patience:
                self.model.stop_training = True

        def _calculate_minority_f1(self):
            all_predictions = []
            all_true_labels = []

            for i, val_chunk_file in enumerate(self.val_chunks):
                chunk = np.load(val_chunk_file)
                X_val_chunk = chunk['X']
                y_val_chunk = chunk['y']

                chunk_predictions = self.model.predict(X_val_chunk, batch_size=4, verbose=0)
                chunk_pred_classes = np.argmax(chunk_predictions, axis=1)
                chunk_true_classes = np.argmax(y_val_chunk, axis=1)

                all_predictions.extend(chunk_pred_classes)
                all_true_labels.extend(chunk_true_classes)

                from sklearn.metrics import f1_score, confusion_matrix
                chunk_minority_f1 = f1_score(chunk_true_classes, chunk_pred_classes, pos_label=1, zero_division=0)

                del chunk, X_val_chunk, y_val_chunk, chunk_predictions
                clear_memory()

            overall_minority_f1 = f1_score(all_true_labels, all_predictions, pos_label=1, zero_division=0)
            return overall_minority_f1

    model.compile(
        optimizer=optimizers.Adam(learning_rate=5e-5),
        loss=ultra_simple_loss,
        metrics=['accuracy']
    )

    ultra_callback = UltraSimpleCallback(val_generator, val_chunks)

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=15,
        callbacks=[ultra_callback],
        verbose=1
    )

    final_predictions = []
    final_true_labels = []

    for i, val_chunk_file in enumerate(val_chunks):
        chunk = np.load(val_chunk_file)
        X_val_chunk = chunk['X']
        y_val_chunk = chunk['y']

        chunk_predictions = model.predict(X_val_chunk, batch_size=4, verbose=0)
        chunk_pred_classes = np.argmax(chunk_predictions, axis=1)
        chunk_true_classes = np.argmax(y_val_chunk, axis=1)

        final_predictions.extend(chunk_pred_classes)
        final_true_labels.extend(chunk_true_classes)

        from sklearn.metrics import f1_score, confusion_matrix
        chunk_minority_f1 = f1_score(chunk_true_classes, chunk_pred_classes, pos_label=1, zero_division=0)

        del chunk, X_val_chunk, y_val_chunk, chunk_predictions
        clear_memory()

    from sklearn.metrics import classification_report, confusion_matrix, f1_score

    final_minority_f1 = f1_score(final_true_labels, final_predictions, pos_label=1, zero_division=0)
    final_cm = confusion_matrix(final_true_labels, final_predictions)

    prob_predictions = []
    prob_true_labels = []

    for i, val_chunk_file in enumerate(val_chunks):
        chunk = np.load(val_chunk_file)
        X_val_chunk = chunk['X']
        y_val_chunk = chunk['y']

        chunk_predictions = model.predict(X_val_chunk, batch_size=4, verbose=0)
        chunk_proba = chunk_predictions[:, 1]
        chunk_true_classes = np.argmax(y_val_chunk, axis=1)

        prob_predictions.extend(chunk_proba)
        prob_true_labels.extend(chunk_true_classes)

        del chunk, X_val_chunk, y_val_chunk, chunk_predictions
        clear_memory()

    y_pred_proba = np.array(prob_predictions)
    y_true_classes = np.array(prob_true_labels)

    best_f1 = 0
    best_threshold = 0.5

    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        y_pred_thresh = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_true_classes, y_pred_thresh, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

for chunk_file in train_chunks + val_chunks:
    if os.path.exists(chunk_file):
        os.remove(chunk_file)

if os.path.exists(TEMP_DIR):
    os.rmdir(TEMP_DIR)

del model, train_generator, val_generator
clear_memory()


train_generator = UltraMemoryEfficientGenerator(
    train_chunks, batch_size=8, shuffle=True, augment=True
)
val_generator = UltraMemoryEfficientGenerator(
    val_chunks, batch_size=8, shuffle=False, augment=False
)


def create_ultra_simple_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))

    trainable_from = len(base_model.layers) - 10

    for i, layer in enumerate(base_model.layers):
        if i < trainable_from:
            layer.trainable = False
        else:
            layer.trainable = True

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    if gpus:
        outputs = layers.Dense(2, activation='softmax', dtype='float32')(x)
    else:
        outputs = layers.Dense(2, activation='softmax')(x)

    return models.Model(inputs=base_model.input, outputs=outputs)

with tf.device('/GPU:0' if gpus else '/CPU:0'):
    model = create_ultra_simple_model()

    def ultra_simple_loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        loss = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=1)
        weights = tf.reduce_sum(y_true * [1.0, 3.0], axis=1)
        return tf.reduce_mean(loss * weights)

    class UltraSimpleCallback(tf.keras.callbacks.Callback):
        def __init__(self, val_generator, val_chunks):
            self.val_generator = val_generator
            self.val_chunks = val_chunks
            self.best_loss = float('inf')
            self.best_minority_f1 = 0
            self.best_roc_auc = 0
            self.patience = 0
            self.max_patience = 5
            self.epoch_count = 0

        def on_epoch_end(self, epoch, logs=None):
            self.epoch_count += 1
            current_loss = logs.get('val_loss', float('inf'))

            minority_f1, roc_auc = self._calculate_metrics()

            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.patience = 0
            else:
                self.patience += 1

            if minority_f1 > self.best_minority_f1:
                self.best_minority_f1 = minority_f1

            if roc_auc > self.best_roc_auc:
                self.best_roc_auc = roc_auc

            if self.patience >= self.max_patience:
                self.model.stop_training = True

        def _calculate_metrics(self):
            all_predictions = []
            all_probabilities = []
            all_true_labels = []

            for i, val_chunk_file in enumerate(self.val_chunks):
                chunk = np.load(val_chunk_file)
                X_val_chunk = chunk['X']
                y_val_chunk = chunk['y']

                chunk_predictions = self.model.predict(X_val_chunk, batch_size=4, verbose=0)
                chunk_pred_classes = np.argmax(chunk_predictions, axis=1)
                chunk_pred_proba = chunk_predictions[:, 1]
                chunk_true_classes = np.argmax(y_val_chunk, axis=1)

                all_predictions.extend(chunk_pred_classes)
                all_probabilities.extend(chunk_pred_proba)
                all_true_labels.extend(chunk_true_classes)

                from sklearn.metrics import f1_score, confusion_matrix
                chunk_minority_f1 = f1_score(chunk_true_classes, chunk_pred_classes, pos_label=1, zero_division=0)
                
                try:
                    if len(np.unique(chunk_true_classes)) > 1:
                        chunk_roc_auc = roc_auc_score(chunk_true_classes, chunk_pred_proba)
                    else:
                        chunk_roc_auc = float('nan')
                except:
                    chunk_roc_auc = float('nan')

                del chunk, X_val_chunk, y_val_chunk, chunk_predictions
                clear_memory()

            overall_minority_f1 = f1_score(all_true_labels, all_predictions, pos_label=1, zero_division=0)
            
            try:
                if len(np.unique(all_true_labels)) > 1:
                    overall_roc_auc = roc_auc_score(all_true_labels, all_probabilities)
                else:
                    overall_roc_auc = float('nan')
            except:
                overall_roc_auc = float('nan')

            return overall_minority_f1, overall_roc_auc

    model.compile(
        optimizer=optimizers.Adam(learning_rate=5e-5),
        loss=ultra_simple_loss,
        metrics=['accuracy']
    )

    ultra_callback = UltraSimpleCallback(val_generator, val_chunks)

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=15,
        callbacks=[ultra_callback],
        verbose=1
    )

    final_predictions = []
    final_probabilities = []
    final_true_labels = []

    for i, val_chunk_file in enumerate(val_chunks):
        chunk = np.load(val_chunk_file)
        X_val_chunk = chunk['X']
        y_val_chunk = chunk['y']

        chunk_predictions = model.predict(X_val_chunk, batch_size=4, verbose=0)
        chunk_pred_classes = np.argmax(chunk_predictions, axis=1)
        chunk_pred_proba = chunk_predictions[:, 1]
        chunk_true_classes = np.argmax(y_val_chunk, axis=1)

        final_predictions.extend(chunk_pred_classes)
        final_probabilities.extend(chunk_pred_proba)
        final_true_labels.extend(chunk_true_classes)

        from sklearn.metrics import f1_score, confusion_matrix
        chunk_minority_f1 = f1_score(chunk_true_classes, chunk_pred_classes, pos_label=1, zero_division=0)
        
        try:
            if len(np.unique(chunk_true_classes)) > 1:
                chunk_roc_auc = roc_auc_score(chunk_true_classes, chunk_pred_proba)
            else:
                chunk_roc_auc = float('nan')
        except:
            chunk_roc_auc = float('nan')

        del chunk, X_val_chunk, y_val_chunk, chunk_predictions
        clear_memory()

    from sklearn.metrics import classification_report, confusion_matrix, f1_score

    final_minority_f1 = f1_score(final_true_labels, final_predictions, pos_label=1, zero_division=0)
    
    try:
        if len(np.unique(final_true_labels)) > 1:
            final_roc_auc = roc_auc_score(final_true_labels, final_probabilities)
        else:
            final_roc_auc = float('nan')
    except:
        final_roc_auc = float('nan')

    final_cm = confusion_matrix(final_true_labels, final_predictions)

    y_pred_proba = np.array(final_probabilities)
    y_true_classes = np.array(final_true_labels)

    best_f1 = 0
    best_threshold = 0.5

    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        y_pred_thresh = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_true_classes, y_pred_thresh, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

for chunk_file in train_chunks + val_chunks:
    if os.path.exists(chunk_file):
        os.remove(chunk_file)

if os.path.exists(TEMP_DIR):
    os.rmdir(TEMP_DIR)

del model, train_generator, val_generator
clear_memory()


train_generator = UltraMemoryEfficientGenerator(
    train_chunks, batch_size=8, shuffle=True, augment=True
)
val_generator = UltraMemoryEfficientGenerator(
    val_chunks, batch_size=8, shuffle=False, augment=False
)


def create_balanced_cnn_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(2, activation='softmax')
    ])
    
    return model

with tf.device('/GPU:0' if gpus else '/CPU:0'):
    model = create_balanced_cnn_model()

    def balanced_focal_loss(y_true, y_pred, alpha=0.7, gamma=2.0):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        ce_loss = -y_true * tf.math.log(y_pred)
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        focal_weight = tf.pow(1 - p_t, gamma)
        
        alpha_t = tf.where(tf.equal(y_true[:, 1:2], 1), alpha, 1 - alpha)
        alpha_t = tf.concat([1 - alpha_t, alpha_t], axis=1)
        
        focal_loss = alpha_t * focal_weight * ce_loss
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=1))

    def simple_balanced_loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        ce_loss = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=1)
        
        class_weights = tf.reduce_sum(y_true * [1.0, 1.5], axis=1)
        
        return tf.reduce_mean(ce_loss * class_weights)

    class UltraSimpleCallback(tf.keras.callbacks.Callback):
        def __init__(self, val_generator, val_chunks):
            self.val_generator = val_generator
            self.val_chunks = val_chunks
            self.best_loss = float('inf')
            self.best_minority_f1 = 0
            self.best_roc_auc = 0
            self.patience = 0
            self.max_patience = 7
            self.epoch_count = 0

        def on_epoch_end(self, epoch, logs=None):
            self.epoch_count += 1
            current_loss = logs.get('val_loss', float('inf'))

            minority_f1, roc_auc = self._calculate_metrics()

            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.patience = 0
            else:
                self.patience += 1

            if minority_f1 > self.best_minority_f1:
                self.best_minority_f1 = minority_f1

            if roc_auc > self.best_roc_auc:
                self.best_roc_auc = roc_auc

            if self.patience >= self.max_patience:
                self.model.stop_training = True

        def _calculate_metrics(self):
            all_predictions = []
            all_probabilities = []
            all_true_labels = []

            for i, val_chunk_file in enumerate(self.val_chunks):
                chunk = np.load(val_chunk_file)
                X_val_chunk = chunk['X']
                y_val_chunk = chunk['y']

                chunk_predictions = self.model.predict(X_val_chunk, batch_size=4, verbose=0)
                chunk_pred_classes = np.argmax(chunk_predictions, axis=1)
                chunk_pred_proba = chunk_predictions[:, 1]
                chunk_true_classes = np.argmax(y_val_chunk, axis=1)

                all_predictions.extend(chunk_pred_classes)
                all_probabilities.extend(chunk_pred_proba)
                all_true_labels.extend(chunk_true_classes)

                from sklearn.metrics import f1_score, confusion_matrix
                chunk_minority_f1 = f1_score(chunk_true_classes, chunk_pred_classes, pos_label=1, zero_division=0)
                
                try:
                    if len(np.unique(chunk_true_classes)) > 1:
                        chunk_roc_auc = roc_auc_score(chunk_true_classes, chunk_pred_proba)
                    else:
                        chunk_roc_auc = float('nan')
                except:
                    chunk_roc_auc = float('nan')

                del chunk, X_val_chunk, y_val_chunk, chunk_predictions
                clear_memory()

            overall_minority_f1 = f1_score(all_true_labels, all_predictions, pos_label=1, zero_division=0)
            
            try:
                if len(np.unique(all_true_labels)) > 1:
                    overall_roc_auc = roc_auc_score(all_true_labels, all_probabilities)
                else:
                    overall_roc_auc = float('nan')
            except:
                overall_roc_auc = float('nan')

            return overall_minority_f1, overall_roc_auc

    model.compile(
        optimizer=optimizers.Adam(learning_rate=5e-5),
        loss=simple_balanced_loss,
        metrics=['accuracy']
    )

    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=3, 
        min_lr=1e-7,
        verbose=1
    )

    ultra_callback = UltraSimpleCallback(val_generator, val_chunks)

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=20,
        callbacks=[ultra_callback, lr_scheduler],
        verbose=1
    )

    final_predictions = []
    final_probabilities = []
    final_true_labels = []

    for i, val_chunk_file in enumerate(val_chunks):
        chunk = np.load(val_chunk_file)
        X_val_chunk = chunk['X']
        y_val_chunk = chunk['y']

        chunk_predictions = model.predict(X_val_chunk, batch_size=4, verbose=0)
        chunk_pred_classes = np.argmax(chunk_predictions, axis=1)
        chunk_pred_proba = chunk_predictions[:, 1]
        chunk_true_classes = np.argmax(y_val_chunk, axis=1)

        final_predictions.extend(chunk_pred_classes)
        final_probabilities.extend(chunk_pred_proba)
        final_true_labels.extend(chunk_true_classes)

        from sklearn.metrics import f1_score, confusion_matrix
        chunk_minority_f1 = f1_score(chunk_true_classes, chunk_pred_classes, pos_label=1, zero_division=0)
        
        try:
            if len(np.unique(chunk_true_classes)) > 1:
                chunk_roc_auc = roc_auc_score(chunk_true_classes, chunk_pred_proba)
            else:
                chunk_roc_auc = float('nan')
        except:
            chunk_roc_auc = float('nan')

        del chunk, X_val_chunk, y_val_chunk, chunk_predictions
        clear_memory()

    from sklearn.metrics import classification_report, confusion_matrix, f1_score

    final_minority_f1 = f1_score(final_true_labels, final_predictions, pos_label=1, zero_division=0)
    
    try:
        if len(np.unique(final_true_labels)) > 1:
            final_roc_auc = roc_auc_score(final_true_labels, final_probabilities)
        else:
            final_roc_auc = float('nan')
    except:
        final_roc_auc = float('nan')

    final_cm = confusion_matrix(final_true_labels, final_predictions)

    y_pred_proba = np.array(final_probabilities)
    y_true_classes = np.array(final_true_labels)

    best_f1 = 0
    best_threshold = 0.5

    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        y_pred_thresh = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_true_classes, y_pred_thresh, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

for chunk_file in train_chunks + val_chunks:
    if os.path.exists(chunk_file):
        os.remove(chunk_file)

if os.path.exists(TEMP_DIR):
    os.rmdir(TEMP_DIR)

del model, train_generator, val_generator
clear_memory()



feature_cols = [
    'filtered_acc_x', 'filtered_acc_y', 'filtered_acc_z'
]

max_length = 300
step_size = 300
fs = 100
num_classes = 2

def create_enhanced_timeseries_features(window_signals):
    if window_signals.shape[0] != max_length:
        if window_signals.shape[0] < max_length:
            pad_len = max_length - window_signals.shape[0]
            window_signals = np.pad(window_signals, ((0, pad_len), (0, 0)), mode='constant')
        else:
            window_signals = window_signals[:max_length, :]

    signals_T = window_signals.T
    
    enhanced_features = [window_signals]
    
    acc_magnitude = np.sqrt(np.sum(window_signals**2, axis=1, keepdims=True))
    
    acc_x, acc_y, acc_z = window_signals[:, 0], window_signals[:, 1], window_signals[:, 2]
    tilt_x = np.arctan2(acc_y, np.sqrt(acc_x**2 + acc_z**2)).reshape(-1, 1)
    tilt_y = np.arctan2(-acc_x, np.sqrt(acc_y**2 + acc_z**2)).reshape(-1, 1)
    
    enhanced_features.extend([acc_magnitude, tilt_x, tilt_y])
    
    jerk_signals = np.diff(window_signals, axis=0, prepend=window_signals[0:1])
    jerk_magnitude = np.sqrt(np.sum(jerk_signals**2, axis=1, keepdims=True))
    
    enhanced_features.extend([jerk_signals, jerk_magnitude])
    
    window_size = 30
    rolling_features = []
    
    for i in range(3):
        sig = window_signals[:, i]
        
        rolling_mean = np.convolve(sig, np.ones(window_size)/window_size, mode='same')
        
        rolling_std = np.array([np.std(sig[max(0, j-window_size//2):min(len(sig), j+window_size//2)])
                               for j in range(len(sig))])
        
        rolling_energy = np.convolve(sig**2, np.ones(window_size)/window_size, mode='same')
        
        rolling_features.extend([rolling_mean.reshape(-1, 1), 
                               rolling_std.reshape(-1, 1), 
                               rolling_energy.reshape(-1, 1)])
    
    enhanced_features.extend(rolling_features)
    
    fft_window_size = 64
    hop_length = 8
    freq_features = []
    
    for i in range(3):
        sig = signals_T[i]
        
        fft_feature_timeline = []
        for start in range(0, len(sig) - fft_window_size + 1, hop_length):
            window_sig = sig[start:start + fft_window_size]
            windowed_sig = window_sig * signal.windows.hann(fft_window_size)
            fft_result = np.fft.fft(windowed_sig)
            fft_magnitude = np.abs(fft_result)
            
            meaningful_freqs = fft_window_size // 8
            fft_focused = fft_magnitude[:meaningful_freqs]
            
            fft_mean = np.mean(fft_focused)
            fft_max = np.max(fft_focused)
            fft_energy = np.sum(fft_focused**2)
            
            fft_feature_timeline.append([fft_mean, fft_max, fft_energy])
        
        fft_timeline = np.array(fft_feature_timeline)
        
        original_indices = np.linspace(0, len(fft_timeline)-1, len(fft_timeline))
        target_indices = np.linspace(0, len(fft_timeline)-1, max_length)
        
        interp_features = []
        for j in range(3):
            interp_feature = np.interp(target_indices, original_indices, fft_timeline[:, j])
            interp_features.append(interp_feature.reshape(-1, 1))
        
        freq_features.extend(interp_features)
    
    enhanced_features.extend(freq_features)
    
    wavelet_features = []
    wavelet_window_size = 64
    
    for i in range(3):
        sig = signals_T[i]
        
        wavelet_timeline = []
        for start in range(0, len(sig) - wavelet_window_size + 1, hop_length):
            window_sig = sig[start:start + wavelet_window_size]
            
            scales = [2, 4, 8, 16]
            wavelet_energy = []
            
            for scale in scales:
                wavelet_kernel = signal.ricker(scale * 4, scale)
                if len(wavelet_kernel) <= len(window_sig):
                    convolved = np.convolve(window_sig, wavelet_kernel, mode='same')
                    energy = np.sum(convolved**2)
                else:
                    energy = 0
                wavelet_energy.append(energy)
            
            wavelet_timeline.append(wavelet_energy)
        
        wavelet_timeline = np.array(wavelet_timeline)
        
        interp_wavelet_features = []
        for j in range(len(scales)):
            if len(wavelet_timeline) > 0:
                original_indices = np.linspace(0, len(wavelet_timeline)-1, len(wavelet_timeline))
                target_indices = np.linspace(0, len(wavelet_timeline)-1, max_length)
                interp_feature = np.interp(target_indices, original_indices, wavelet_timeline[:, j])
            else:
                interp_feature = np.zeros(max_length)
            interp_wavelet_features.append(interp_feature.reshape(-1, 1))
        
        wavelet_features.extend(interp_wavelet_features)
    
    enhanced_features.extend(wavelet_features)
    
    spectral_features = []
    
    for i in range(3):
        sig = signals_T[i]
        
        f, t, Sxx = spectrogram(sig, fs=fs, nperseg=64, noverlap=32)
        
        bands = {
            'low': (0, 5),
            'mid': (5, 15),
            'high': (15, 30)
        }
        
        band_powers = []
        for band_name, (low_freq, high_freq) in bands.items():
            freq_mask = (f >= low_freq) & (f <= high_freq)
            if np.any(freq_mask):
                band_power = np.mean(Sxx[freq_mask, :], axis=0)
                
                if len(band_power) != max_length:
                    original_indices = np.linspace(0, len(band_power)-1, len(band_power))
                    target_indices = np.linspace(0, len(band_power)-1, max_length)
                    band_power = np.interp(target_indices, original_indices, band_power)
                
                band_powers.append(band_power.reshape(-1, 1))
            else:
                band_powers.append(np.zeros((max_length, 1)))
        
        spectral_features.extend(band_powers)
    
    enhanced_features.extend(spectral_features)
    
    combined_features = np.concatenate(enhanced_features, axis=1)
    
    return combined_features

def augment_enhanced_signals(signals, augmentation_type="noise"):
    augmented = signals.copy()
    
    if augmentation_type == "noise":
        noise_std = 0.01 * np.std(signals, axis=0, keepdims=True)
        noise = np.random.normal(0, noise_std, signals.shape)
        augmented = signals + noise
        
    elif augmentation_type == "time_shift":
        shift_amount = np.random.randint(-10, 11)
        augmented = np.roll(signals, shift_amount, axis=0)
        
    elif augmentation_type == "amplitude_scale":
        scale_factor = np.random.uniform(0.9, 1.1)
        augmented = signals * scale_factor
    
    return augmented

data_records = []
minority_count = 0
majority_count = 0

augmentation_types = ["noise", "time_shift", "amplitude_scale"]

for (pid, fid), group_data in df.groupby(['participant_full_id', 'file_id']):
    group_data = group_data.sort_values('elapsed_time')

    if hasattr(group_data[feature_cols], 'to_pandas'):
        signals = group_data[feature_cols].to_pandas().values
        label_values = group_data['Potential_FoF'].to_pandas().values
    else:
        signals = group_data[feature_cols].values
        label_values = group_data['Potential_FoF'].values

    scaler = StandardScaler()
    signals = scaler.fit_transform(signals)

    T = signals.shape[0]

    start_positions = range(0, T, step_size)

    for start in start_positions:
        end = start + max_length

        if end > T:
            window_signals = signals[start:T]
            window_labels = label_values[start:T]
            
            if window_signals.shape[0] < max_length:
                pad_len = max_length - window_signals.shape[0]
                window_signals = np.pad(window_signals, ((0, pad_len), (0, 0)), mode='constant')
                window_labels = np.pad(window_labels, (0, pad_len), mode='constant', constant_values=0)
        else:
            window_signals = signals[start:end]
            window_labels = label_values[start:end]

        minority_samples = np.sum(window_labels == 1)
        total_samples = len(window_labels)
        minority_ratio = minority_samples / total_samples

        enhanced_signals = create_enhanced_timeseries_features(window_signals)

        if minority_ratio > 0.3:
            window_label = 1
            minority_count += 1

            data_records.append((enhanced_signals, window_label))

            for aug_type in augmentation_types:
                aug_signals = augment_enhanced_signals(enhanced_signals, aug_type)
                data_records.append((aug_signals, window_label))
                minority_count += 1

        else:
            window_label = 0
            majority_count += 1
            data_records.append((enhanced_signals, window_label))

X_enhanced_timeseries = np.array([rec[0] for rec in data_records])
y_labels = np.array([rec[1] for rec in data_records])
y_onehot = tf.keras.utils.to_categorical(y_labels, num_classes=num_classes)

X_images = X_enhanced_timeseries


class UltraMemoryEfficientGenerator(tf.keras.utils.Sequence):
    def __init__(self, chunk_files, batch_size=8, shuffle=True, augment=False, for_lstm=True):
        super().__init__()
        self.chunk_files = chunk_files
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.for_lstm = for_lstm
        self.current_chunk_data = None
        self.current_chunk_index = 0
        self.indices_in_chunk = []
        self.chunk_pointer = 0
        self._load_next_chunk()

    def _load_next_chunk(self):
        if self.chunk_pointer >= len(self.chunk_files):
            self.chunk_pointer = 0
            
        chunk_file = self.chunk_files[self.chunk_pointer]
        
        chunk = np.load(chunk_file)
        X_chunk = chunk['X']
        y_chunk = chunk['y']
        
        if self.for_lstm and len(X_chunk.shape) == 4:
            batch_size = X_chunk.shape[0]
            converted_data = []
            
            for i in range(batch_size):
                img = X_chunk[i]
                
                start_row = img.shape[0] // 3
                end_row = start_row + 100
                
                time_series_data = img[start_row:end_row, :, :].mean(axis=1)
                
                if time_series_data.shape[0] != 300:
                    from scipy.interpolate import interp1d
                    old_indices = np.linspace(0, time_series_data.shape[0]-1, time_series_data.shape[0])
                    new_indices = np.linspace(0, time_series_data.shape[0]-1, 300)
                    
                    interpolated = np.zeros((300, 3))
                    for channel in range(3):
                        f = interp1d(old_indices, time_series_data[:, channel], kind='linear')
                        interpolated[:, channel] = f(new_indices)
                    
                    time_series_data = interpolated
                
                time_series_data = (time_series_data - time_series_data.mean()) / (time_series_data.std() + 1e-8)
                
                converted_data.append(time_series_data)
            
            X_chunk = np.array(converted_data).astype(np.float32)
            
        elif self.for_lstm and len(X_chunk.shape) == 3:
            pass
        
        self.current_chunk_data = (X_chunk, y_chunk)
        self.indices_in_chunk = list(range(len(X_chunk)))
        
        if self.shuffle:
            np.random.shuffle(self.indices_in_chunk)
            
        self.chunk_pointer += 1

    def __len__(self):
        if self.current_chunk_data is None:
            return 0
        return (len(self.current_chunk_data[0]) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.indices_in_chunk))
        
        if start_idx >= len(self.indices_in_chunk):
            self._load_next_chunk()
            start_idx = 0
            end_idx = min(self.batch_size, len(self.indices_in_chunk))
        
        batch_indices = self.indices_in_chunk[start_idx:end_idx]
        X_batch = self.current_chunk_data[0][batch_indices]
        y_batch = self.current_chunk_data[1][batch_indices]
        
        if self.augment and self.for_lstm:
            augmented_X = []
            for sample in X_batch:
                noise = np.random.normal(0, 0.01, sample.shape)
                augmented_sample = sample + noise
                augmented_X.append(augmented_sample)
            X_batch = np.array(augmented_X)
            
        return X_batch, y_batch

def use_real_timeseries_data(X_timeseries, y_onehot, batch_size=32, test_size=0.2):
    from sklearn.model_selection import train_test_split
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_timeseries, y_onehot, 
        test_size=test_size, 
        random_state=42, 
        stratify=np.argmax(y_onehot, axis=1)
    )
    
    return X_train, X_val, y_train, y_val

def create_proper_lstm_chunks(X_timeseries, y_onehot, chunk_size=1000, test_size=0.2):
    import tempfile
    import os
    from sklearn.model_selection import train_test_split
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_timeseries, y_onehot, 
        test_size=test_size, 
        random_state=42, 
        stratify=np.argmax(y_onehot, axis=1)
    )
    
    TEMP_DIR = tempfile.mkdtemp(prefix='proper_lstm_chunks_')
    
    train_chunks = []
    for i in range(0, len(X_train), chunk_size):
        end_idx = min(i + chunk_size, len(X_train))
        
        X_chunk = X_train[i:end_idx]
        y_chunk = y_train[i:end_idx]
        
        chunk_file = os.path.join(TEMP_DIR, f'train_chunk_{len(train_chunks):03d}.npz')
        np.savez_compressed(chunk_file, X=X_chunk, y=y_chunk)
        train_chunks.append(chunk_file)
    
    val_chunks = []
    for i in range(0, len(X_val), chunk_size):
        end_idx = min(i + chunk_size, len(X_val))
        
        X_chunk = X_val[i:end_idx]
        y_chunk = y_val[i:end_idx]
        
        chunk_file = os.path.join(TEMP_DIR, f'val_chunk_{len(val_chunks):03d}.npz')
        np.savez_compressed(chunk_file, X=X_chunk, y=y_chunk)
        val_chunks.append(chunk_file)
    
    return train_chunks, val_chunks, TEMP_DIR

train_chunks, val_chunks, temp_dir = create_proper_lstm_chunks(X_enhanced_timeseries, y_onehot)
train_generator = UltraMemoryEfficientGenerator(train_chunks, batch_size=8, for_lstm=True)
val_generator = UltraMemoryEfficientGenerator(val_chunks, batch_size=8, for_lstm=True)


train_generator = UltraMemoryEfficientGenerator(
    train_chunks, batch_size=8, shuffle=True, augment=True
)
val_generator = UltraMemoryEfficientGenerator(
    val_chunks, batch_size=8, shuffle=False, augment=False
)


def create_ultra_simple_model():
    model = models.Sequential([
        layers.LSTM(64, return_sequences=True,  input_shape=(300, X_enhanced_timeseries.shape[2]), dropout=0.2, recurrent_dropout=0.2),
        layers.LSTM(32, return_sequences=False, dropout=0.2, recurrent_dropout=0.2),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(2, activation='softmax')
    ])
    
    return model

with tf.device('/GPU:0' if gpus else '/CPU:0'):
    model = create_ultra_simple_model()

    def balanced_loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        ce_loss = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=1)
        
        class_weights = tf.reduce_sum(y_true * [1.0, 2.0], axis=1)
        
        return tf.reduce_mean(ce_loss * class_weights)

    class UltraSimpleCallback(tf.keras.callbacks.Callback):
        def __init__(self, val_generator, val_chunks):
            self.val_generator = val_generator
            self.val_chunks = val_chunks
            self.best_loss = float('inf')
            self.best_minority_f1 = 0
            self.best_roc_auc = 0
            self.patience = 0
            self.max_patience = 5
            self.epoch_count = 0

        def on_epoch_end(self, epoch, logs=None):
            self.epoch_count += 1
            current_loss = logs.get('val_loss', float('inf'))

            minority_f1, roc_auc = self._calculate_metrics()

            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.patience = 0
            else:
                self.patience += 1

            if minority_f1 > self.best_minority_f1:
                self.best_minority_f1 = minority_f1

            if roc_auc > self.best_roc_auc:
                self.best_roc_auc = roc_auc

            if self.patience >= self.max_patience:
                self.model.stop_training = True

        def _calculate_metrics(self):
            all_predictions = []
            all_probabilities = []
            all_true_labels = []

            for i, val_chunk_file in enumerate(self.val_chunks):
                chunk = np.load(val_chunk_file)
                X_val_chunk = chunk['X']
                y_val_chunk = chunk['y']

                if len(X_val_chunk.shape) == 4:
                    batch_size = X_val_chunk.shape[0]
                    X_val_chunk = np.random.randn(batch_size, 300, 3).astype(np.float32)

                chunk_predictions = self.model.predict(X_val_chunk, batch_size=4, verbose=0)
                chunk_pred_classes = np.argmax(chunk_predictions, axis=1)
                chunk_pred_proba = chunk_predictions[:, 1]
                chunk_true_classes = np.argmax(y_val_chunk, axis=1)

                all_predictions.extend(chunk_pred_classes)
                all_probabilities.extend(chunk_pred_proba)
                all_true_labels.extend(chunk_true_classes)

                from sklearn.metrics import f1_score, confusion_matrix
                chunk_minority_f1 = f1_score(chunk_true_classes, chunk_pred_classes, pos_label=1, zero_division=0)
                
                try:
                    if len(np.unique(chunk_true_classes)) > 1:
                        chunk_roc_auc = roc_auc_score(chunk_true_classes, chunk_pred_proba)
                    else:
                        chunk_roc_auc = float('nan')
                except:
                    chunk_roc_auc = float('nan')

                del chunk, X_val_chunk, y_val_chunk, chunk_predictions
                clear_memory()

            overall_minority_f1 = f1_score(all_true_labels, all_predictions, pos_label=1, zero_division=0)
            
            try:
                if len(np.unique(all_true_labels)) > 1:
                    overall_roc_auc = roc_auc_score(all_true_labels, all_probabilities)
                else:
                    overall_roc_auc = float('nan')
            except:
                overall_roc_auc = float('nan')

            return overall_minority_f1, overall_roc_auc

    model.compile(
        optimizer=optimizers.Adam(learning_rate=5e-4),
        loss=balanced_loss,
        metrics=['accuracy']
    )

    ultra_callback = UltraSimpleCallback(val_generator, val_chunks)

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=15,
        callbacks=[ultra_callback],
        verbose=1
    )

    final_predictions = []
    final_probabilities = []
    final_true_labels = []

    for i, val_chunk_file in enumerate(val_chunks):
        chunk = np.load(val_chunk_file)
        X_val_chunk = chunk['X']
        y_val_chunk = chunk['y']

        if len(X_val_chunk.shape) == 4:
            batch_size = X_val_chunk.shape[0]
            X_val_chunk = np.random.randn(batch_size, 300, 3).astype(np.float32)

        chunk_predictions = model.predict(X_val_chunk, batch_size=4, verbose=0)
        chunk_pred_classes = np.argmax(chunk_predictions, axis=1)
        chunk_pred_proba = chunk_predictions[:, 1]
        chunk_true_classes = np.argmax(y_val_chunk, axis=1)

        final_predictions.extend(chunk_pred_classes)
        final_probabilities.extend(chunk_pred_proba)
        final_true_labels.extend(chunk_true_classes)

        from sklearn.metrics import f1_score, confusion_matrix
        chunk_minority_f1 = f1_score(chunk_true_classes, chunk_pred_classes, pos_label=1, zero_division=0)
        
        try:
            if len(np.unique(chunk_true_classes)) > 1:
                chunk_roc_auc = roc_auc_score(chunk_true_classes, chunk_pred_proba)
            else:
                chunk_roc_auc = float('nan')
        except:
            chunk_roc_auc = float('nan')

        del chunk, X_val_chunk, y_val_chunk, chunk_predictions
        clear_memory()

    from sklearn.metrics import classification_report, confusion_matrix, f1_score

    final_minority_f1 = f1_score(final_true_labels, final_predictions, pos_label=1, zero_division=0)
    
    try:
        if len(np.unique(final_true_labels)) > 1:
            final_roc_auc = roc_auc_score(final_true_labels, final_probabilities)
        else:
            final_roc_auc = float('nan')
    except:
        final_roc_auc = float('nan')

    final_cm = confusion_matrix(final_true_labels, final_predictions)

    y_pred_proba = np.array(final_probabilities)
    y_true_classes = np.array(final_true_labels)

    best_f1 = 0
    best_threshold = 0.5

    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        y_pred_thresh = (y_pred_proba >= threshold).astype(int)

        f1 = f1_score(y_true_classes, y_pred_thresh, pos_label=1, zero_division=0)
        cm_thresh = confusion_matrix(y_true_classes, y_pred_thresh)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

for chunk_file in train_chunks + val_chunks:
    if os.path.exists(chunk_file):
        os.remove(chunk_file)

if os.path.exists(TEMP_DIR):
    os.rmdir(TEMP_DIR)

del model, train_generator, val_generator
clear_memory()




