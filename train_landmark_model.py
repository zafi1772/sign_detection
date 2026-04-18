"""
train_landmark_model.py — Build a fast landmark-based sign classifier.

Scans  dataset/landmarks/*.npy  (one file per word),
trains a small Keras MLP on the 63-feature landmark vectors,
and saves the model + label map to  models/.

Run after every new collect_signs.py session:
    python train_landmark_model.py
"""

import os, json, glob
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

LANDMARK_DIR = os.path.join('dataset', 'landmarks')
MODEL_OUT    = os.path.join('models', 'landmark_model.keras')
LABELS_OUT   = os.path.join('models', 'landmark_labels.json')


def load_dataset():
    files = sorted(glob.glob(os.path.join(LANDMARK_DIR, '*.npy')))
    if not files:
        raise FileNotFoundError(
            f'No .npy files found in {LANDMARK_DIR}.\n'
            'Run collect_signs.py first to record sign data.')

    X, y, labels = [], [], []
    for idx, path in enumerate(files):
        word    = os.path.splitext(os.path.basename(path))[0]
        samples = np.load(path, allow_pickle=True)
        print(f'  [{idx:2d}] {word:20s}  {len(samples):4d} samples')
        X.extend(samples)
        y.extend([idx] * len(samples))
        labels.append(word)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), labels


def build_model(n_features, n_classes):
    import tensorflow as tf
    from tf_keras import layers, models, regularizers

    model = models.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(256, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(n_classes, activation='softmax'),
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def main():
    print('\n===  Loading dataset  ===')
    X, y, labels = load_dataset()
    n_classes = len(labels)
    print(f'\n  {len(X)} total samples, {n_classes} classes\n')

    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    # 90/10 train-val split
    split = int(0.9 * len(X))
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]

    print('===  Training  ===')
    from tf_keras.callbacks import EarlyStopping, ReduceLROnPlateau

    model = build_model(X.shape[1], n_classes)
    model.summary()

    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=7, verbose=1),
    ]

    model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=150,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
    )

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f'\n  Validation accuracy: {val_acc*100:.1f}%')

    model.save(MODEL_OUT)
    with open(LABELS_OUT, 'w', encoding='utf-8') as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    print(f'\n  Model saved  → {MODEL_OUT}')
    print(f'  Labels saved → {LABELS_OUT}')
    print(f'\n  Words recognised: {", ".join(labels)}')
    print('\n  Restart the app — it will automatically use the new model.')


if __name__ == '__main__':
    main()
