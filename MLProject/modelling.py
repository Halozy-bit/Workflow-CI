"""modelling.py — Kriteria 3 (MLProject entry point)

Versi untuk MLflow Project / CI. Menjalankan training MobileNetV2 dan
logging ke MLflow (tracking store lokal di dalam run 'mlflow run').

Dijalankan otomatis oleh CI via:
    mlflow run MLProject --env-manager=local
"""

import os
import argparse
import numpy as np
import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

IMG = 224
SEED = 42


def load_data(data_dir):
    tr = np.load(os.path.join(data_dir, 'train.npz'))
    va = np.load(os.path.join(data_dir, 'val.npz'))
    te = np.load(os.path.join(data_dir, 'test.npz'))
    return tr['X'], tr['y'], va['X'], va['y'], te['X'], te['y']


def build_model(lr, dropout=0.3):
    base = MobileNetV2(input_shape=(IMG, IMG, 3), include_top=False, weights='imagenet')
    base.trainable = False
    model = keras.Sequential([
        base,
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(2),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(dropout),
        layers.Dense(128, activation='relu'),
        layers.Dropout(dropout),
        layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    return model


def main(args):
    tf.random.set_seed(SEED)

    # autolog cukup untuk CI (tracking otomatis ke mlruns/ dalam run)
    mlflow.tensorflow.autolog(log_models=True)

    Xtr, ytr, Xva, yva, Xte, yte = load_data(args.data_dir)
    print(f'train={len(Xtr)} | val={len(Xva)} | test={len(Xte)}')

    neg = (ytr == 0).sum(); pos = (ytr == 1).sum()
    class_weight = {0: (neg + pos) / (2 * neg), 1: (neg + pos) / (2 * pos)}

    with mlflow.start_run(run_name='ci-training'):
        mlflow.log_params({
            'learning_rate': args.lr,
            'epochs': args.epochs,
            'model': 'MobileNetV2',
            'dataset': 'APTOS2019',
        })

        model = build_model(args.lr)
        model.fit(
            Xtr, ytr,
            validation_data=(Xva, yva),
            epochs=args.epochs,
            batch_size=32,
            class_weight=class_weight,
            verbose=1,
        )

        test_loss, test_acc = model.evaluate(Xte, yte, verbose=0)
        mlflow.log_metrics({'test_loss': test_loss, 'test_accuracy': test_acc})
        print(f'Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='aptos_preprocessing')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    main(parser.parse_args())
