"""Training loop for the baseline (and, later, focal-loss) CNN, one benchmark at a time."""
import tensorflow as tf
from model import build_model
from pipeline import build_dataset

def run_training(benchmark, folds_df, root, loss='binary_crossentropy', epochs=50, patience=10, tag="baseline"):
    """Trains a fresh model on one benchmark. `loss` can be 'binary_crossentropy'
    (baseline) or a configured binary_focal_loss(...) instance (Day 6-7)."""
    bench_df = folds_df[folds_df["benchmark"] == benchmark]
    train_df = bench_df[bench_df["fold"] != 0]
    val_df = bench_df[bench_df["fold"] == 0]
    print(f"{benchmark}: {len(train_df)} train, {len(val_df)} val")

    train_ds = build_dataset(train_df, benchmark=benchmark, split="train", root=root, shuffle=True)
    val_ds = build_dataset(val_df, benchmark=benchmark, split="train", root=root, shuffle=False)

    model = build_model()
    model.compile(
        optimizer='adam',
        loss=loss,
        metrics=[
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='pr_auc', curve='PR'),
        ]
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f'/content/drive/MyDrive/iccad_checkpoints/{benchmark}_{tag}_{{epoch:02d}}.keras',
            save_best_only=True, monitor='val_loss'
        ),
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
    return model, history, val_ds, val_df, train_df
