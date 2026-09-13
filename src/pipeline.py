"""tf.data pipeline: loads, resizes, and normalizes images for training."""
import os
import numpy as np
import tensorflow as tf
from PIL import Image

def load_and_label(fname, label, benchmark, split, root, resolution=224):
    """Loads one image, resizes/normalizes it, and converts its label to 0/1.
    Called through tf.py_function, so all arguments arrive as tensors and
    must be decoded to plain Python values first."""
    fname = fname.numpy().decode('utf-8')
    label = label.numpy().decode('utf-8')
    benchmark = benchmark.numpy().decode('utf-8')
    split = split.numpy().decode('utf-8')
    root = root.numpy().decode('utf-8')

    path = os.path.join(root, benchmark, split, fname)
    img = Image.open(path)
    img = img.resize((resolution, resolution))
    img_arr = np.array(img) / 255
    img_arr = np.expand_dims(img_arr, axis=-1)

    labelN = 1 if label == "HS" else 0
    return img_arr.astype(np.float32), np.int32(labelN)


def build_dataset(df, benchmark, split, root, batch_size=32, shuffle=False, resolution=224):
    """Builds a batched, prefetched tf.data.Dataset from a filtered fold DataFrame."""
    filenames = df["filename"].values
    labels = df["label"].values
    n = len(df)

    benchmark_col = np.repeat(benchmark, n)
    split_col = np.repeat(split, n)
    root_col = np.repeat(root, n)

    ds = tf.data.Dataset.from_tensor_slices((filenames, labels, benchmark_col, split_col, root_col))

    def wrapper(fname, label, bmk, spl, rt):
        img, lbl = tf.py_function(
            func=lambda f, l, b, s, r: load_and_label(f, l, b, s, r, resolution),
            inp=[fname, label, bmk, spl, rt],
            Tout=[tf.float32, tf.int32]
        )
        img.set_shape((resolution, resolution, 1))
        lbl.set_shape(())
        return img, lbl

    if shuffle:
        ds = ds.shuffle(buffer_size=n)
    ds = ds.map(wrapper).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
