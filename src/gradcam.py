"""Grad-CAM: visualize which regions of an image drove a trained CNN's prediction."""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from data import load_clip


def build_grad_model(model, last_conv_layer_name, input_shape=(224, 224, 1)):
    """Rebuilds a model exposing the given layer's output alongside the final
    prediction. Uses a manual layer-by-layer graph rebuild (not model.output)
    because Keras' .output attribute access fails on loaded Sequential models.
    training=False is set on EVERY layer call, not just the outer model call --
    otherwise BatchNorm/Dropout silently run in training-mode behavior."""
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    conv_output = None
    for layer in model.layers:
        x = layer(x, training=False)
        if layer.name == last_conv_layer_name:
            conv_output = x
    return tf.keras.Model(inputs=inputs, outputs=[conv_output, x])


def load_for_gradcam(benchmark, split, filename, root, resolution=224):
    """Loads and preprocesses a single image for Grad-CAM (adds channel + batch dims,
    since this runs outside the tf.data pipeline that normally handles batching)."""
    img = load_clip(root, benchmark, split, filename)
    img = img.resize((resolution, resolution))
    img_arr = np.array(img) / 255
    img_arr = np.expand_dims(img_arr, axis=-1)
    img_arr = np.expand_dims(img_arr, axis=0)
    return img_arr.astype(np.float32)


def make_gradcam_heatmap(grad_model, img_array):
    """Returns (heatmap, prediction) for one image. training=False on the grad_model
    call is required for consistent inference-mode behavior."""
    with tf.GradientTape() as tape:
        conv_out, pred = grad_model(img_array, training=False)
        target = pred[:, 0]

    grads = tape.gradient(target, conv_out)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    weights_reshaped = tf.reshape(weights, (1, 1, 1, -1))
    heatmap = tf.reduce_sum(conv_out * weights_reshaped, axis=-1)
    heatmap = tf.nn.relu(heatmap)
    return heatmap, pred.numpy()[0][0]


def show_gradcam(original_img_array, heatmap, title=""):
    """Renders one heatmap overlaid on its original image. Does NOT call plt.show()
    internally -- caller controls that, so multiple calls can share one figure."""
    heatmap = tf.image.resize(heatmap[..., tf.newaxis], (224, 224))
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy().squeeze()

    plt.imshow(original_img_array.squeeze(), cmap='gray')
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.title(title)
    plt.axis('off')
