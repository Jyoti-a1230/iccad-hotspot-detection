"""Binary focal loss, per Lin et al. (2017), for the hotspot detection baseline comparison.

FL(pt) = -alpha_t * (1 - pt)^gamma * log(pt)

Implemented as a factory function returning a Keras-compatible loss (y_true, y_pred) -> scalar,
so gamma/alpha can be configured per experiment without rewriting the loss itself.
"""
import tensorflow as tf


def binary_focal_loss(gamma=2.0, alpha=0.75):
    """Returns a focal loss function configured with the given gamma/alpha.

    gamma: controls down-weighting of easy (confidently-correct) examples,
           regardless of class. gamma=0 reduces this to plain weighted cross-entropy.
    alpha: class-weighting coefficient. alpha_t = alpha for the positive class (HS),
           (1 - alpha) for the negative class (NHS). alpha close to 1 favors the
           minority (positive) class, matching this project's HS-as-positive convention.
    """
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Clip predictions to avoid log(0) -> -inf, and to avoid pt exactly at 1
        # producing a zero gradient region that can stall training.
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        # pt = the model's predicted probability of the TRUE class:
        #   if y_true == 1 (HS): pt = y_pred
        #   if y_true == 0 (NHS): pt = 1 - y_pred
        pt = tf.where(tf.equal(y_true, 1.0), y_pred, 1.0 - y_pred)

        # alpha_t = alpha for HS, (1 - alpha) for NHS
        alpha_t = tf.where(tf.equal(y_true, 1.0), alpha, 1.0 - alpha)

        # FL = -alpha_t * (1 - pt)^gamma * log(pt)
        focal_weight = tf.pow(1.0 - pt, gamma)
        loss = -alpha_t * focal_weight * tf.math.log(pt)

        return tf.reduce_mean(loss)

    return loss_fn
