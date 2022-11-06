import tensorflow as tf


def l2_loss(hr_batch, sr_batch, lambda_l2=0.5):
    """Simple l2 loss

    Args:
        hr_batch (tf_tensor): batch of ground truth grids
        sr_batch (tf_tensor): batch of super resolved grids
        lambda_l2 (float, optional): Weighting of the norm. Defaults to 0.5.
    Returns:
        loss (float): average l2 loss
    """

    difference = hr_batch - sr_batch
    l2 = tf.norm(difference, ord=2, axis=1)
    loss = lambda_l2 * l2
    return loss
