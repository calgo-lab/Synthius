from __future__ import annotations

from typing import TYPE_CHECKING

import tensorflow as tf
from tensorflow.keras import Model, layers

if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy as np
    import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class WGAN(Model):
    """WGAN (Wasserstein Generative Adversarial Network) model implementation.

    This class implements the WGAN model with a gradient penalty (WGAN-GP).
    It includes both the generator and discriminator networks, along with
    training and sample generation methods.

    Attributes:
        generator (Model): The generator network.
        discriminator (Model): The discriminator network.
        batch_size (int): Batch size for training.
        critic_iters (int): Number of critic iterations per generator iteration.
        lambda_gp (float): Gradient penalty coefficient.
        generator_optimizer (tf.keras.optimizers.Adam): Optimizer for the generator.
        discriminator_optimizer (tf.keras.optimizers.Adam): Optimizer for the discriminator.

    Usage Example:
    ----------------------
    ```python
    wgan = WGAN(n_features=n_features)
    train_dataset = data_batcher(data)

    wgan.train(train_dataset, num_epochs=20_000, log_interval=300, log_training=True)

    synthetic_samples = wgan.generate_samples(13_000)
    synthetic_data = pd.DataFrame(synthetic_samples, columns=data.columns

    ```
    """

    def __init__(  # noqa: PLR0913
        self: WGAN,
        n_features: int,
        base_nodes: int = 64,
        batch_size: int = 256,
        critic_iters: int = 5,
        lambda_gp: float = 10.0,
    ) -> None:
        """Initialize the WGAN model.

        Args:
            n_features (int): Number of features for the generator output.
            base_nodes (int): Number of base nodes for the discriminator layers. Defaults to 64.
            batch_size (int): Batch size for training. Defaults to 256.
            critic_iters (int): Number of critic iterations per generator iteration. Defaults to 5.
            lambda_gp (float): Gradient penalty coefficient. Defaults to 10.0.
        """
        super().__init__()
        self.generator = self.Generator(n_features)
        self.discriminator = self.Discriminator(base_nodes)
        self.batch_size = batch_size
        self.critic_iters = critic_iters
        self.lambda_gp = lambda_gp
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)

    class Generator(Model):
        """Generator model for the WGAN.

        This class defines the structure of the generator network used in the WGAN.
        """

        def __init__(self: WGAN.Generator, n_features: int) -> None:
            """Initialize the Generator model.

            Args:
                n_features (int): Number of features for the output layer.
            """
            super().__init__()
            self.dense1 = layers.Dense(n_features * 2, activation="relu")
            self.dense2 = layers.Dense(round(1.5 * n_features), activation="relu")
            self.dense3 = layers.Dense(n_features, activation="sigmoid")

        def call(self: WGAN.Generator, inputs: tf.Tensor) -> tf.Tensor:
            """Forward pass for the Generator.

            Args:
                inputs (tf.Tensor): Input tensor.

            Returns:
                tf.Tensor: Output tensor.
            """
            x = self.dense1(inputs)
            x = self.dense2(x)
            return self.dense3(x)

    class Discriminator(Model):
        """Discriminator model for the WGAN.

        This class defines the structure of the discriminator network used in the WGAN.
        """

        def __init__(self: WGAN.Discriminator, base_nodes: int) -> None:
            """Initialize the Discriminator model.

            Args:
                base_nodes (int): Number of base nodes for the layers.
            """
            super().__init__()
            self.dense1 = layers.Dense(base_nodes, activation="leaky_relu")
            self.dense2 = layers.Dense(2 * base_nodes, activation="leaky_relu")
            self.dense3 = layers.Dense(4 * base_nodes, activation="leaky_relu")
            self.dense4 = layers.Dense(1)

        def call(self: WGAN.Discriminator, inputs: tf.Tensor) -> tf.Tensor:
            """Forward pass for the Discriminator.

            Args:
                inputs (tf.Tensor): Input tensor.

            Returns:
                tf.Tensor: Output tensor.
            """
            x = self.dense1(inputs)
            x = self.dense2(x)
            x = self.dense3(x)
            return self.dense4(x)

    @tf.function
    def train_step(self: WGAN, real_data: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """Perform one training step.

        Args:
            real_data (tf.Tensor): Real data tensor.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: Generator and Discriminator loss.
        """
        noise = tf.random.normal([self.batch_size, 100], dtype=tf.float32)
        with tf.GradientTape(persistent=True) as tape:
            fake_data = self.generator(noise, training=True)
            real_data_cast = tf.cast(real_data, tf.float32)
            real_output = self.discriminator(real_data_cast, training=True)
            fake_output = self.discriminator(fake_data, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)
            gp = self.gradient_penalty(real_data_cast, fake_data)
            disc_loss += self.lambda_gp * gp

        gradients_of_generator = tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables),
        )

        return gen_loss, disc_loss

    def train(
        self: WGAN,
        train_dataset: Iterator[tf.Tensor],
        num_epochs: int = 10_000,
        log_interval: int = 500,
        *,
        log_training: bool = False,
    ) -> None:
        """Train the WGAN model.

        Args:
            train_dataset (Iterator[tf.Tensor]): Training dataset iterator.
            num_epochs (int): Number of epochs to train for. Defaults to 10_000.
            log_interval (int, optional): Interval of epochs at which to log training progress. Defaults to 500.
            log_training (bool, optional): Whether to log training progress. Defaults to False.
        """
        for epoch in range(num_epochs):
            for _ in range(self.critic_iters):
                real_data = next(train_dataset)
                g_loss, d_loss = self.train_step(real_data)

            if log_training and epoch % log_interval == 0:
                logger.info(
                    "Epoch %s, Generator loss: %s, Discriminator loss: %s",
                    epoch,
                    g_loss.numpy(),
                    d_loss.numpy(),
                )

    def generate_samples(self: WGAN, num_samples: int) -> np.ndarray:
        """Generate synthetic samples from the generator.

        Args:
            num_samples (int): Number of samples to generate.

        Returns:
            np.ndarray: Generated samples.
        """
        noise = tf.random.normal([num_samples, 100])
        synthetic_data = self.generator(noise, training=False)
        return synthetic_data.numpy()

    @staticmethod
    def discriminator_loss(real_output: tf.Tensor, fake_output: tf.Tensor) -> tf.Tensor:
        """Calculate the Discriminator loss.

        Args:
            real_output (tf.Tensor): Output from the Discriminator for real data.
            fake_output (tf.Tensor): Output from the Discriminator for fake data.

        Returns:
            tf.Tensor: Discriminator loss.
        """
        return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)

    @staticmethod
    def generator_loss(fake_output: tf.Tensor) -> tf.Tensor:
        """Calculate the Generator loss.

        Args:
            fake_output (tf.Tensor): Output from the Discriminator for fake data.

        Returns:
            tf.Tensor: Generator loss.
        """
        return -tf.reduce_mean(fake_output)

    def gradient_penalty(self: WGAN, real_data: tf.Tensor, fake_data: tf.Tensor) -> tf.Tensor:
        """Calculate the gradient penalty for WGAN-GP.

        Args:
            real_data (tf.Tensor): Real data tensor.
            fake_data (tf.Tensor): Fake data tensor.

        Returns:
            tf.Tensor: Gradient penalty.
        """
        alpha = tf.random.uniform([self.batch_size, 1], 0.0, 1.0, dtype=tf.float32)
        real_data_cast = tf.cast(real_data, tf.float32)
        diff = fake_data - real_data_cast
        interpolated = real_data_cast + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.discriminator(interpolated, training=True)
        gradients = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1]))
        return tf.reduce_mean((norm - 1.0) ** 2)


def data_batcher(data: pd.DataFrame, batch_size: int) -> Iterator[tf.Tensor]:
    """Create a data batcher for training.

    Args:
        data (pd.DataFrame): Input data as DataFrame.
        batch_size (int): Batch size. Defaults to 256.

    Returns:
        Iterator[tf.Tensor]: Data iterator.
    """
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size, drop_remainder=True).repeat()
    return iter(dataset)
