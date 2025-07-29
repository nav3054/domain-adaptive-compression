import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, GroupNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.datasets import mnist
import tensorflow_datasets as tfds

from sklearn.manifold import TSNE
import umap.umap_ as umap
from skimage.metrics import structural_similarity as ssim

tf.random.set_seed(0)
np.random.seed(0)

LATENT_DIMS = 8

# Mish activation function
def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))

# sampling function using the reparameterization trick
def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# Custom SSIM loss function
def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

class AdaptiveVAE(keras.Model):
    def __init__(self, encoder, decoder, lambda_target=1.0, ssim_weight=0.5, **kwargs):
        super(AdaptiveVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.lambda_target = lambda_target # target lambda value
        self.ssim_weight = ssim_weight  # weight factor for SSIM loss
        self.current_lambda = tf.Variable(1.0, trainable=False, dtype=tf.float32)
        
        # trackers
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.bce_loss_tracker = keras.metrics.Mean(name="bce_loss")
        self.ssim_loss_tracker = keras.metrics.Mean(name="ssim_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.lambda_tracker = keras.metrics.Mean(name="lambda")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.bce_loss_tracker,
            self.ssim_loss_tracker,
            self.kl_loss_tracker,
            self.lambda_tracker,
        ]

    def call(self, inputs, training=None):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return reconstruction

    def test_step(self, data):
        if isinstance(data, tuple):
            data = data[0] 

        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)

        # MSE loss
        # mse_loss = tf.reduce_mean(tf.reduce_sum(tf.keras.losses.mse(data, reconstruction), axis=[1, 2]))

        # BCE loss
        bce_loss = tf.reduce_mean(tf.reduce_sum(tf.keras.losses.binary_crossentropy(data, reconstruction), axis=[1, 2]))

        # SSIM loss
        ssim_loss_value = ssim_loss(data, reconstruction)

        # combined reconstruction loss 
        reconstruction_loss = (1 - self.ssim_weight) * bce_loss + self.ssim_weight * ssim_loss_value
        
        kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        total_loss = reconstruction_loss + self.current_lambda * kl_loss # total loss

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.bce_loss_tracker.update_state(bce_loss)
        self.ssim_loss_tracker.update_state(ssim_loss_value)
        self.kl_loss_tracker.update_state(kl_loss)
        self.lambda_tracker.update_state(self.current_lambda)

        return {
            "val_total_loss": self.total_loss_tracker.result(),
            "val_reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "val_bce_loss": self.bce_loss_tracker.result(),
            "val_ssim_loss": self.ssim_loss_tracker.result(),
            "val_kl_loss": self.kl_loss_tracker.result(),
            "val_lambda": self.lambda_tracker.result(),
        }

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]  # Extract the actual input tensor

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)

            # MSE loss
            # mse_loss = tf.reduce_mean(tf.reduce_sum(tf.keras.losses.mse(data, reconstruction), axis=[1, 2]))

            # BCE loss
            bce_loss = tf.reduce_mean(tf.reduce_sum(tf.keras.losses.binary_crossentropy(data, reconstruction), axis=[1, 2]))

            # SSIM loss
            ssim_loss_value = ssim_loss(data, reconstruction)

            reconstruction_loss = (1 - self.ssim_weight) * bce_loss + self.ssim_weight * ssim_loss_value
            kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            total_loss = reconstruction_loss + self.current_lambda * kl_loss # total loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        epsilon = 1e-8
        lambda_update = self.lambda_target * reconstruction_loss / (kl_loss + epsilon)
        lambda_update = tf.clip_by_value(lambda_update, 0.01, 10)
        self.current_lambda.assign(lambda_update)

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.bce_loss_tracker.update_state(bce_loss)
        self.ssim_loss_tracker.update_state(ssim_loss_value)
        self.kl_loss_tracker.update_state(kl_loss)
        self.lambda_tracker.update_state(self.current_lambda)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "bce_loss": self.bce_loss_tracker.result(),
            "ssim_loss": self.ssim_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "lambda": self.lambda_tracker.result(),
        }


def residual_block(x, filters):
    shortcut = x
    x = layers.Conv2D(filters // 2, 1, padding="same")(x)
    x = GroupNormalization(groups=8)(x)
    x = layers.Activation(mish)(x)

    x = layers.Conv2D(filters // 2, 3, padding="same")(x)
    x = GroupNormalization(groups=8)(x)
    x = layers.Activation(mish)(x)

    x = layers.Conv2D(filters, 1, padding="same")(x)
    x = GroupNormalization(groups=8)(x)

    x = layers.Add()([x, shortcut])
    x = layers.Activation(mish)(x)
    return x


# Encoder function
def build_encoder(input_shape, latent_dim):
    encoder_inputs = keras.Input(shape=input_shape)

    x = layers.Conv2D(32, 5, strides=2, padding="same")(encoder_inputs) 
    x = GroupNormalization(groups=8)(x)
    x = layers.Activation(mish)(x)

    x = layers.Conv2D(64, 3, strides=2, padding="same")(x)
    x = GroupNormalization(groups=8)(x)
    x = layers.Activation(mish)(x)

    x = residual_block(x, 64)

    x = layers.Conv2D(128, 3, strides=1, padding="same")(x)
    x = GroupNormalization(groups=8)(x)
    x = layers.Activation(mish)(x)

    x = layers.Conv2D(256, 3, strides=1, padding="same")(x) 
    x = GroupNormalization(groups=8)(x)
    x = layers.Activation(mish)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation=mish)(x)  
    x = layers.Dense(128, activation=mish)(x) 

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = layers.Lambda(sampling, name="z")([z_mean, z_log_var])

    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    return encoder

# Decoder function
def build_decoder(latent_dim, output_shape):
    latent_inputs = keras.Input(shape=(latent_dim,))

    x = layers.Dense(64, activation=mish)(latent_inputs)
    x = layers.Dense(7 * 7 * 64, activation=mish)(x)
    x = layers.Reshape((7, 7, 64))(x)

    x = residual_block(x, 64)

    x = layers.Conv2DTranspose(64, 4, strides=2, padding="same")(x)
    x = GroupNormalization(groups=8)(x)
    x = layers.Activation(mish)(x)

    x = layers.Conv2DTranspose(32, 4, strides=2, padding="same")(x)
    x = GroupNormalization(groups=8)(x)
    x = layers.Activation(mish)(x)

    x = residual_block(x, 32)  # residual block

    decoder_outputs = layers.Conv2D(output_shape[-1], 5, activation="sigmoid", padding="same")(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    return decoder


# Train model function
def train_adaptive_vae(data_train, data_test=None, epochs=20, batch_size=64, latent_dim=10, ssim_weight=0.5):
    input_shape = data_train.shape[1:]
    encoder = build_encoder(input_shape, latent_dim)
    decoder = build_decoder(latent_dim, input_shape)

    vae = AdaptiveVAE(encoder, decoder, lambda_target=0.02, ssim_weight=ssim_weight)

    # COMPILING THE MODEL
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate = 1e-4), loss=lambda y_true, y_pred: 0.0)

    ###### CALLBACKS #######
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_val_total_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-9,
        verbose=1,
        mode="min"
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_val_total_loss",
        patience=8,
        mode='min',
        restore_best_weights=True
    )

    ### FITTING THE MODEL ###
    history = vae.fit(
        data_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(data_test, data_test) if data_test is not None else None,
        callbacks=[early_stopping, reduce_lr]
    )
    return vae, history



def run_mnist_example():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    # check shapes
    print("x_train shape :", x_train.shape)
    print("y_train shape :", y_train.shape)
    print("x_test shape :", x_test.shape)
    print("y_test shape :", y_test.shape)
    
    # check normalization
    print("x_train min :", x_train.min())
    print("x_train max :", x_train.max())
    print("x_test min :", x_test.min())
    print("x_test max :", x_test.max())

    EPOCHS = 500
    BATCH_SIZE = 64
    SSIM_WEIGHT = 0.7

    latent_dim = LATENT_DIMS
    return train_adaptive_vae(x_train, x_test, epochs=EPOCHS, batch_size=BATCH_SIZE, latent_dim=latent_dim, ssim_weight=SSIM_WEIGHT)

vae, history = run_mnist_example() 
# BCE-SSIM ; lambda_target = 0.02 ; latent_dims = 8



#########################################
#########################################
####      CHECK VAE PERFORMANCE      ####   
#########################################
#########################################

# to check performace of the VAE
def check_vae_performance(model, x_train, x_test, y_test, num_samples=10):

    print("[EVAL_PERF] Checking VAE performance...")

    ## Check for Overfitting
    train_loss = model.evaluate(x_train, x_train, verbose=0)
    test_loss = model.evaluate(x_test, x_test, verbose=0)

    # extract total loss (last element in the list)
    train_total_loss = train_loss[-1] 
    test_total_loss = test_loss[-1]
    print(f"Train Total Loss: {train_total_loss:.4f}, Test Total Loss: {test_total_loss:.4f}")
  
    if test_total_loss > train_total_loss + 0.1:
        print("[FIX] Possible Overfitting")
    else:
        print("No significant overfitting")
    '''  
    z_mean, z_log_var, z_sampled = model.encoder(x_test[:num_samples])
    kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)).numpy()

    print(f"KL Divergence: {kl_loss:.4f}")
    if kl_loss < 0.01:
        print("[FIX] KL loss too low. Model might be ignoring the latent space (Posterior Collapse)")
    '''

    # plot reconstructed images
    def plot_reconstructions(model, data, num_images=10):
        encoded_imgs = model.encoder(data[:num_images])
        decoded_imgs = model.decoder(encoded_imgs[2])

        plt.figure(figsize=(num_images, 2))
        for i in range(num_images):
            # Original images
            plt.subplot(2, num_images, i + 1)
            plt.imshow(data[i].squeeze(), cmap="gray")
            plt.axis("off")

            # Reconstructed images
            plt.subplot(2, num_images, num_images + i + 1)
            plt.imshow(decoded_imgs[i].numpy().squeeze(), cmap="gray")
            plt.axis("off")

        plt.suptitle("Original vs Reconstructed Images")
        plt.show()

    plot_reconstructions(model, x_test)

    # Latent Space Interpolation
    def latent_interpolation(model, data):
        z_mean, _, _ = model.encoder(data[:2])
        z1, z2 = z_mean.numpy()
        num_steps = 10
        interpolated_z = np.array([z1 * (1 - t) + z2 * t for t in np.linspace(0, 1, num_steps)])
        reconstructed_imgs = model.decoder(interpolated_z).numpy()

        plt.figure(figsize=(num_steps, 2))
        for i in range(num_steps):
            plt.subplot(1, num_steps, i + 1)
            plt.imshow(reconstructed_imgs[i].squeeze(), cmap="gray")
            plt.axis("off")
        plt.suptitle("Latent Space Interpolation")
        plt.show()

    latent_interpolation(model, x_test)
    print("[EVAL_PERF] VAE checks completed")


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
check_vae_performance(vae, x_train, x_test, y_test)



########################################
########################################
####       ALL VISUALISATIONS       ####
########################################
########################################

def run_all_visualizations(vae, history):

    plot_training_metrics(history) # plot training metrics
    plot_loss_components(history) # plot loss components
    visualize_digit_morphing(vae) # visualize digit morphing in latent space
    visualize_latent_space_tsne(vae) # visualize latent space with t-SNE

# function to plot training metrics
def plot_training_metrics(history):
    metrics = ['loss', 'reconstruction_loss', 'kl_loss', 'lambda']
    val_metrics = ['val_val_total_loss', 'val_val_reconstruction_loss', 'val_val_kl_loss', 'val_val_lambda']
    titles = ['Total Loss', 'Reconstruction Loss', 'KL Divergence Loss', 'Lambda Value']

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    colors = ['#1f77b4', '#ff7f0e']  # blue for training, orange for validation

    for i, (metric, val_metric, title) in enumerate(zip(metrics, val_metrics, titles)):
        ax = axes[i]

        # Plot training metrics
        ax.plot(history.history[metric], color=colors[0], label='Training')
        # Plot validation metrics
        if val_metric in history.history:
            ax.plot(history.history[val_metric], color=colors[1], label='Validation')

        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)

        if metric == 'lambda':
            ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig("training_metrics.png", dpi=300, bbox_inches="tight")
    plt.show()


# function to plot losses
def plot_loss_components(history):
    epochs = range(1, len(history.history['loss']) + 1)
    plt.figure(figsize=(12, 6))
    
  # plot reconstruction loss
    plt.plot(epochs, history.history['reconstruction_loss'], 'b-', label='Reconstruction Loss', linewidth=2)

    # plot weighted KL loss
    weighted_kl = np.array(history.history['lambda']) * np.array(history.history['kl_loss'])
    plt.plot(epochs, weighted_kl, 'r-', label='Weighted KL Loss (λ × KL)', linewidth=2)

    # Plot the total loss
    plt.plot(epochs, history.history['loss'], 'g-', label='Total Loss', linewidth=2)

    plt.title('Loss Components Throughout Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig("loss_components.png", dpi=300, bbox_inches="tight")
    plt.show()

# Function for digit morphing visualization
def visualize_digit_morphing(vae, n=15, digit_size=28, figsize=(15, 15)):
    # n*n 2D manifold of digits
    figure = np.zeros((digit_size * n, digit_size * n, 1))

    grid_x = np.linspace(-3, 3, n)
    grid_y = np.linspace(-3, 3, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            # Create a point in 2D space (first 2 dimensions of latent space)
            z_sample = np.zeros((1, vae.encoder.outputs[0].shape[1]))
            z_sample[:, 0] = xi
            z_sample[:, 1] = yi

            # Decode this point to get a digit
            x_decoded = vae.decoder.predict(z_sample, verbose=0)
            digit = x_decoded[0].reshape(digit_size, digit_size, 1)

            # Place the digit in the overall figure
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit

    plt.figure(figsize=figsize)
    # Reshape for display
    figure = figure.reshape((n * digit_size, n * digit_size))
    plt.imshow(figure, cmap="Greys_r")
    plt.axis("off")
    plt.title("Latent Space Visualization - Digit Morphing")
    plt.tight_layout()
    plt.savefig("digit_morphing.png", dpi=300, bbox_inches="tight")
    plt.show()

# function for digit morphing animation
def digit_morphing_animation(vae, start_digit=5, end_digit=8, steps=10, digit_size=28):
    # load mnist data
    (x_train, y_train), _ = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)

    # get examples of start and end digits
    start_example = x_train[np.where(y_train == start_digit)[0][0:1]]
    end_example = x_train[np.where(y_train == end_digit)[0][0:1]]

    # encoding examples to get their latent representations
    z_start = vae.encoder.predict(start_example, verbose=0)[0]
    z_end = vae.encoder.predict(end_example, verbose=0)[0]

    # creating interpolations in latent space
    alpha_values = np.linspace(0, 1, steps)

    plt.figure(figsize=(12, 2))

    for i, alpha in enumerate(alpha_values):
        # Linear interpolation in latent space
        z_interp = z_start * (1 - alpha) + z_end * alpha
        # Reshape for decoder input
        z_interp = np.reshape(z_interp, (1, z_interp.shape[0]))
        # Decode the interpolated point
        x_decoded = vae.decoder.predict(z_interp, verbose=0)
        # Plot the decoded digit
        plt.subplot(1, steps, i + 1)
        plt.imshow(x_decoded[0].reshape(digit_size, digit_size), cmap='Greys_r')
        plt.axis('off')

    plt.suptitle(f"Morphing from {start_digit} to {end_digit}")
    plt.tight_layout()
    plt.savefig(f"morphing_{start_digit}_to_{end_digit}.png", dpi=300, bbox_inches="tight")
    plt.show()

# Function for t-SNE visualization of latent space
def visualize_latent_space_tsne(vae, n_samples=5000):
    # loading MNIST data
    (x_train, y_train), _ = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)

    indices = np.random.choice(x_train.shape[0], n_samples, replace=False)
    x_sample = x_train[indices]
    y_sample = y_train[indices]

    # encoding samples to get latent space rep.
    z_mean, _, _ = vae.encoder.predict(x_sample, verbose=0)

    # t-SNE
    print("Applying t-SNE dimensionality reduction")
    tsne = TSNE(n_components=2, random_state=0, perplexity=30, learning_rate=200)
    z_tsne = tsne.fit_transform(z_mean)

    # Plotting t-SNE vis
    plt.figure(figsize=(12, 10))
    colors = cm.rainbow(np.linspace(0, 1, 10))

    for i in range(10):
        indices = y_sample == i
        plt.scatter(z_tsne[indices, 0], z_tsne[indices, 1],
                   c=[colors[i]], label=f'Digit {i}',
                   alpha=0.7, edgecolors='w', s=50)

    plt.title("t-SNE Visualization of Latent Space", fontsize=14)
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig("tsne_visualization.png", dpi=300, bbox_inches="tight")
    plt.show()

'''
# function for 3D t-SNE visualization
def plot_3d_tsne(vae, n_samples=2000):
    # loading MNIST data
    (x_train, y_train), _ = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)

    # Take a subset of samples
    indices = np.random.choice(x_train.shape[0], n_samples, replace=False)
    x_sample = x_train[indices]
    y_sample = y_train[indices]

    # Encode samples to get latent space representation
    z_mean, _, _ = vae.encoder.predict(x_sample, verbose=0)

    # Apply t-SNE for dimensionality reduction to 3D
    print("Applying 3D t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=3, random_state=0, perplexity=30, learning_rate=200)
    z_tsne = tsne.fit_transform(z_mean)

    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Define colormap
    colors = cm.rainbow(np.linspace(0, 1, 10))

    # Plot each digit class with a different color
    for i in range(10):
        indices = y_sample == i
        ax.scatter(z_tsne[indices, 0], z_tsne[indices, 1], z_tsne[indices, 2],
                  c=[colors[i]], label=f'Digit {i}',
                  alpha=0.7, edgecolors='w', s=50)

    ax.set_title("3D t-SNE Visualization of Latent Space", fontsize=14)
    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax.set_zlabel("t-SNE Dimension 3", fontsize=12)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("tsne_3d_visualization.png", dpi=300, bbox_inches="tight")
    plt.show()
'''


def main():
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

if __name__ == "__main__":
    main()
run_all_visualizations(vae, history)

# Function to calculate metrics
def compute_metrics(originals, reconstructions):
    mse_values, psnr_values, ssim_values = [], [], []

    for i in range(len(originals)):
        original = originals[i]
        reconstructed = reconstructions[i]

        # Convert to 0-255 range for metrics
        original_uint8 = (original * 255).astype(np.uint8)
        reconstructed_uint8 = (reconstructed * 255).astype(np.uint8)

        # MSE Calculation (Lower is better, perfect = 0)
        mse = np.mean((original - reconstructed) ** 2)
        mse_values.append(mse)

        # PSNR Calculation (Higher is better, perfect = inf)
        psnr = 10 * np.log10(1 / mse) if mse != 0 else 100
        psnr_values.append(psnr)

        # SSIM Calculation (Higher is better, perfect = 1)
        #ssim_value = ssim(original_uint8, reconstructed_uint8, data_range=255, multichannel=True)
        #ssim_values.append(ssim_value)

    # averaging scores over all images
    avg_mse = np.mean(mse_values)
    avg_psnr = np.mean(psnr_values)
    #avg_ssim = np.mean(ssim_values)

    print(f"\nMetrics Summary:")
    print(f"MSE (Lower is better, perfect=0): {avg_mse:.4f}")
    print(f"PSNR (Higher is better, perfect=∞): {avg_psnr:.2f} dB")
    #print(f"SSIM (Higher is better, perfect=1): {avg_ssim:.4f}")
    # return avg_mse, avg_psnr, avg_ssim
    return avg_mse, avg_psnr

# test reconstructions
num_samples = 100  # number of test images to evaluate
x_sample = x_test[:num_samples]
reconstructions = vae.predict(x_sample)

# Compute and print all metrics
# mse, psnr, ssim_score = compute_metrics(x_sample, reconstructions)
mse, psnr = compute_metrics(x_sample, reconstructions)




########################################
########################################
####       FINE-TUNING SCRIPT       ####
########################################
########################################

def fine_tune_on_emnist(pretrained_vae, epochs=50, batch_size=64):
    
    emnist_variant = 'letters'  # using the letters variant of EMNIST

    # loading EMNIST dataset
    emnist_ds = tfds.load(f'emnist/{emnist_variant}', split=['train', 'test'], as_supervised=True)
    emnist_train, emnist_test = emnist_ds

    # preprocessing data
    def preprocess(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        # EMNIST images need to be transposed since they're stored differently from MNIST
        image = tf.transpose(image, perm=[1, 0, 2])
        return image, image

    train_ds = (
        emnist_train
        .map(preprocess)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    test_ds = (
        emnist_test
        .map(preprocess)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_images = np.array([img for img, _ in test_ds.take(10).as_numpy_iterator()])
    val_images = np.vstack(val_images)

    ###  Callbacks  ###
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_val_total_loss",
        patience=8,
        restore_best_weights=True,
        mode='min'
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_val_total_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-9,
        verbose=1,
        mode='min'
    )

    pretrained_vae.optimizer.learning_rate.assign(1e-5)

    # Fine-tune the model
    history = pretrained_vae.fit(
        train_ds,
        epochs=epochs,
        validation_data=test_ds,
        callbacks=[early_stopping, reduce_lr]
    )

    return pretrained_vae, history

fine_tuned_vae, emnist_history = fine_tune_on_emnist(vae, epochs=500, batch_size=64)






###########################################
###########################################
#### VISUALISATION FOR EMNIST AFTER FT ####
###########################################
###########################################

def visualize_reconstructions(vae, data, n=10):
    random_indices = np.random.choice(len(data), n, replace=False)
    samples = data[random_indices]
  
    # reconstructions
    z_mean, z_log_var, z = vae.encoder(samples)
    reconstructions = vae.decoder(z)

    # Plot original vs reconstructed imgs
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # original imgs
        plt.subplot(2, n, i + 1)
        plt.imshow(samples[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.title('Original')

        # reconstructed imgs
        plt.subplot(2, n, i + n + 1)
        plt.imshow(reconstructions[i].numpy().reshape(28, 28), cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.title('Reconstructed')

    plt.tight_layout()
    plt.show()


def visualize_latent_space_emnist(vae, data, labels, n=5000, method='tsne'):
    subset_idx = np.random.choice(len(data), min(n, len(data)), replace=False)
    subset_data = data[subset_idx]
    subset_labels = labels[subset_idx]

    z_mean, _, _ = vae.encoder(subset_data)
    z_mean = z_mean.numpy()

    # dim reduction method
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=0)
    elif method == 'umap':
        reducer = umap.UMAP(n_components=2, random_state=0)
    else:
        raise ValueError("Method must be either 'tsne' or 'umap'")

    z_proj = reducer.fit_transform(z_mean)

    label_chars = np.array([chr(label + 64) for label in subset_labels])  # 1 → 'A'

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(z_proj[:, 0], z_proj[:, 1], c=subset_labels, cmap='tab20', alpha=0.6, s=10)
    plt.title(f'{method.upper()} visualization of EMNIST latent space', fontsize=16)
    plt.xlabel(f'{method.upper()} Dimension 1')
    plt.ylabel(f'{method.upper()} Dimension 2')

    # create legend
    classes = np.unique(subset_labels)
    handles = [plt.Line2D([], [], marker='o', linestyle='', color=scatter.cmap(scatter.norm(c)), label=chr(c + 64)) for c in classes]
    plt.legend(handles=handles, title="Letters", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def generate_samples(vae, n_samples=10, latent_dim=16):
    z_sample = np.random.normal(0, 1, size=(n_samples, latent_dim))
    generated_images = vae.decoder(z_sample).numpy()
    plt.figure(figsize=(12, 3))
    for i in range(n_samples):
        plt.subplot(1, n_samples, i+1)
        plt.imshow(generated_images[i].reshape(28, 28), cmap='gray')
        plt.axis('off')

    plt.suptitle('Generated Samples')
    plt.tight_layout()
    plt.show()

def interpolate_digits(vae, data, start_idx, end_idx, steps=10):
    start_img = np.expand_dims(data[start_idx], axis=0)
    end_img = np.expand_dims(data[end_idx], axis=0)

    start_z_mean, _, _ = vae.encoder(start_img)
    end_z_mean, _, _ = vae.encoder(end_img)

    # Create interpolation steps
    alphas = np.linspace(0, 1, steps)
    interpolated_images = []

    for alpha in alphas:
        interpolated_z = (1-alpha) * start_z_mean + alpha * end_z_mean
        interpolated_img = vae.decoder(interpolated_z).numpy()
        interpolated_images.append(interpolated_img[0])

    # Plot
    plt.figure(figsize=(15, 3))
    for i, img in enumerate(interpolated_images):
        plt.subplot(1, steps, i+1)
        plt.imshow(img.reshape(28, 28), cmap='gray')
        plt.axis('off')

    plt.suptitle(f'Interpolation from image {start_idx} to {end_idx}')
    plt.tight_layout()
    plt.show()


def evaluate_reconstruction_quality(vae, data):
    z_mean, z_log_var, z = vae.encoder(data)
    reconstructions = vae.decoder(z)
    
    # MSE
    mse = tf.reduce_mean(tf.keras.losses.mse(data, reconstructions))
    # KL divergence
    kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))

    print(f"Reconstruction MSE: {mse:.4f}")
    print(f"KL Divergence: {kl_loss:.4f}")

    return mse, kl_loss




########################################
batch_size=64

emnist_variant = 'letters' 

# load the EMNIST dataset
emnist_ds = tfds.load(f'emnist/{emnist_variant}', split=['train', 'test'], as_supervised=True)
emnist_train, emnist_test = emnist_ds

# preprocessing EMNIST data
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    # EMNIST images need to be transposed since they're stored differently from MNIST
    image = tf.transpose(image, perm=[1, 0, 2])
    return image, image  

train_ds = (
        emnist_train
        .map(preprocess)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
)

test_ds = (
        emnist_test
        .map(preprocess)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
)

# convert test dataset to a numpy array
test_images = []
test_labels = []

for img, label in emnist_test.take(10000): 
    test_images.append(tf.transpose(img, perm=[1, 0, 2]).numpy().astype(np.float32) / 255.0)
    test_labels.append(label.numpy())

test_images = np.array(test_images)
test_labels = np.array(test_labels)



visualize_reconstructions(fine_tuned_vae, test_images, n=10)
visualize_latent_space_emnist(fine_tuned_vae, test_images, test_labels, n=7000, method='tsne')  # For t-SNE
#visualize_latent_space_emnist(fine_tuned_vae, test_images, test_labels, n=7000, method='umap')  # For UMAP

mse, kl = evaluate_reconstruction_quality(fine_tuned_vae, test_images)
generate_samples(fine_tuned_vae, n_samples=10, latent_dim=LATENT_DIMS)
interpolate_digits(fine_tuned_vae, test_images, start_idx=10, end_idx=20, steps=10)


#from google.colab import files
#vae.save("/content/adaptive_ResVAE.h5")
