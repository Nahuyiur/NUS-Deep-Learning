import matplotlib.pyplot as plt


def sample_batch(dataset):
    batch = dataset.take(1).get_single_element()
    if isinstance(batch, tuple):
        batch = batch[0]
    return batch.numpy()


def display_grid(train_imgs, recon_train_imgs, test_imgs, recon_test_imgs, gen_imgs, num_imgs = 10, block=False):
                 

    plt.figure(figsize=(10, 8))

    for i in range(num_imgs):
        plt.subplot(5, num_imgs, i+1)
        plt.imshow(train_imgs[i], cmap='gray')
        plt.title('Train')
        plt.axis('off')

        plt.subplot(5, num_imgs, i + num_imgs + 1)
        plt.imshow(recon_train_imgs[i], cmap='gray')
        plt.title('R.Train')
        plt.axis('off')

        plt.subplot(5, num_imgs, i + 2 * num_imgs + 1)
        plt.imshow(test_imgs[i], cmap='gray')
        plt.title('Test')
        plt.axis('off')

        plt.subplot(5, num_imgs, i + 3 * num_imgs + 1)
        plt.imshow(recon_test_imgs[i], cmap='gray')
        plt.title('R.Test')
        plt.axis('off')

        plt.subplot(5, num_imgs, i + 4 * num_imgs + 1)
        plt.imshow(gen_imgs[i], cmap='gray')
        plt.title('Gen')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show(block = block)

def display(
    images, n=10, size=(20, 3), cmap="gray_r", as_type="float32", save_to=None
):
    """
    Displays n random images from each one of the supplied arrays.
    """
    if images.max() > 1.0:
        images = images / 255.0
    elif images.min() < 0.0:
        images = (images + 1.0) / 2.0

    plt.figure(figsize=size)
    for i in range(n):
        _ = plt.subplot(1, n, i + 1)
        plt.imshow(images[i].astype(as_type), cmap=cmap)
        plt.axis("off")

    if save_to:
        plt.savefig(save_to)
        print(f"\nSaved to {save_to}")

    plt.show()
