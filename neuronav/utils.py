import numpy as np
import tarfile
import os
from urllib.request import urlretrieve


def run_episode(
    env,
    agent,
    max_steps,
    start_pos=None,
    reward_locs=None,
    random_start=False,
    update_agent=True,
):
    obs = env.reset(
        agent_pos=start_pos, reward_locs=reward_locs, random_start=random_start
    )
    agent.reset()
    steps = 0
    episode_return = 0
    done = False
    while not done and steps < max_steps:
        act = agent.sample_action(obs)
        obs_new, reward, done, _ = env.step(act)
        if update_agent:
            agent.update([obs, act, obs_new, reward, done])
        obs = obs_new
        steps += 1
        episode_return += reward
    return agent, steps, episode_return


def onehot(value, max_value):
    vec = np.zeros(max_value, dtype=np.int32)
    value = np.clip(value, 0, max_value - 1)
    vec[value] = 1
    return vec


def twohot(value, max_value):
    vec_1 = np.zeros(max_value, dtype=np.float32)
    vec_2 = np.zeros(max_value, dtype=np.float32)
    vec_1[value[0]] = 1
    vec_2[value[1]] = 1
    return np.concatenate([vec_1, vec_2])


def create_circular_mask(h, w, center=None, radius=None):
    if center is None:
        center = [int(w / 2), int(h / 2)]
    if radius is None:
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask


def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x, axis=axis)


# Taken from https://mattpetersen.github.io/load-cifar10-with-numpy
def cifar10(path=None):
    r"""Return (train_images, train_labels, test_images, test_labels).

    Args:
        path (str): Directory containing CIFAR-10. Default is
            /home/USER/data/cifar10 or C:\Users\USER\data\cifar10.
            Create if nonexistant. Download CIFAR-10 if missing.

    Returns:
        Tuple of (train_images, train_labels, test_images, test_labels), each
            a matrix. Rows are examples. Columns of images are pixel values,
            with the order (red -> blue -> green). Columns of labels are a
            onehot encoding of the correct class.
    """
    url = "https://www.cs.toronto.edu/~kriz/"
    tar = "cifar-10-binary.tar.gz"
    files = [
        "cifar-10-batches-bin/data_batch_1.bin",
        "cifar-10-batches-bin/data_batch_2.bin",
        "cifar-10-batches-bin/data_batch_3.bin",
        "cifar-10-batches-bin/data_batch_4.bin",
        "cifar-10-batches-bin/data_batch_5.bin",
        "cifar-10-batches-bin/test_batch.bin",
    ]

    if path is None:
        # Set path to /home/USER/data/mnist or C:\Users\USER\data\mnist
        path = os.path.join(os.path.expanduser("~"), "data", "cifar10")

    # Create path if it doesn't exist
    os.makedirs(path, exist_ok=True)

    # Download tarfile if missing
    if tar not in os.listdir(path):
        urlretrieve("".join((url, tar)), os.path.join(path, tar))
        print("Downloaded %s to %s" % (tar, path))

    # Load data from tarfile
    with tarfile.open(os.path.join(path, tar)) as tar_object:
        # Each file contains 10,000 color images and 10,000 labels
        fsize = 10000 * (32 * 32 * 3) + 10000

        # There are 6 files (5 train and 1 test)
        buffr = np.zeros(fsize * 6, dtype="uint8")

        # Get members of tar corresponding to data files
        # -- The tar contains README's and other extraneous stuff
        members = [file for file in tar_object if file.name in files]

        # Sort those members by name
        # -- Ensures we load train data in the proper order
        # -- Ensures that test data is the last file in the list
        members.sort(key=lambda member: member.name)

        # Extract data from members
        for i, member in enumerate(members):
            # Get member as a file object
            f = tar_object.extractfile(member)
            # Read bytes from that file object into buffr
            buffr[i * fsize : (i + 1) * fsize] = np.frombuffer(f.read(), "B")

    # Parse data from buffer
    # -- Examples are in chunks of 3,073 bytes
    # -- First byte of each chunk is the label
    # -- Next 32 * 32 * 3 = 3,072 bytes are its corresponding image

    # Labels are the first byte of every chunk
    labels = buffr[::3073]

    # Pixels are everything remaining after we delete the labels
    pixels = np.delete(buffr, np.arange(0, buffr.size, 3073))
    images = (pixels.reshape(-1, 3072).astype("float32") / 255).reshape(
        -1, 32, 32, 3, order="F"
    )

    # Split into train and test
    train_images, test_images = images[:50000], images[50000:]
    train_labels, test_labels = labels[:50000], labels[50000:]

    def _onehot(integer_labels):
        """Return matrix whose rows are onehot encodings of integers."""
        n_rows = len(integer_labels)
        n_cols = integer_labels.max() + 1
        onehot = np.zeros((n_rows, n_cols), dtype="uint8")
        onehot[np.arange(n_rows), integer_labels] = 1
        return onehot

    return train_images, _onehot(train_labels), test_images, _onehot(test_labels)
