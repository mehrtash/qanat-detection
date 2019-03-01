import numpy as np
from skimage.transform import resize


def np_dice_coef(y_true, y_pred):
    if np.amax(y_pred) > 1 or np.amax(y_true) > 1:
        print('warning: values must be between 0 and 1!')
    smooth = 1.
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def correct_exposure(image):
    from skimage import exposure
    frame = np.zeros_like(image)
    frame[:50, :] = 1
    image_masked = np.multiply(image, frame)
    nz = image_masked.ravel()[np.flatnonzero(image_masked)]
    mean_white = np.mean(nz)
    image = exposure.rescale_intensity(image, in_range=(0, mean_white))
    return image


def shrink_image(image, shrink_factor):
    image_dim = len(image.shape)
    if image_dim == 3:
        image = np.lib.pad(image, ((0, 0), (1, 1), (0, 0)), mode='constant', constant_values=0)
    elif image_dim == 2:
        image = np.lib.pad(image, ((0, 0), (1, 1)), mode='constant', constant_values=0)
    return resize(image, (image.shape[0] / shrink_factor, image.shape[1] / shrink_factor))


def expand_image(image, expand_factor):
    image_dim = len(image.shape)
    image = resize(image, (image.shape[0] * expand_factor, image.shape[1] * expand_factor))
    if image_dim == 3:
        image = image[:, 1:-1, :]
    elif image_dim == 2:
        image = image[:, 1:-1]
    return image


def read_uid_list_from_file(filename):
    uids = []
    with open(filename) as f:
        for uid in f.read().splitlines():
            uids.append(uid)
    return uids


def flip_a_coin():
    return np.random.binomial(1, 0.5)


# https://www.kaggle.com/stainsby/fast-tested-rle
def run_length_encode(mask):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    inds = mask.flatten()
    runs = np.where(inds[1:] != inds[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    rle = ' '.join([str(r) for r in runs])
    return rle
