# File work tools
import os
import glob

# General data work tools
import warnings
import pandas as pd
import numpy as np
import random
import itertools

# Image preprocessing tools
import cv2
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from skimage.feature import hog
from sklearn.decomposition import PCA
from skimage.feature import graycomatrix, graycoprops


# This function is used to receive dataset paths, store wanted images in training and testing paths, and the size of the images to be resized to.
# The function returns the preprocessed data to be used in an ML model of our choice.
def preprocess_data(dataset_path, n, ratio, size, pca_comps_thresh, div_func='minmax'):
    """
    Preprocesses the data to be used in the model.
    :param dataset_path: The path to the dataset.
    :param n: The number of images to sample from each class in the dataset.
    :param ratio: The ratio of the sampled data to be used for training.
    :param size: The row/column size to which the images will be resized.
    :param pca_comps_thresh: The threshold for the number of components to reduce the dataset to using PCA.
    :param div_func: The division function to be used to normalize or standardize the images.
    :return: The preprocessed data.
    """

    if ratio < 0 or ratio > 1:
        raise ValueError("The ratio must be between 0 and 1.")

    if ratio < 0.5:
        warnings.warn("The ratio is less than 0.5. Not recommended for training the model.")

    mult_div_func, uno_div_func = None, None

    if div_func not in ['minmax', 'standard']:
        raise ValueError("The division function must be either minmax or standard.")
    elif div_func == 'minmax':
        mult_div_func = minmax_normalize_multiple
        uno_div_func = minmax_normalize
    else:
        mult_div_func = standardize_multiple
        uno_div_func = standardize

    images, labels = sample_and_resize(dataset_path, n, size)  # Sample the images from the dataset path to the training path and testing path

    normalized_images = mult_div_func(images)  # Normalize the data to be between 0 and 1 using the min-max normalization
    label, le = label_data(labels)             # Convert the labels to numbers for the model to be able to process them
    images_features = feature_extraction(images, normalized_images, pca_comps_thresh, uno_div_func)  # Extract features from the images and align them in a dataframe

    x_train, x_test, y_train, y_test = train_test_split(images_features, label, test_size=1-ratio, random_state=42)  # Split the data into training and testing datasets

    return x_train, x_test, y_train, y_test, le


# This method receives an array of images paths, and returns a list of resized images a list of corresponding labels.
def sample_and_resize(data_path, n, size=128):
    images = []
    labels = []

    for artist_path in glob.glob(data_path + r'/*'):
        label = artist_path.split("\\")[-1]  # Extract the artist name from the directory path.

        artist_images = glob.glob(os.path.join(artist_path, "*.png"))  # List of images
        random.shuffle(artist_images)

        # Adjust the number of images based on availability
        n = min(n, len(artist_images))
        artist_images = artist_images[:n]

        for img_path in artist_images:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # Load image with all channels (RGBA)
            if img is None or img.shape[-1] != 4:  # Ensure it's RGBA
                continue
            img = cv2.resize(img, (size, size))  # Resize
            images.append(img)
            labels.append(label)

    return np.array(images), np.array(labels)



def label_data(labels):
    """
    Converts the labels to numbers for the model to be able to process them, utilizing sklearn LabelEncoder.
    :param labels: The labels of the images in the dataset.
    :return: The encoded labels for the dataset.
    """
    # Converting the labels to numbers for the model to be able to process them.
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    labels_encoded = le.transform(labels)

    # Giving our data conventional names for easier use in the model.
    return labels_encoded, le


def standardize_multiple(images):
    """
    Standardize the data.
    :param images: The training dataset.
    :return: Normalized training and testing datasets.
    """
    images_standardized = []
    for img in images:

        # If the image is colored, standardize each channel separately.
        if len(img.shape) == 3:
            mean, std = cv2.meanStdDev(img)
            mean = np.asarray(mean).reshape(-1)
            std = np.asarray(std).reshape(-1)
            std_img = (img.astype(np.float32) - mean) / std
            images_standardized.append(std_img)

        # If the image is grayscale, standardize it directly.
        else:
            mean, std = cv2.meanStdDev(img)
            std_img = (img.astype(np.float32) - mean) / std
            images_standardized.append(std_img)

    return np.array(images_standardized)


def standardize(image):
    """
    Standardize the data.
    :param image: The training dataset.
    :return: Normalized training and testing datasets.
    """
    # If the image is colored, standardize each channel separately.
    if len(image.shape) == 3:
        mean, std = cv2.meanStdDev(image)
        mean = np.asarray(mean).reshape(-1)
        std = np.asarray(std).reshape(-1)
        return (image.astype(np.float32) - mean) / std

    # If the image is grayscale, standardize it directly.
    else:
        mean, std = cv2.meanStdDev(image)
        return (image.astype(np.float32) - mean) / std


def minmax_normalize_multiple(images):
    """
    Normalizes the data to be between 0 and 1 using the min-max normalization.
    :param images: The training dataset.
    :return: Normalized training and testing datasets.
    """
    images_normalized = []
    for img in images:
        min_val = np.min(img)
        max_val = np.max(img)
        if max_val == min_val:
            normalized_img = np.full_like(img, 0.5)
        else:
            normalized_img = (img.astype(np.float32) - min_val) / (max_val - min_val)
        images_normalized.append(normalized_img)

    return np.array(images_normalized)


def minmax_normalize(image):
    """
    Normalizes the data to be between 0 and 1 using the min-max normalization. Works for both grayscale and colored images.
    :param image: The training dataset.
    :return: Normalized training and testing datasets.
    """
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val == min_val:
        return np.full_like(image, 0.5)
    else:
        return (image.astype(np.float32) - min_val) / (max_val - min_val)


def vector_images(dataset):
    """
    Vectorizes the images in the dataset into 1D arrays.
    :param dataset: The dataset of images to be vectorized.
    :return: The vectorized images.
    """
    vectorized_images = []
    for img in dataset:
        vectorized_img = img.reshape(-1)
        vectorized_images.append(vectorized_img)
    return np.array(vectorized_images)


def rgba_to_gray(img):
    if img.shape[-1] == 4:  # Ensure RGBA
        b, g, r, a = cv2.split(img)
        gray = cv2.cvtColor(cv2.merge([b, g, r]), cv2.COLOR_BGR2GRAY)
        return gray
    elif img.shape[-1] == 3:  # Standard RGB
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img  # Grayscale image


# The following function is used to vectorize the images by extracting features from them, and aligning them in a dataframe.
# The input must be a 4 dimensional array. In our case, an array of colored images. Won't work with grayscale images.
def feature_extraction(dataset, norm_dataset, pca_components_threshold, div_func):
    """
    Extracts features from the images and aligns them in a dataframe.
    :param dataset: the dataset of images to be filtered.
    :param norm_dataset: the normalized dataset of images to be filtered.
    :param pca_components_threshold: the threshold for the number of components to reduce the dataset to using PCA.
    :param div_func: the division function to be used to normalize or standardize the images.
    :return: the dataframe of the extracted features.
    """
    arrays_to_combine = []  # A list to store the reduced images

    '''
    # FEATURE 1 - Original Images Pixel Values
    original_images = vector_images(norm_dataset)  # Reduce the images to the given number of components using PCA
    reduced_original_images = dataset_pca_reduction(original_images, norm_dataset, pca_components_threshold)  # Reduce the images to the given number of components using PCA
    arrays_to_combine.append(reduced_original_images)  # A list to store the reduced images
    '''

    # FEATURE 2 - Hog Features :

    # The following parameters are used to create the HOG filters:
    hog_hyperparameters = {
        'orientations': [6, 9, 12],            # Number of orientation bins for the gradient histogram
        'pixels_per_cell': [(4, 4), (6, 6)],   # Size of a cell
        'cells_per_block': [(2, 2)],           # Number of cells in each block
        'block_norm': ['L2-Hys']                   # Block normalization method
    }

    hog_images_dict = hog_images(hog_hyperparameters, norm_dataset, div_func)  # Apply the HOG filter to the images in the dataset

    for label, features in hog_images_dict.items():
        pca = PCA(n_components=pca_components_threshold)
        reduced_features = pca.fit_transform(features)
        arrays_to_combine.append(reduced_features)

    # FEATURE 3 - Haralick Features :

    hl_images = haralick_images(dataset, div_func)  # Apply the Haralick filter to the images in the dataset
    arrays_to_combine.append(hl_images)       # Add the reduced Haralick images to the list of reduced images

    '''
    # FEATURE 4 - Sobel Features

    ks = [3, 5, 7]   # Represents the kernel size of the filter (K x K)

    sobel_images_dict = sobel_images(ks, norm_dataset, div_func)          # Apply the Sobel filter to the images in the dataset

    reduced_sobel_images_dict = {}
    for label, images_pixels in sobel_images_dict.items():
        reduced_sobel_images_dict[label] = dataset_pca_reduction(images_pixels, norm_dataset, pca_components_threshold)

    for label, images_pixels in reduced_sobel_images_dict.items():
        arrays_to_combine.append(images_pixels)


    # FEATURE 5 - Gabor Features
    
    # The following parameters are used to create the Gabor and Sobel filters.
    f  = [0.1, 0.5]     # Represents the frequency of the sine component
    o  = [0, 45, 90, 135]    # Represents the orientation of the filter
    sa = [1.0]               # Represents the spatial aspect ratio of the filter.
    sd = [0.5, 1.0]          # Represents the standard deviation of the filter
    p  = [0]                 # Represents the phase offset of the filter
    ks = [3, 7]              # Represents the kernel size of the filter (K x K)
    
    filters = create_gabor_filters(f, o, sa, sd, p, ks)  # Create the Gabor filters based on the parameters above
    
    gabor_images_dict = gabor_images(filters, norm_dataset, div_func)     # Apply the Gabor filters to the images in the dataset

    reduced_gabor_images_dict = {}
    for label, images_pixels in gabor_images_dict.items():
        reduced_gabor_images_dict[label] = dataset_pca_reduction(images_pixels, norm_dataset, pca_components_threshold)

    for label, features in reduced_gabor_images_dict.items():
        arrays_to_combine.append(features)
    '''

    # Combination of all selected features:
    combined_array = np.hstack(arrays_to_combine)
    combined_df = pd.DataFrame(combined_array)

    return combined_df


def haralick_images(dataset, div_func):
    """
    Applies the Haralick filter to the images in the dataset.
    :param dataset: The dataset of images to be filtered.
    :param div_func: The division function to be used to normalize or standardize the images.
    :return: The filtered images.
    """

    distances = [1, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    haralick_imgs = []

    for img in dataset:
        gray_img = rgba_to_gray(img)
        glcms = graycomatrix(gray_img, distances, angles, 256, symmetric=True, normed=True)

        contrast      = graycoprops(glcms, prop='contrast')       # Contrast
        dissimilarity = graycoprops(glcms, prop='dissimilarity')  # Dissimilarity
        homogeneity   = graycoprops(glcms, prop='homogeneity')    # Homogeneity
        energy        = graycoprops(glcms, prop='energy')         # Energy
        correlation   = graycoprops(glcms, prop='correlation')    # Correlation

        # Normalize each feature
        features = [div_func(features) for features in [contrast, dissimilarity, homogeneity, energy, correlation]]
        haralick_image = np.array(features).flatten()
        haralick_imgs.append(haralick_image)

    return np.array(haralick_imgs)


def hog_images(hyperparameters, dataset, div_func):
    """
    Applies the HOG filter to the images in the dataset.
    :param hyperparameters: The hyperparameters of the HOG filter.
    :param dataset: The dataset of images to be filtered.
    :param div_func: The division function to be used to normalize or standardize the images.
    :return: The filtered images.
    """
    hog_images_dict = {}  # A dictionary to store the filtered images
    count = 1  # A counter to label the HOG images for each hyperparameter

    for orientation, pixels_per_cell, cells_per_block, block_norm in itertools.product(hyperparameters['orientations'], hyperparameters['pixels_per_cell'], hyperparameters['cells_per_block'], hyperparameters['block_norm']):

        hog_label = 'HOG_' + str(count)
        hog_curr_images = []

        for img in dataset:
            gray = rgba_to_gray(img)
            hog_image = hog(gray, orientations=orientation, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, block_norm=block_norm)
            hog_image = div_func(hog_image)
            hog_image = hog_image.reshape(-1)
            hog_curr_images.append(hog_image)

        hog_images_dict[hog_label] = np.array(hog_curr_images)
        count += 1

    return hog_images_dict


def create_gabor_filters(freq, orient, aspect, std_dev, phase_offset, kernel_size):
    """
    Creates a Gabor filter based on the given parameters.
    :param freq: The frequency of the sine component.
    :param orient: The orientation of the filter.
    :param aspect: The spatial aspect ratio of the filter.
    :param std_dev: The standard deviation of the filter.
    :param phase_offset: The phase offset of the filter.
    :param kernel_size: The kernel size of the filter (K x K).
    :return: A Gabor filters list over all possible combinations of the given parameters.
    """
    combos = list(itertools.product(freq, orient, aspect, std_dev, phase_offset, kernel_size))  # All possible combinations of the filter parameters
    filters = []

    for freq, orient, aspect, std_dev, phase_offset, kernel_size in combos:
        gabor_filter = cv2.getGaborKernel((kernel_size, kernel_size), std_dev, orient, freq, aspect, phase_offset, ktype=cv2.CV_32F)
        filters.append(gabor_filter)

    return filters


# This function takes a dataset of images and a list of Gabor filters, and applies the filters to the images.
# The result is a list of images filters by each individual filter.
def gabor_images(filter_list, dataset, div_func):
    """
    Applies the Gabor filters to the images in the dataset.
    :param filter_list: The list of Gabor filters to be applied to the images.
    :param dataset: The dataset of images to be filtered.
    :param div_func: The division function to be used to normalize or standardize the images.
    :return: A list of images filtered by each individual filter.
    """
    gabor_images_dict = {}
    count = 1

    # This loop applies the Gabor filters to the images in the dataset and adds the filtered images to the list of filtered images.
    for filt in filter_list:

        curr_gabor_images = []

        gabor_label = 'Gabor' + str(count)
        for image in range(dataset.shape[0]):

            input_image = dataset[image, :, :, :]  # Get the image from the dataset
            img = input_image                      # Copy the image to a new variable

            gabor_image = cv2.filter2D(img, -1, filt)
            gabor_image = div_func(gabor_image)
            curr_gabor_images.append(gabor_image)

        gabor_images_dict[gabor_label] = np.array(curr_gabor_images)
        count += 1

    return gabor_images_dict


def sobel_images(kernel_sizes, dataset, div_func):
    """
    Applies the Sobel filter to the images in the dataset.
    :param kernel_sizes: The kernel size of the Sobel filter.
    :param dataset: The dataset of images to be filtered.
    :param div_func: The division function to be used to normalize or standardize the images.
    :return: The filtered images.
    """
    sobel_images_dict = {}   # A dictionary to store the filtered images
    count = 1                # A counter to label the Sobel images for each kernel size

    for kernel_size in kernel_sizes:
        sobels = []
        sobel_label = 'Sobel_' + str(count)

        for img in dataset:
            # Convert to grayscale if the image has multiple channels
            if len(img.shape) == 3 and img.shape[2] == 3:  # RGB or BGR
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            elif len(img.shape) == 3 and img.shape[2] == 4:  # RGBA
                gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            elif len(img.shape) == 2:  # Grayscale image
                gray = img
            else:
                raise ValueError(f"Unexpected image format: {img.shape}")

            # Apply Gaussian blur
            blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

            # Sobel edge detection
            edge_sobel_x = cv2.Sobel(blur, cv2.CV_32F, 1, 0, kernel_size)
            edge_sobel_y = cv2.Sobel(blur, cv2.CV_32F, 0, 1, kernel_size)
            edge_sobel = cv2.magnitude(edge_sobel_x, edge_sobel_y)

            # If the image was in color, merge the channels back (i.e., RGB/BGR)
            if len(img.shape) == 3 and img.shape[2] == 3:
                edge_sobel = cv2.merge([edge_sobel, edge_sobel, edge_sobel])
            elif len(img.shape) == 3 and img.shape[2] == 4:
                edge_sobel = cv2.merge([edge_sobel, edge_sobel, edge_sobel, edge_sobel])

            # Apply the division function
            edge_sobel = div_func(edge_sobel)

            # Flatten the image
            edge_sobel = edge_sobel.reshape(-1)
            sobels.append(edge_sobel)

        # Store the sobel images for the current kernel size
        sobel_images_dict[sobel_label] = np.array(sobels)
        count += 1

    return sobel_images_dict


# This function receives a flat image dataset, the original dataset of the images, and the pca threshold to reduce the dataset.
# It returns the reduced dataset.
def dataset_pca_reduction(dataset, original_dataset, threshold):
    """
    Reduces the dataset to the given number of components using PCA.
    :param dataset: The dataset to be reduced, expected to be a 4D array representing 3D images.
    :param original_dataset: The original shape of the images in the dataset.
    :param threshold: The number of components to reduce the dataset to.
    :return: The reduced dataset.
    """
    # Ensure original_dataset has the correct shape
    if len(original_dataset.shape) == 2:  # Already flattened dataset
        raise ValueError("original_dataset should not be flattened. Ensure it's in (num_images, height, width[, channels]) format.")

    if len(original_dataset.shape) == 3:  # Grayscale images
        original_dataset = original_dataset[:, :, :, np.newaxis]  # Add a singleton channel dimension

    # Unpack dimensions
    num_images, rows, cols, channels = original_dataset.shape

    # Reshape the dataset to its original 4D shape if needed
    if dataset.ndim == 2:  # Flattened dataset
        dataset = dataset.reshape(num_images, rows, cols, channels)

    # Split channels
    dataset_b, dataset_g, dataset_r = [], [], []
    for image in original_dataset:
        if len(image.shape) == 2:  # Grayscale image
            reshaped_image = image.reshape(-1)
            dataset_b.append(reshaped_image)  # Treat grayscale as single-channel
        elif len(image.shape) == 3 and image.shape[2] == 3:  # RGB or BGR image
            b_img, g_img, r_img = cv2.split(image)
            dataset_b.append(b_img.reshape(-1))
            dataset_g.append(g_img.reshape(-1))
            dataset_r.append(r_img.reshape(-1))
        elif len(image.shape) == 3 and image.shape[2] == 4:  # RGBA image
            b_img, g_img, r_img, _ = cv2.split(image)  # Discard alpha channel
            dataset_b.append(b_img.reshape(-1))
            dataset_g.append(g_img.reshape(-1))
            dataset_r.append(r_img.reshape(-1))
        else:
            raise ValueError(f"Unexpected image format: {image.shape}")

    # PCA reduction for each channel
    max_components = 0
    for channel in [dataset_b, dataset_g, dataset_r]:
        channel = np.array(channel)
        channel = channel.reshape(num_images, -1)
        pca_channel = PCA(n_components=min(channel.shape[1], threshold))
        pca_channel.fit_transform(channel)
        n_components = np.argmax(np.cumsum(pca_channel.explained_variance_ratio_) >= threshold) + 1
        max_components = max(max_components, n_components)

    # Reduce each channel
    reduced_dataset = []
    for channel in [dataset_b, dataset_g, dataset_r]:
        pca_channel = PCA(n_components=max_components)
        reduced_channel = pca_channel.fit_transform(channel)
        reduced_dataset.append(reduced_channel)

    # Merge and reshape
    reduced_dataset = np.array(reduced_dataset)
    reduced_dataset = reduced_dataset.transpose(1, 0, 2)
    reduced_dataset = reduced_dataset.reshape(num_images, -1)

    # Final PCA reduction
    pca = PCA(n_components=threshold)
    reduced_dataset = pca.fit_transform(reduced_dataset)

    return reduced_dataset