import matplotlib.pyplot as plt
import pandas as pd
from skimage.feature import SIFT
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import cv2
import os


def extract_patches(img, patch_size, patches, target_size):
    resized_img = cv2.resize(img, target_size)
    gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.equalizeHist(gray_img)
    height, width = gray_img.shape
    for y in range(0, height - patch_size + 1, patch_size):
        for x in range(0, width - patch_size + 1, patch_size):
            patch = gray_img[y:y + patch_size, x:x + patch_size]
            patches.append(patch)


def load_images_with_shapes(dataset_path):
    images, shapes, indices = [], [], []
    for img_name in os.listdir(dataset_path):
        img_path = os.path.join(dataset_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        shapes.append(img.shape)
        indices.append(img_name)
        images.append(img)
    return images, shapes, indices


def extract_features(patches):
    descriptors_list = []
    for patch in patches:
        try:
            sift = SIFT()
            sift.detect_and_extract(patch)
            _, descriptors = sift.keypoints, sift.descriptors
        except:
            continue
        if descriptors is not None:
            descriptors_list.append(descriptors)
    return descriptors_list


def build_visual_dictionary(descriptors_list, n_clusters):
    all_descriptors = np.vstack(descriptors_list)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(all_descriptors)
    return kmeans


def build_bow_histogram(descriptors, kmeans):
    histogram = np.zeros(len(kmeans.cluster_centers_))
    clusters = kmeans.predict(descriptors)
    for cluster_idx in clusters:
        histogram[cluster_idx] += 1
    return histogram


def build_bow_matrix_by_images(images, patch_size, kmeans, target_size):
    bow_matrix = []
    index_to_histogram = {}
    for idx, image in enumerate(images):
        patches = []
        extract_patches(image, patch_size, patches, target_size)

        descriptors_list = extract_features(patches)
        if descriptors_list:
            descriptors = np.vstack(descriptors_list)
            histogram = build_bow_histogram(descriptors, kmeans)
        else:
            histogram = np.zeros(len(kmeans.cluster_centers_))
        bow_matrix.append(histogram)
        index_to_histogram[idx] = histogram

    return np.array(bow_matrix), index_to_histogram


def convert_to_tfidf(bow_matrix):
    tfidf_transformer = TfidfTransformer()
    tfidf_matrix = tfidf_transformer.fit_transform(bow_matrix)
    return tfidf_matrix


def plot_bow_histogram(histogram, image_name):
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(histogram)), histogram, color='blue')
    plt.xlabel('Centroids (visual words)')
    plt.ylabel('Frequency of occurrence')
    plt.title(f'BoVW histogram for the image {image_name}')
    plt.show()


def compute_query_bow(query_img, patch_size, kmeans):
    query_patches = []
    extract_patches(query_img, patch_size, query_patches, target_size=(query_img.shape[1], query_img.shape[0]))
    descriptors = extract_features(query_patches)
    if descriptors:
        query_descriptors = np.vstack(descriptors)
        query_histogram = build_bow_histogram(query_descriptors, kmeans)
        return query_histogram
    return None


def find_closest_images(query_histogram, bow_matrix, top_n=3):
    distances = np.array([np.linalg.norm(query_histogram - bow) for bow in bow_matrix])
    closest_indices = np.argsort(distances)[:top_n]
    closest_distances = distances[closest_indices]
    return [(index, dist) for index, dist in zip(closest_indices, closest_distances)]


if __name__ == '__main__':
    datapath, query_path = 'landscapes', 'query_object.jpg'
    patch_size, target_size = 128, (1024, 1024)
    n_clusters = 10

    # Step 1: Uploading images and their sizes
    images, shapes, indices = load_images_with_shapes(datapath)
    for idx, (name, shape) in enumerate(zip(indices, shapes)):
        print(f"Index: {idx}, File: {name}, Shape: {shape}")
    print()

    # Step 2: Slicing images into patches
    patches = []
    for image in images:
        extract_patches(image, patch_size, patches, target_size)
    print(f'Number of patches: {len(patches)}\n')

    # Step 3: Extracting SIFT descriptors
    descriptors_list = extract_features(patches)
    print(f'Total descriptors extracted: {len(descriptors_list)}')
    print(f"Example descriptor (first patch): {descriptors_list[0][0]}")
    print(f"Size of descriptor: {len(descriptors_list[0][0])}\n")

    # Step 4: Creating visual words
    kmeans = build_visual_dictionary(descriptors_list, n_clusters)
    print(f'Visual words (clusters): {len(kmeans.cluster_centers_)}\n')

    # Step 5: Construction of BoVW histograms for each image
    bow_matrix, index_to_histogram = build_bow_matrix_by_images(images, patch_size, kmeans, target_size)
    print(f'BoVW Matrix Shape: {bow_matrix.shape}\n')
    bow_df = pd.DataFrame(bow_matrix, columns=[f'Word{i}' for i in range(bow_matrix.shape[1])])
    bow_df.insert(0, 'FileName', indices)
    print("\nBoW Table:")
    print(bow_df.to_string(index=False))

    # Step 6: Converting BoVW to TF-IDF
    tfidf_matrix = convert_to_tfidf(bow_matrix)
    print(f'TF-IDF Matrix Shape: {tfidf_matrix.shape}\n')
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f'Word{i + 1}' for i in range(bow_matrix.shape[1])])
    tfidf_df.insert(0, 'FileName', indices)
    print("\nTF-IDF Table:")
    print(f'{tfidf_df.to_string(index=False)}\n')

    # Step 7: Histogram visualisation for all images
    for idx, histogram in enumerate(bow_matrix):
        plot_bow_histogram(histogram, indices[idx])

    # Step 8: Uploading and processing of the query image
    query_img = cv2.imread(query_path)
    query_histogram = compute_query_bow(query_img, patch_size, kmeans)
    print(f"Query Image Shape: {query_img.shape}")

    print("\nBoW Histogram for Query Image:")
    print(query_histogram)
    query_tfidf = TfidfTransformer().fit_transform([query_histogram]).toarray()[0]
    print("\nTF-IDF Vector for Query Image:")
    print(query_tfidf)

    plt.imshow(query_img, cmap='gray')
    plt.title("Query Image")
    plt.show()

    # Step 9: Search for closest images
    closest_images = find_closest_images(query_histogram, bow_matrix, top_n=3)
    print("\nClosest Images to Query:")
    for index, distance in closest_images:
        print(f"Index: {index}, Distance: {distance}")
        plt.imshow(images[index])
        plt.title(f"Closest Image {index}, Distance: {distance}")
        plt.show()
