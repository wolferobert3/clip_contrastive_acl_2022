import tensorflow as tf
import numpy as np
from os import path, listdir
from sklearn.decomposition import PCA
import plotly.express as px
import pickle
from nltk import pos_tag, word_tokenize
import re
import random
import copy

#Math Functions
def cosine_similarity(a, b):
    return ((np.dot(a, b)) / (np.sqrt(np.dot(a, a)) * np.sqrt(np.dot(b, b))))

def std_deviation(J):
    mean_J = np.mean(J)
    var_J = sum([(j - mean_J)**2 for j in J])
    return (np.sqrt(var_J / (len(J)-1)))

def create_permutation(a, b):
    permutation = random.sample(a+b, len(a+b))
    return permutation[:int(len(permutation)*.5)], permutation[int(len(permutation)*.5):]

def pearson_r(X, Y):
    mean_x = np.mean(X)
    mean_y = np.mean(Y)

    squared_difference_x = sum([(x - mean_x)**2 for x in X])
    squared_difference_y = sum([(y - mean_y)**2 for y in Y])
    
    numerator = sum([(X[i] - mean_x) * (Y[i] - mean_y) for i in range(len(X))])
    denominator = np.sqrt(squared_difference_x * squared_difference_y)

    return (numerator / denominator)

#Visualization Functions
def visualize_pca(vectors, labels, color_labels, num_components):

    if num_components == 2:

        pca = PCA(n_components = 2)
        components = pca.fit_transform(vectors, y = labels)
        variance = pca.explained_variance_ratio_.sum() * 100

        figure = px.scatter(
            components, x = 0, y = 1, color = color_labels,
            title = f'Total Explained Variance: {variance:.2f}%',
            hover_name = labels,
            labels = {'0': 'PC 1', '1': 'PC 2'}
            )

        figure.show()
        return

    if num_components == 3:

        pca = PCA(n_components = 3)
        components = pca.fit_transform(vectors, y = labels)
        variance = pca.explained_variance_ratio_.sum() * 100

        figure = px.scatter_3d(
            components, x = 0, y = 1, z = 2, color = color_labels,
            title = f'Total Explained Variance: {variance:.2f}%',
            hover_name = labels,
            labels = {'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
            )

        figure.show()
        return

    else:

        pca = PCA(n_components = num_components)
        components = pca.fit_transform(vectors, y = labels)

        total_var = pca.explained_variance_ratio_.sum() * 100

        axis_labels = {str(i): f"PC {i+1}" for i in range(num_components)}

        fig = px.scatter_matrix(
            components,
            color = color_labels,
            dimensions = range(num_components),
            labels = axis_labels,
            hover_name = labels,
            title=f'Total Explained Variance: {total_var:.2f}%',
        )
        
        fig.update_traces(diagonal_visible=False)
        fig.show()
        return

def pca_transform(embedding_array, pcs, subtract_mean = True):

    pca = PCA(n_components = pcs)

    if subtract_mean:
        common_mean = np.mean(embedding_array, axis=0)
        transformed_array = embedding_array - common_mean
    else:
        transformed_array = np.array(embedding_array, copy = True)

    if pcs == 0:
        return transformed_array

    pca.fit_transform(transformed_array)
    pc_top = pca.components_
    
    pc_remove = np.matmul(np.matmul(transformed_array, pc_top.T), pc_top)
    transformed_embeddings = transformed_array - pc_remove

    return transformed_embeddings, pc_top