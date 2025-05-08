import pandas as pd
from mftools.preprocess.binarise import niblack
from mftools.preprocess.normalise import normalise_individual
import numpy as np
import scipy.ndimage as ndi
from skimage import io, img_as_ubyte, morphology, feature
from skimage.segmentation import watershed
import glob
import os
from progress.bar import IncrementalBar

import matplotlib
import matplotlib.pyplot as plt
import scienceplots
matplotlib.rcParams.update({'font.size': 14})
plt.rcParams['figure.dpi'] = 150
plt.style.use(['science', 'ieee'])


root_dir = './data/'
out_dir = './out/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
if not os.path.exists(f'{out_dir}segmentations/'):
    os.makedirs(f'{out_dir}segmentations/')
if not os.path.exists(f'{out_dir}directions/'):
    os.makedirs(f'{out_dir}directions/')
filenames = sorted(glob.glob(f'{root_dir}*.tif'))
n_images = len(filenames)
df_directions = pd.DataFrame(columns=['filename', 'grain_id', 'grain_size', 'direction'])

with IncrementalBar('Processing', max=len(filenames), suffix='%(percent).1f%% - %(eta)ds') as bar:
    for counter, filename in enumerate(filenames):
        filename = os.path.basename(filename)
        image = io.imread(f'{root_dir}{filename}')

        # Binarise
        image = 255 - image
        image_whitebalance = normalise_individual(image, 20)
        image_binary = niblack(image, window_size=29)
        image_binary = img_as_ubyte(image_binary)
        image_binary_closed = morphology.area_closing(image_binary, area_threshold=1000)
        image_binary_closed = morphology.area_opening(image_binary_closed, area_threshold=50)

        # Erode alpha grains
        n_erosions = 9
        image_eroded = np.copy(image_binary_closed)
        for n in range(n_erosions):
            image_eroded = morphology.erosion(image_eroded)
        image_eroded_labelled, _ = ndi.label(image_eroded)

        distance = ndi.distance_transform_edt(image_eroded)

        # Watershed
        coords = feature.peak_local_max(distance, labels=image_eroded_labelled, num_peaks_per_label=1)
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndi.label(mask)
        labels = watershed(-distance, markers, mask=image_binary_closed, watershed_line=True)
        labels = morphology.area_closing(labels, area_threshold=200)

        n_grains = np.max(labels)
        directions_patch = np.zeros(n_grains)
        for n in np.arange(1, n_grains, 1):

            # Isolate grain and skip if small
            grain = labels == n
            if np.sum(grain) < 2000:
                continue

            # Eigenvectors
            y, x = np.nonzero(grain)
            x = x - np.mean(x)
            y = y - np.mean(y)
            coords = np.vstack([x, y])
            cov = np.cov(coords)
            try:
                evals, evecs = np.linalg.eig(cov)
            except:
                continue
            sort_indices = np.argsort(evals)[::-1]
            x_v1, y_v1 = evecs[:, sort_indices[0]]
            x_v2, y_v2 = evecs[:, sort_indices[1]]

            # Find angle between principle axis and bottom edge of image
            theta = np.degrees(np.arctan(-x_v1 / y_v1))
            if theta == -0.0:
                theta = 0.0
            directions_patch[n] = theta
            df_directions = df_directions._append({'filename': filename, 'grain_id': n, 'grain_size': np.sum(grain), 'direction': theta}, ignore_index=True)

        bar.next()

print(df_directions)
df_directions.to_csv(f'{out_dir}directions.csv')
