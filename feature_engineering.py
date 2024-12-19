



import numpy as np
import logging
from skimage import exposure
from scipy.ndimage import gaussian_laplace
import pywt 
from concurrent.futures import ThreadPoolExecutor


class FeatureEngineer:
    def __init(self, config=None):
        self.logger = logger.getLogger("feature engineer")
        self.config = config or {}
        self.wavelet = self.config.get('wavelet', 'db1')
        self.wavelet_level = self.config.get('wavelet_level', 1)
        # Sigma for log filter
        self.log_sigma = self.config.get('log_sigma', 1.0) 
        # Clahe filter
        self.clahe_clip_limit = self.config.get('clahe_clip_limit', 0.01)

    def cahe(self, volume):
        """
        Applies CLAHE to each slice in the volume
        volume shape: (Z, Y, X)
        """
        self.logger.info("applying cahe to each slice...")
        processed_slices = []
        for i in range (volume.shape[0]):
            # selects the i-th slice and :, : together make sure that all the x and y is loaded
            slice_ = volume[i, :, :]
            # equalize_adapthist expects values in [0,1], so ensure normalization
            slice_norm = (slice_ - slice_.min()) / (slice_.max() - slice_.min() + 1e-8)
            clahe_slice = exposure.equalize_adapthist(slice_norm, clip_limit=self.clahe_clip_limit)
            # Optionally rescale back to original intensity range
            clahe_slice = clahe_slice * (slice_.max() - slice_.min()) + slice_.min()
            processed_slices.append(clahe_slice)
        return np.stack(processed_slices, axis=0)

    def apply_log(self, volume):
        """
        Applies Laplacian of Gaussian filter to each slice
        LoG highlights edges and can help with bone fracture features 
        """
        self.logger.info("applying laplacian of gaussian filter...")
        # You can apply directly to the volume (3D), but let's do it slice-wise.
        processed_slices = []
        for i in range(volume.shape[0]):
            slice_ = volume[i, :, :]
            log_slice = gaussian_laplace(slice_, sigma=self.log_sigma)
            processed_slices.append(log_slice)
        return np.stack(processed_slices, axis=0)

    def apply_wavelet(self, volume):
        """
        applies D. Wavelet Transform to each slice in a 3D volume

        Parameters:
            volume (numpy.ndarray): A 3D numpy array of shape (Z, Y, X), where Z is the number of slices

        Returns:
            numpy.ndarray: A 3D numpy array with wavelet-transformed slices
        """
        self.logger.info("applying wavelet transform...")
        
        # Input validation: ensure the input is a 3D array
        if len(volume.shape) != 3:
            raise ValueError("input volume must be a 3D array with shape (Z, Y, X)")

        processed_slices = []

        for i in range(volume.shape[0]):
            slice_ = volume[i, :, :]
            
            # 2D wavelet decomposition
            coeffs = pywt.wavedec2(slice_, wavelet=self.wavelet, level=self.wavelet_level)
            
            # Reconstruct the slice
            reconstructed = pywt.waverec2(coeffs, wavelet=self.wavelet)
            
            # Ensure reconstructed slice has the same dimensions as the original to prevent handling errors
            reconstructed = reconstructed[:slice_.shape[0], :slice_.shape[1]]
            
            # ?Optional/worse? normalization back to original range
            min_val, max_val = slice_.min(), slice_.max()
            reconstructed = np.clip(reconstructed, min_val, max_val)
            
            processed_slices.append(reconstructed)

        # Stack proc. slices back into 3D volume
        return np.stack(processed_slices, axis=0)



    def process_volume(self, volume, training=True):
        """
        Runs the full preprocessing/feature engineering pipeline using:
        - CLAHE
        - LoG
        - Wavelet
        """
        self.logger.info("processing volume with clahe, log and wavelet transforms...")

        # 1. clahe to enhance contrast 
        volume = self.apply_clahe(volume)

        # 2. log filter to highlight edges
        volume = self.apply_log(volume)

        # 3. wavelet transform to get multi-scale features
        volume = self.apply_wavelet(volume)

        # Normalize data
        volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)

        # Further augmentations??
        return volume