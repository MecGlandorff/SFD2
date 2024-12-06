

import pandas as pd
import logging

class VertebraeDetector:
    def __init__(self, threshold=0.5):
        """
        Init vertebraedector with prob treshold. This threshold is the prob of the vertebrae Cx that should be 
        surpassed for the slide to be assigned to vertebrae Cx
        """

        self.threshold = threshold
        self.logger = logging.getLogger('VertebraeDetector')
        self.vertebrae = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
        

    def detect_vertebrae(self, df):
        """
        Detect the vertebrae in each slice of the DataFrame
        
        param: DataFrame containing vertebrae probabilities.
        returns: DataFrame with an additional column 'VertebraeDetected'"""

        try:
            self.logger.info('Detecting vertebrae in slices of CT-scan...')
            df['VertebraeDetected'] = df.apply(self._detect_in_row, axis=1)
            self.logger.info('Vertebrae detection completed!')
            return df
        
        except Exception as e:
            self.logger.error(f'Error during vertebrae detection: {e}')
            raise


    def _detect_in_row(self, row):
        """
        Detect vertebrae in a single row based on the threshold.
        
        param row: Series representing a row in the DataFrame.
        return List of the detected vertebrae (example "C1" in the row)
        Take notice, it can return more than 1 vertebrae, as a slice is not always perfectly outside of edges and! vertebraes overlap!
        """

        detected = [v for v in self.vertebrae if row[v] >= self.threshold]
        return detected