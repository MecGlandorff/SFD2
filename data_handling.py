# data_handling.py

import pandas as pd
import logging

class DataHandler:
    def __init__(self, config):
        """
        Initialize the DataHandler with configuration settings.
        
        param config: Dictionary containing configuration parameters, this can be found & changed in config.yaml"
        """
        self.file_paths = config['data_paths']
        self.data = {}
        self.merged_data = None
        self.logger = logging.getLogger('DataHandler')

    def load_data(self):
        """
        Load data from file paths
        """
        try:
            self.logger.info('loading data...')
            for name, path in self.file_paths.items():
                self.data[name] = pd.read_csv(path)
                self.logger.info(f'{name} loaded successfully from {path}')
        except Exception as e:
            self.logger.error(f'Error loading data: {e}')
            raise

    def preprocess_data(self):
        """
        Preprocess the loaded data & handle missing values. REMINDER for self!!! watch out with this, since this might in the future cause issues 
        because missing values are handled incorrectly!
        """
        try:
            # Loggin statement
            self.logger.info('preprocessing data...')
            # Handle missing values
            for df_name, df in self.data.items():
                self.data[df_name] = df.dropna()
                
                # Loggin statement
                self.logger.info(f'{df_name}: missing values handled')
            
            # Below add the other preprocessing steps as merging etc.

            # Merge merge merggee
            self.merge_data()

        except Exception as e:
            self.logger.error(f'error during preprocessing: {e}')
            raise

    # Merge data method
    def merge_data(self):
        """ Merges datasets pandas"""

        try:
            # Loggin' statement 
            self.logger.info("merging data")

            # Merge the segments. Later look into how and which for now just generalized method
            df_segmented_data = self.data['train_segmented']
            df_meta_train = self.data['meta_train_clean']

            # Loggin' statement
            self.logger.info("merging data :)")
            merged_df = pd.merge(df_segmented_data, df_meta_train, on=['StudyInstanceUID', 'Slice'], how='inner', 
            suffixes=("_segmented", "meta"))

            self.logger.info("merge completed")

        # To check accuracy merge meta_segmentation_clean in here as well. Is optional though, RF*S and in later code maybe delete?
            if 'meta_segmentation_clean' in self.data:
                df_meta_segmentation = self.data['meta_segmentation_clean']

                # Loggin statement
                self.logger.info("merging with meta_segmentation_clean for True labels")

                merged_df = pd.merge(merged_df,df_meta_segmentation, on=['StudyInstanceUID', "Slice"], how='left',
                suffixes=('','_gt')) # _gt will be later defined as "ground truth" so to compare to prediction. 
                #Columns frrom the right DF (merged_df) will have no change in name, but left (df_meta_segmentation) will have
                # _gt added to it. 
                

                # Loggin statement
                self.logger.info('True labels merged')
            
            self.merged_data = merged_df
    
        except Exception as e:
            self.logger.error(f"error during merging data: {e}")
            raise

    def get_data(self):
        """
        Returns the preppropd data =*)
        """
        return self.data
