



import pandas as pd
import json
import os
import logging

class SequenceGenerator:
    def __init__(self):
        self.logger = logging.getLogger('SequenceGenerator')
        self.vertebrae = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']


    def generate_sequences(self, df):
         ###RF*S How do we handle slides with multiple vertebrae in it?!
        """ Generates sequences of slides per vertebrae 
        df input: dataframe with detected vertebrae
        returns dictionary containing sequences for each vertebrae"""

        try:
            self.logger.info('generating sequences...')
            sequences = {v: {} for v in self.vertebrae}
            for v in vertebrae:
                vertebrae_df = df[df['VertebraeDetected'].apply(lambda x: v in x)]
                grouped = vertebrae_df.groupby("StudyInstanceUID")
                for study_uid, group in grouped:
                    sequences = group.sort_values("slice")["slice"].tolist()
                    sequences[v][study_uid]

                self.logger.info("sequences generated sucessfuly")
                return sequences
            
        except Exception as e:
            self.logger.error(f"error during sequence generation: {e}")


    def save_sequences(self, sequences, output_dir):
        """ Save sequences to a JSON file """

        try: 
            self.logger.info('saving sequences...')
            os.makedirs(output_dir, exist_ok=True) 
            output_path = os.path.join(output_dir, "sequences.json")

            with open(output_path, "w") as f:
                json.dump(sequences, f, indent =4)

            # loggin statement
            self.logger.info(f"sequences saved to {output_path}")

        except Exception as e:
            self.logger.error(f"error saving sequences: {e}")
            raise

    
    def save_detection_results(self, df, output_dir):
        try:
            self.logger.info("saving detection results...")

            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "detection_results.csv")
            df.to_csv(output_path, index=False)

            self.logger.info(f"detection results saved to {output_path}")

        except Exception as e:
            self.logger.error(f"error saving detection results: {e}")
            raise


    
