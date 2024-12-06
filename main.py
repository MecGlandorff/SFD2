



# NS*F = note for self
# N*W = not working
# RS*F = reminder for self


# Main
import logging
import yaml
import argparse
from seq_output import SequenceGenerator
from data_handling import DataHandler
from vertebrae_detector import VertebraeDetector

def setup_logging(log_level=logging.INFO):
    # NF*S https://docs.python.org/3/library/logging.html
    """
    Logging function. You can determine here how much logging info you want
    """  
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler('slice_detector.log', mode='w'),
            logging.StreamHandler()
                    ])
    
def load_config(config_path):
    """ Load the configuration settings from a YAML file"""
   
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    # Parser
    parser = argparse.ArgumentParser(description='SliceDetector Program')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file')
    args = parser.parse_args()
    
    # logging
    setup_logging()
    logger = logging.getLogger('Main')

    try:
        # Load config
        config = load_config(args.config)
        logger.info('Configuration loaded.')

        # modules initalizer
        data_handler = DataHandler(config)
        vertebrae_detector = VertebraeDetector(threshold=config['threshold'])
        sequence_generator = SequenceGenerator()

        # Data loading and preprocessing
        data_handler.load_data()
        data_handler.preprocess_data()
        data = data_handler.get_data()

        # Vertebrae detection
        detection_results = vertebrae_detector.detect_vertebrae(data['train_segmented'])

        # Sequence gen. and output
        sequences = sequence_generator.generate_sequences(detection_results)
        output_dir = config['output_dir']
        sequence_generator.save_sequences(sequences, output_dir)
        sequence_generator.save_detection_results(detection_results, output_dir)

        logger.info('SliceDetector program completed successfully.')

    except Exception as e:
        logger.error(f'An error occurred: {e}')
        raise

if __name__ == '__main__':
    main()