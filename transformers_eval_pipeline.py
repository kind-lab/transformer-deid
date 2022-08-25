import argparse
import logging
from transformer_deid import model_evaluation_functions as eval

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
_LOGGER = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        'Evaluate multiple transformer-based deid models over multiple datasets'
    )
    parser.add_argument(
        '-m',
        '--model_list',
        nargs='*',
        type=str,
        help=
        'list of model file locations;\neach should contain config.json, pytorch_model.bin, and training_args.bin'
    )
    parser.add_argument(
        '-d',
        '--data_dir',
        type=str,
        help=
        'folder containing test and training data for dataset used for training'
    )
    parser.add_argument(
        '-t',
        '--test_data',
        nargs='*',
        type=str,
        help=
        'list of directories of test data; each should contain txt and ann folders'
    )
    parser.add_argument(
        '-c',
        '--metric',
        type=str,
        default='overall_f1',
        help='one of the keys of results_binary or None; default to overall_f1'
    )

    args = parser.parse_args()

    return args


def main():
    # TO DO: should I make the inputs a folder of models and a folder of test data?
    #        how should I return the information?

    args = parse_args()

    model_list = args.model_list
    dataDir = args.data_dir
    test_data_list = args.test_data
    metric = args.metric

    results = eval.eval_model_list(
        model_list, dataDir, test_data_list, output_metric=metric
    )

    return results


if __name__ == '__main__':
    main()
