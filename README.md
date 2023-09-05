# transformer-deid
Fine tune transformer models to deidentify clinical medical data. 

## Setup
Install dependencies in a conda environment:
```
conda env create -n transformer_deid --file environment.yml
```

## Data
Data must be in CSV stand-off format: a subfolder (txt/) contains the documents in individual text files with the document identifier as the file stem and `.txt` as the extension. Another subfolder (ann/) contains a set of CSV files with the annotations with the same document identifier as the file stem and `.gs` as the extension. The tests/data subfolder contains an example of documents stored in this format.

## Training

Models supported:
- BERT
- DistilBERT
- RoBERTa

To run from the repository directory, 
```
python transformer_deid/train.py -m <model_architecture> -i <dataset path> -o <output path> -e <number of epochs>
```

Options:
* `-m --model_architecture   Name of model {bert | distilbert | roberta}.`
* `-i --train_path           Path to dataset directory.`
* `-o --output_path          Model save directory.`
* `-e --epochs               Number of epochs.`

## Evaluation

For evaluation, see [Pyclipse](https://github.com/kind-lab/pyclipse).
