# Training notebook runbook

## Usage

Here are instructions for training a model using the Colab notebook script.

### Repo clone config
At the **Clone repo** section, put in GitHub user and access token in indicated locations.

### Modify hyperparameters for training
In the first few lines after **Train transformer**, there are hyperparameters that can be changed for training, including epochs and batch size.

### Authorize google account
When the first cell is run, follow the prompt for authorization.

## Features

The Colab script does the following:
- Copy dataset from gcp bucket to Colab environment 
- Clone repo
- Train model
- Save checkpoints on gcp bucket while training
- Evaluate every checkpoint after training
- Update desired google sheet with evaluation results
