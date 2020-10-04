import pandas as pd
from simpletransformers.classification import ClassificationModel
import torch
from sklearn.metrics import accuracy_score
import wandb
import logging
import json


def clean_df(data: pd.DataFrame) -> pd.DataFrame:
    data = data[['document', 'label']]
    data['document'] = data['document'].map(lambda doc: str(doc))
    data['label'] = data['label'].map(lambda label: int(label))
    return data


def read_json(file_loc: str = "/electra/config.json") -> dict:
    fp = open(file_loc, mode='r', encoding='utf-8')
    json_data = json.load(fp=fp)
    fp.close()
    return json_data


# Read data (with parameter file_loc)
config = read_json()

# File pointer initialize
train, test = None, None

# Read and clean DataFrame
if config.get("train_data"):
    train = pd.read_csv(config["train_data"], sep='\t')
    train = clean_df(train)
if config.get("test_data"):
    test = pd.read_csv(config["test_data"], sep='\t')
    test = clean_df(test)

# Check cuda is available
cuda_available = torch.cuda.is_available()

config["cuda_available"] = cuda_available

if __name__ == "__main__":

    # Weight and Biases init setting
    # If you want to set run name, then edit config
    if config.get("wandb_project"):
        wandb.init(project=config["wandb_project"])

    # Just console logger
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.INFO)

    # Classification Model
    # If you wanted, change config and use whatever you want, even BERT model
    model = ClassificationModel(model_type=config["model_type"],
                                model_name=config["model_name"],
                                num_labels=config["num_labels"],
                                use_cuda=cuda_available,
                                args=config)

    # Check dataset is not None
    if train is not None:
        if test is None:
            # When test dataset is None, only training will be proceed
            model.train_model(train_df=train, acc=accuracy_score)
        else:
            # Evaluate with test dataset
            model.train_model(train_df=train, eval_df=test, acc=accuracy_score)
    else:
        # Without training, only evaluate
        result, model_outputs, wrong_predictions = model.eval_model(eval_df=test, acc=accuracy_score)
