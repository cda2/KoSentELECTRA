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


def read_json(file_loc: str = "/base-dir/config.json") -> dict:
    fp = open(file_loc, mode='r', encoding='utf-8')
    json_data = json.load(fp=fp)
    fp.close()
    return json_data


config = read_json()

train = pd.read_csv(config["train_data"], sep='\t')
test = pd.read_csv(config["test_data"], sep='\t')

train = clean_df(train)
test = clean_df(test)

cuda_available = torch.cuda.is_available()

config["cuda_available"] = cuda_available

if __name__ == "__main__":
    if config.get("wandb_project"):
        wandb.init(project=config["wandb_project"])

    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.INFO)

    model = ClassificationModel(model_type=config["model_type"],
                                model_name=config["model_name"],
                                num_labels=config["num_labels"],
                                use_cuda=cuda_available,
                                args=config)

    model.train_model(train_df=train, eval_df=test, acc=accuracy_score)

    result, model_outputs, wrong_predictions = model.eval_model(eval_df=test, acc=accuracy_score)
