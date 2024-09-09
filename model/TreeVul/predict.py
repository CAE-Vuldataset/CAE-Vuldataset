import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

import re
from allennlp.training.metrics import metric
from allennlp.data import Instance, Token, Vocabulary, allennlp_collate
import json
from typing import Dict, Any
import sys

import json
import logging
from typing import Any, Dict

from allennlp.commands.subcommand import Subcommand
from allennlp.common import logging as common_logging
from allennlp.common.logging import prepare_global_logging
from allennlp.common.util import prepare_environment, import_module_and_submodules
from allennlp.data import DataLoader
from allennlp.data.batch import Batch
from allennlp.models.archival import load_archive
from allennlp.training.util import evaluate
from allennlp.nn import util

from cal_metrics import cal_metrics_cwe_tree, cal_metrics_cwe

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)


def _test(archive_file,
         input_file,
         test_config=None, 
         weights_file=None, 
         output_file=None, 
         predictions_output_file=None, 
         batch_size=64,
         cuda_device=0,
         seed=2022,
         package="Baseline",
         prepare_first=False,
         cwe_tree_path="valid_cwe_tree.json",
         batch_weight_key="",
         file_friendly_logging=False) -> Dict[str, Any]:
    
    common_logging.FILE_FRIENDLY_LOGGING = file_friendly_logging

    logging.getLogger("allennlp.common.params").disabled = True
    logging.getLogger("allennlp.nn.initializers").disabled = True
    logging.getLogger("allennlp.modules.token_embedders.embedding").setLevel(logging.INFO)

    import_module_and_submodules(package)
    overrides = test_config or ""

    archive = load_archive(
        archive_file,
        weights_file=weights_file,
        cuda_device=cuda_device,
        overrides=overrides,
    )
    config = archive.config
    prepare_environment(config)
    model = archive.model

    model.eval()

    if prepare_first:
        dataset_reader = archive.validation_dataset_reader  # validation reader defined in test_config_treevul.json is used for loading the CWE description data
        instances = list(dataset_reader.read(cwe_tree_path))
        vocab = Vocabulary()
        dataset = Batch(instances)
        dataset.index_instances(vocab)
        model_input = util.move_to_device(dataset.as_tensor_dict(), cuda_device)
        model.forward_cwe_description(**model_input)

    # Load the evaluation data
    dataset_reader = archive.dataset_reader

    evaluation_data_path = input_file
    logger.info("Reading evaluation data from %s", evaluation_data_path)

    data_loader_params = config.pop("validation_data_loader", None)
    if data_loader_params is None:
        data_loader_params = config.pop("data_loader")
    if batch_size:
        data_loader_params["batch_size"] = batch_size
    data_loader = DataLoader.from_params(
        params=data_loader_params, reader=dataset_reader, data_path=evaluation_data_path
    )

    data_loader.index_with(model.vocab)

    metrics = evaluate(
        model,
        data_loader,
        cuda_device,
        batch_weight_key,
        output_file=output_file,
        predictions_output_file=predictions_output_file,
    )

    logger.info("Finished evaluating.")

    return metrics


def test(test_set, package, model, weights=None, cuda=0, seed=2022):
    # wrapping api for test

    if "baseline" in model or "notree" in model:
        # neural models that directly map an input patch to its CWE category
        # CodeBERT, Bi-LSTM and TreeVul-h (ablation_notree)
        depth = 2
        config = json.load(open("test_config.json", 'r'))
        config["model"]["depth"] = depth
        config["dataset_reader"]["depth"] = depth
        prepare_first = False
        batch_size = 8
    else:
        # neural models with hierarchical and chained architecture (i.e., incorporate the knowledge of CWE tree structure)
        # use our proposed tree structure aware and beam searched based inference algorithm
        # TreeVul and TreeVul-t (ablation_noedit)
        max_depth = 3
        beam_size = 5  # 5
        top_k = 1  # also support top-k
        config = json.load(open("test_config_tree.json", 'r'))
        config["model"]["max_depth"] = max_depth
        config["dataset_reader"]["max_depth"] = max_depth
        config["model"]["top_k"] = top_k
        config["model"]["beam_size"] = beam_size
        config["model"]["train_label_embedding"] = False
        batch_size = 1  # for beam search
        prepare_first = True

    # 创建对应路径下的test_results文件夹
    if not os.path.exists(f"{config['path']}/test_results/"):
        os.makedirs(f"{config['path']}/test_results/")

    config["trainer"]["cuda_device"] = cuda
    config["model"]["device"] = f"cuda:{cuda}"

    cwe_tree_path = f"{config['path']}/valid_cwe_tree.json"

    output_file = f"{config['path']}/test_results/{model}_metric.json"
    predictions_output_file = f"{config['path']}/test_results/{model}_result.json"
    
    _test(archive_file=f"{config['path']}/{model}/model.tar.gz", input_file=f"{config['path']}/{test_set}.json", 
         test_config=config, weights_file=weights, output_file=output_file, predictions_output_file=predictions_output_file, 
         batch_size=batch_size, cuda_device=cuda, seed=seed, package=package,
         prepare_first=prepare_first, cwe_tree_path=cwe_tree_path)


# DATA_PATH = "xxx"
# MODEL_PATH = "xxx"

if __name__ == "__main__":
    test_set = "test_set"
    package = "TreeVul"
    # package = "Baseline"
    model = "out_treevul"
    weights = None  # use weights in the archived file (model output)
    seed = 2022
    cuda = 0

    # also need to set the CUDA environment (already set at the very beginning of this script)
    # CUBLAS_WORKSPACE_CONFIG=:16:8

    # test for a single run
    test(test_set=test_set, package=package, model=model, weights=weights, cuda=cuda, seed=seed)