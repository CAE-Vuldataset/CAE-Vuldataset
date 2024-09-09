from multiprocessing.sharedctypes import Value
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.callbacks.callback import TrainerCallback
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.data_loaders.data_loader import DataLoader
from allennlp.data.tokenizers.pretrained_transformer_tokenizer import PretrainedTransformerTokenizer
from allennlp.data.token_indexers.pretrained_transformer_indexer import PretrainedTransformerIndexer
from allennlp.data.batch import Batch
from allennlp.data.data_loaders.data_loader import TensorDict
from allennlp.nn import util
from typing import Dict, Any, Sequence, List, Optional

from overrides import overrides
import logging
import json
import numpy as np
from .reader_cwe import ReaderCWE

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@TrainerCallback.register("prepare_cwe_tree")
class PrepareCWEData(TrainerCallback):

    def __init__(self,
                 reader: DatasetReader = None,
                 cwe_tree_path: str = "cwe_tree_valid.json",
                 serialization_dir: str = None) -> None:
        super().__init__(serialization_dir)
        PTM = "microsoft/codebert-base"
        reader = reader or ReaderCWE(tokenizer = PretrainedTransformerTokenizer(PTM, add_special_tokens=True, max_length=64),
                                    token_indexers = {"tokens": PretrainedTransformerIndexer(PTM, namespace="tags")})
        self._instances = list(reader.read(cwe_tree_path))
        logger.info(f"num of cwe nodes {len(self._instances)}")
        
    @overrides
    def on_start(self, trainer: "GradientDescentTrainer", is_primary: bool, **kwargs) -> None:
        self.trainer = trainer
        
        model = trainer.model  # trainer._pytorch_model
        # model.train()

        logger.info("preparing cwe nodes before training")
        vocab = Vocabulary()
        cuda_device = model._get_prediction_device()
        dataset = Batch(self._instances)
        dataset.index_instances(vocab)
        model_input = util.move_to_device(dataset.as_tensor_dict(), cuda_device)
        model.forward_cwe_description(**model_input)


@TrainerCallback.register("custom_validation")
class CustomValidation(TrainerCallback):
    def __init__(self,
                 serialization_dir: str = None) -> None:
        super().__init__(serialization_dir)
        pass
        
    @overrides
    def on_epoch(self, trainer: "GradientDescentTrainer", metrics: Dict[str, Any], epoch: int, is_primary: bool, **kwargs) -> None:
        # prepare the label embeddings before the validation, since unlike the training the lable embeddings won't change during the validation
        model = trainer.model
        model.eval()

        logger.info("preparing the CWE embeddings for validation")
        # update the label embedding using the current model before validation
        model.forward_cwe_description()  # the logic is in model.forward_cwe_description


@TrainerCallback.register("scheduled_sampling")
class ScheduledSampling(TrainerCallback):
    # adjust the ratio of teacher forcing learning during the training (scheduled sampling)
    def __init__(self, decay_strategy: str = "linear", start_learning_ratio: float=1.0, end_learning_ratio: float = 0, start_learning_epochs: int = 0, end_learning_epochs: int = 50, k=13, serialization_dir: str = None) -> None:
        super().__init__(serialization_dir)

        if decay_strategy not in ["linear", "inverse_sigmoid"]:
            raise ValueError("decay_strategy must be one of the (linear, inverse_sigmoid)")

        self._decay_strategy = decay_strategy  # choose one strategy for ratio decay
        self._start_learning_ratio = start_learning_ratio  # maximum learning ratio
        self._end_learning_ratio = end_learning_ratio  # minimum learning ratio
        self._start_learning_epochs = start_learning_epochs  # keep start_learning_ratio in the epochs < start_learning_epochs
        self._end_learning_epochs = end_learning_epochs  # keep end_learning_ratio in epochs > end_learning_epochs

        self._k = k  # for inverse sigmoid decay

        # for linear decay
        if decay_strategy == "linear":
            self._k = (start_learning_ratio - end_learning_ratio) / (start_learning_epochs - end_learning_epochs)
    
    def linear_decay(self, i):
        # linear decay for schedule sampling
        # i start from 1
        return self._start_learning_epochs - self._k * (i - self._start_learning_epochs)

    def inverse_sigmoid_decay(self, i):
        # inverse sigmoid decay strategy for schedule sampling
        return self._k / (self._k + np.exp(i / self._k))
    
    @overrides
    def on_start(self, trainer: "GradientDescentTrainer", is_primary: bool = True, **kwargs) -> None:
        self.trainer = trainer

        # set the initial learning ratio
        model = trainer.model
        model._teacher_forcing_ratio = self._start_learning_ratio

    @overrides
    def on_epoch(self, trainer: "GradientDescentTrainer", metrics: Dict[str, Any], epoch: int, is_primary: bool, **kwargs) -> None:
        # called at the last epoch, used to set the learning ratio for next epoch
        # the epoch start from 1
        model = trainer.model
        epoch += 2  # next epoch & epoch start from 0

        if epoch <= self._start_learning_epochs:
            model._teacher_forcing_ratio = self._start_learning_ratio
        elif epoch > self._end_learning_epochs:
            model._teacher_forcing_ratio = self._end_learning_ratio
        else:
            # epoch in (start_learning_epochs, end_learning_epochs]
            decay_method = getattr(self, f"{self._decay_strategy}_decay")
            model._teacher_forcing_ratio = decay_method(epoch)
            
        logger.info(f"change teacher forcing ratio into {model._teacher_forcing_ratio}")
    
    @overrides
    def on_batch(self, trainer: "GradientDescentTrainer", batch_inputs: List[TensorDict], batch_outputs: List[Dict[str, Any]], batch_metrics: Dict[str, Any], epoch: int, batch_number: int, is_training: bool, is_primary: bool = True, batch_grad_norm: Optional[float] = None, **kwargs) -> None:
        pass