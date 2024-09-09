import copy, os
import logging
from sys import stderr

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, matthews_corrcoef
from tqdm import tqdm

from utils import debug
from modules.model import DevignModel
import json

def evaluate_loss(model, loss_function, num_batches, data_iter, cuda=False):
    model.eval()
    with torch.no_grad():
        _loss = []
        all_predictions, all_targets = [], []
        for _ in range(num_batches):
            graph, targets = data_iter()
            targets = targets.cuda()
            predictions = model(graph, cuda=True)
            batch_loss = loss_function(predictions, targets.long())
            _loss.append(batch_loss.detach().cpu().item())
            predictions = predictions.detach().cpu()
            if predictions.ndim == 2:
                all_predictions.extend(np.argmax(predictions.numpy(), axis=-1).tolist())
            else:
                all_predictions.extend(
                    predictions.ge(torch.ones(size=predictions.size()).fill_(0.5)).to(
                        dtype=torch.int32).numpy().tolist()
                )
            all_targets.extend(targets.detach().cpu().numpy().tolist())
        return np.mean(_loss).item(), accuracy_score(all_targets, all_predictions) * 100
    pass

def evaluate_metrics(model, loss_function, num_batches, data_iter):
    model.eval()
    with torch.no_grad():
        _loss = []
        all_predictions, all_targets = [], []
        for _ in range(num_batches):
            graph, targets = data_iter()
            targets = targets.cuda()
            predictions = model(graph, cuda=True)
            batch_loss = loss_function(predictions, targets.long())
            _loss.append(batch_loss.detach().cpu().item())
            predictions = predictions.detach().cpu()
            if predictions.ndim == 2:
                all_predictions.extend(np.argmax(predictions.numpy(), axis=-1).tolist())
            else:
                all_predictions.extend(
                    predictions.ge(torch.ones(size=predictions.size()).fill_(0.5)).to(
                        dtype=torch.int32).numpy().tolist()
                )
            all_targets.extend(targets.detach().cpu().numpy().tolist())
        # model.train()
        all_targets = [int(x) for x in all_targets]
        return np.mean(_loss).item(), \
               accuracy_score(all_targets, all_predictions), \
               precision_score(all_targets, all_predictions), \
               recall_score(all_targets, all_predictions), \
               f1_score(all_targets, all_predictions), \
               matthews_corrcoef(all_targets, all_predictions)

def test(model, dataset, loss_function, save_path):
    model.load_state_dict(torch.load(os.path.join(save_path, 'model.bin')))
    model.eval()

    loss, acc, pr, rc, f1, mcc = evaluate_metrics(model, loss_function, dataset.initialize_test_batch(), dataset.get_next_test_batch)

    test_res = {
        'acc': acc,
        'precision': pr,
        'recall': rc,
        'f1': f1,
        'mcc': mcc
    }
    with open(os.path.join(save_path, 'test_res.json'), 'w') as f:
        f.write(json.dumps(test_res, indent=4))
    logging.info(f"test log: \n{json.dumps(test_res, indent=4)}")

def train(model, dataset, epoches, dev_every, loss_function, optimizer, save_path, log_every=5, max_patience=5):
    logging.info('Start training!')
    train_losses = []
    best_model = None
    patience_counter = 0
    best_f1 = float('-inf')
    log_flag = 0
    max_steps = epoches * dev_every
    all_train_acc = []
    all_train_loss = []
    all_valid_acc = []
    all_valid_loss = []
    try:
        pbar = tqdm(range(max_steps), ncols=100, desc='training', mininterval=600)
        for step_count in pbar:
            model.train()

            model.zero_grad()
            graph, targets = dataset.get_next_train_batch()   #first
            targets = targets.cuda()
            predictions = model(graph, cuda=True)

            batch_loss = loss_function(predictions, targets.long())

            train_losses.append(batch_loss.detach().item())
            batch_loss.backward()
            optimizer.step()

            pbar.set_description(f"loss: {np.mean(train_losses).item()}")

            if step_count % dev_every == (dev_every - 1):
                log_flag += 1

                loss, acc, pr, rc, valid_f1, mcc = evaluate_metrics(model, loss_function, dataset.initialize_valid_batch(), dataset.get_next_valid_batch)
                
                if valid_f1 > best_f1:
                    patience_counter = 0
                    best_f1 = valid_f1
                    best_model = copy.deepcopy(model.state_dict())
                    _save_file = open(os.path.join(save_path, 'model.bin'), 'wb')
                    torch.save(model.state_dict(), _save_file)
                    _save_file.close()
                    print(f"[save model] epoch: {log_flag}, valid_f1 = {valid_f1}, best_f1 = {best_f1}")
                else:
                    patience_counter += 1
                    print(f"[patience] epoch: {log_flag}, valid_f1 = {valid_f1}, best_f1 = {best_f1}")
                train_losses = []
                if patience_counter == max_patience:
                    break
            
    except KeyboardInterrupt:
        # debug('Training Interrupted by user!')
        logging.info('Training Interrupted by user!')
    logging.info('Finish training!')

    if best_model is not None:
        model.load_state_dict(best_model)
    _save_file = open(os.path.join(save_path, 'model.bin'), 'wb')
    torch.save(model.state_dict(), _save_file)
    _save_file.close()

    loss, acc, pr, rc, f1, mcc = evaluate_metrics(model, loss_function, dataset.initialize_test_batch(),
                                       dataset.get_next_test_batch)
    test_res = {
        'epoch': log_flag,
        'acc': acc,
        'precision': pr,
        'recall': rc,
        'f1': f1,
        'mcc': mcc
    }
    with open(os.path.join(save_path, 'test_res.json'), 'w') as f:
        f.write(json.dumps(test_res, indent=4))
    logging.info(f"test log: \n{json.dumps(test_res, indent=4)}")
    


