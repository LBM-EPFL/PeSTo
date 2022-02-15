import os
import json
import numpy as np
import torch as pt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.logger import Logger
from src.scoring import bc_scoring, bc_score_names, nanmean
from config import config_data, config_model, config_runtime
from data_handler import Dataset, collate_batch_data
from src.dataset import select_by_sid, select_by_max_ba, select_by_interface_types
from model import Model


def setup_dataloader(config_data, sids_selection_filepath):
    # load selected sids
    sids_sel = np.genfromtxt(sids_selection_filepath, dtype=np.dtype('U'))

    # create dataset
    dataset = Dataset(config_data['dataset_filepath'])

    # data selection criteria
    m = select_by_sid(dataset, sids_sel) # select by sids
    m &= select_by_max_ba(dataset, config_data['max_ba'])  # select by max assembly count
    m &= (dataset.sizes[:,0] <= config_data['max_size']) # select by max size
    m &= (dataset.sizes[:,1] >= config_data['min_num_res'])  # select by min size
    m &= select_by_interface_types(dataset, config_data['l_types'], np.concatenate(config_data['r_types']))  # select by interface type

    # update dataset selection
    dataset.update_mask(m)

    # set dataset types for labels
    dataset.set_types(config_data['l_types'], config_data['r_types'])

    # define data loader
    dataloader = pt.utils.data.DataLoader(dataset, batch_size=config_runtime['batch_size'], shuffle=True, num_workers=8, collate_fn=collate_batch_data, pin_memory=True, prefetch_factor=2)

    return dataloader


def eval_step(model, device, batch_data, criterion, pos_ratios, pos_weight_factor, global_step):
    # unpack data
    X, ids_topk, q, M, y = [data.to(device) for data in batch_data]

    # run model
    z = model.forward(X, ids_topk, q, M)

    # compute weighted loss
    pos_ratios += (pt.mean(y,dim=0).detach() - pos_ratios) / (1.0 + np.sqrt(global_step))
    criterion.pos_weight = pos_weight_factor * (1.0 - pos_ratios) / (pos_ratios + 1e-6)
    dloss = criterion(z, y)

    # re-weighted losses
    loss_factors = (pos_ratios / pt.sum(pos_ratios)).reshape(1,-1)
    losses = (loss_factors * dloss) / dloss.shape[0]

    return losses, y.detach(), pt.sigmoid(z).detach()


def scoring(eval_results, device=pt.device('cpu')):
    # compute sum losses and scores for each entry
    sum_losses, scores = [], []
    for losses, y, p in eval_results:
        sum_losses.append(pt.sum(losses, dim=0))
        scores.append(bc_scoring(y, p))

    # average scores
    m_losses = pt.mean(pt.stack(sum_losses, dim=0), dim=0).numpy()
    m_scores = nanmean(pt.stack(scores, dim=0)).numpy()

    # pack scores
    scores = {'loss': float(np.sum(m_losses))}
    for i in range(m_losses.shape[0]):
        scores[f'{i}/loss'] = m_losses[i]
        for j in range(m_scores.shape[0]):
            scores[f'{i}/{bc_score_names[j]}'] = m_scores[j,i]

    return scores


def logging(logger, writer, scores, global_step, pos_ratios, step_type):
    # debug print
    pr_str = ', '.join([f"{r:.4f}" for r in pos_ratios])
    logger.print(f"{step_type}> [{global_step}] loss={scores['loss']:.4f}, pos_ratios=[{pr_str}]")

    # store statistics
    summary_stats = {k:scores[k] for k in scores if not np.isnan(scores[k])}
    summary_stats['global_step'] = int(global_step)
    summary_stats['pos_ratios'] = list(pos_ratios.cpu().numpy())
    summary_stats['step_type'] = step_type
    logger.store(**summary_stats)

    # detailed information
    for key in scores:
        writer.add_scalar(step_type+'/'+key, scores[key], global_step)

    # debug print
    for c in np.unique([key.split('/')[0] for key in scores if len(key.split('/')) == 2]):
        logger.print(f'[{c}] loss={scores[c+"/loss"]:.3f}, ' + ', '.join([f'{sn}={scores[c+"/"+sn]:.3f}' for sn in bc_score_names]))


def train(config_data, config_model, config_runtime, output_path):
    # create logger
    logger = Logger(output_path, 'train')

    # print configuration
    logger.print(">>> Configuration")
    logger.print(config_data)
    logger.print(config_runtime)

    # define device
    device = pt.device(config_runtime['device'])

    # create model
    model = Model(config_model)

    # debug print
    logger.print(">>> Model")
    logger.print(model)
    logger.print(f"> {sum([int(pt.prod(pt.tensor(p.shape))) for p in model.parameters()])} parameters")

    # reload model if configured
    model_filepath = os.path.join(output_path, 'model_ckpt.pt')
    if os.path.isfile(model_filepath) and config_runtime["reload"]:
        logger.print("Reloading model from save file")
        model.load_state_dict(pt.load(model_filepath))
        # get last global step
        global_step = json.loads([l for l in open(logger.log_lst_filepath, 'r')][-1])['global_step']
        # dynamic positive weight
        pos_ratios = pt.from_numpy(np.array(json.loads([l for l in open(logger.log_lst_filepath, 'r')][-1])['pos_ratios'])).float().to(device)
    else:
        # starting global step
        global_step = 0
        # dynamic positive weight
        pos_ratios = 0.5*pt.ones(len(config_data['r_types']), dtype=pt.float).to(device)

    # debug print
    logger.print(">>> Loading data")

    # setup dataloaders
    dataloader_train = setup_dataloader(config_data, config_data['train_selection_filepath'])
    dataloader_test = setup_dataloader(config_data, config_data['test_selection_filepath'])

    # debug print
    logger.print(f"> training data size: {len(dataloader_train)}")
    logger.print(f"> testing data size: {len(dataloader_test)}")

    # debug print
    logger.print(">>> Starting training")

    # send model to device
    model = model.to(device)

    # define losses functions
    criterion = pt.nn.BCEWithLogitsLoss(reduction="none")

    # define optimizer
    optimizer = pt.optim.Adam(model.parameters(), lr=config_runtime["learning_rate"])

    # restart timer
    logger.restart_timer()

    # summary writer
    writer = SummaryWriter(os.path.join(output_path, 'tb'))

    # min loss initial value
    min_loss = 1e9

    # quick training step on largest data: memory check and pre-allocation
    batch_data = collate_batch_data([dataloader_train.dataset.get_largest()])
    optimizer.zero_grad()
    losses, _, _ = eval_step(model, device, batch_data, criterion, pos_ratios, config_runtime['pos_weight_factor'], global_step)
    loss = pt.sum(losses)
    loss.backward()
    optimizer.step()

    # start training
    for epoch in range(config_runtime['num_epochs']):
        # train mode
        model = model.train()

        # train model
        train_results = []
        for batch_train_data in tqdm(dataloader_train):
            # global step
            global_step += 1

            # set gradient to zero
            optimizer.zero_grad()

            # forward propagation
            losses, y, p = eval_step(model, device, batch_train_data, criterion, pos_ratios, config_runtime['pos_weight_factor'], global_step)

            # backward propagation
            loss = pt.sum(losses)
            loss.backward()

            # optimization step
            optimizer.step()

            # store evaluation results
            train_results.append([losses.detach().cpu(), y.cpu(), p.cpu()])

            # log step
            if (global_step+1) % config_runtime["log_step"] == 0:
                # process evaluation results
                with pt.no_grad():
                    # scores evaluation results and reset buffer
                    scores = scoring(train_results, device=device)
                    train_results = []

                    # logging
                    logging(logger, writer, scores, global_step, pos_ratios, "train")

                    # save model checkpoint
                    model_filepath = os.path.join(output_path, 'model_ckpt.pt')
                    pt.save(model.state_dict(), model_filepath)

            # evaluation step
            if (global_step+1) % config_runtime["eval_step"] == 0:
                # evaluation mode
                model = model.eval()

                with pt.no_grad():
                    # evaluate model
                    test_results = []
                    for step_te, batch_test_data in enumerate(dataloader_test):
                        # forward propagation
                        losses, y, p = eval_step(model, device, batch_test_data, criterion, pos_ratios, config_runtime['pos_weight_factor'], global_step)

                        # store evaluation results
                        test_results.append([losses.detach().cpu(), y.cpu(), p.cpu()])

                        # stop evaluating
                        if step_te >= config_runtime['eval_size']:
                            break

                    # scores evaluation results
                    scores = scoring(test_results, device=device)

                    # logging
                    logging(logger, writer, scores, global_step, pos_ratios, "test")

                    # save model and update min loss
                    if min_loss >= scores['loss']:
                        # update min loss
                        min_loss = scores['loss']
                        # save model
                        model_filepath = os.path.join(output_path, 'model.pt')
                        logger.print("> saving model at {}".format(model_filepath))
                        pt.save(model.state_dict(), model_filepath)

                # back in train mode
                model = model.train()


if __name__ == '__main__':
    # train model
    train(config_data, config_model, config_runtime, '.')
