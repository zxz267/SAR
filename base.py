import torch
import os
from data.dataset import get_dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from model import get_model
from config import cfg
from utils.logger import setup_logger

class Trainer:
    def __init__(self):
        log_folder = os.path.join(cfg.output_root, 'log')
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
        logfile = os.path.join(log_folder, 'train_' + cfg.experiment_name + '.log')
        self.logger = setup_logger(output=logfile, name="Training")
        self.logger.info('Start training: %s' % ('train_' + cfg.experiment_name))

    def get_optimizer(self, model):
        optimizer = optim.AdamW([{'params': model.parameters(), 'initial_lr': cfg.lr}], cfg.lr)
        self.logger.info('The parameters of the model are added to the AdamW optimizer.')
        return optimizer

    def get_schedule(self, optimizer):
        schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                        T_max=cfg.total_epoch,
                                                        eta_min=0)
        self.logger.info('The learning rate schedule for the optimizer has been set to CosineAnnealingLR.')
        return schedule

    def load_model(self, model, optimizer, schedule):
        checkpoint = torch.load(cfg.checkpoint)
        self.logger.info("Loading the model of epoch-{} from {}...".format(checkpoint['last_epoch'], cfg.checkpoint))
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        schedule.load_state_dict(checkpoint['schedule'])
        start_epoch = checkpoint['last_epoch'] + 1
        self.logger.info('The model is loaded successfully.')
        return start_epoch, model

    def save_model(self, model, optimizer, schedule, epoch):
        save = {
            'net': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'schedule': schedule.state_dict(),
            'last_epoch': epoch
        }
        path_checkpoint = os.path.join(cfg.output_root, 'checkpoint', cfg.experiment_name)
        if not os.path.exists(path_checkpoint):
            os.makedirs(path_checkpoint)
        save_path = os.path.join(path_checkpoint, "checkpoint_epoch[%d_%d].pth" % (epoch, cfg.total_epoch))
        torch.save(save, save_path)
        self.logger.info('Save checkpoint to {}'.format(save_path))

    def get_lr(self):
        for g in self.optimizer.param_groups:
            cur_lr = g['lr']
        return cur_lr

    def _make_batch_loader(self):
        self.logger.info("Creating dataset...")
        dataset = get_dataset(cfg.dataset, 'training')
        self.batch_loader = DataLoader(dataset,
                                       batch_size=cfg.batch_size,
                                       num_workers=cfg.num_worker,
                                       shuffle=True,
                                       pin_memory=True,
                                       drop_last=True)
        self.logger.info("The dataset is created successfully.")

    def _make_model(self):
        self.logger.info("Making the model...")
        model = get_model().to(cfg.device)
        optimizer = self.get_optimizer(model)
        schedule = self.get_schedule(optimizer)
        if cfg.continue_train:
            start_epoch, model = self.load_model(model, optimizer, schedule)
        else:
            start_epoch = 0
        model.train()
        self.start_epoch = start_epoch
        self.model = model
        self.optimizer = optimizer
        self.schedule = schedule
        self.logger.info("The model is made successfully.")

class Tester:
    def __init__(self):
        log_folder = os.path.join(cfg.output_root, 'log')
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
        logfile = os.path.join(log_folder, 'eval_' + cfg.experiment_name + '.log')
        self.logger = setup_logger(output=logfile, name="Evaluation")
        self.logger.info('Start evaluation: %s' % ('eval_' + cfg.experiment_name))

    def _make_batch_loader(self):
        self.logger.info("Creating dataset...")
        self.dataset = get_dataset(cfg.dataset, 'evaluation')
        self.batch_loader = DataLoader(self.dataset,
                                       batch_size=cfg.batch_size,
                                       num_workers=cfg.num_worker,
                                       shuffle=False,
                                       pin_memory=True)
        self.logger.info("The dataset is created successfully.")

    def load_model(self, model):
        self.logger.info('Loading the model from {}...'.format(cfg.checkpoint))
        checkpoint = torch.load(cfg.checkpoint)
        model.load_state_dict(checkpoint['net'])
        self.logger.info('The model is loaded successfully.')
        return model

    def _make_model(self):
        self.logger.info("Making the model...")
        model = get_model().to(cfg.device)
        model = self.load_model(model)
        model.eval()
        self.model = model
        self.logger.info("The model is made successfully.")

    def _evaluate(self, outs, meta_info, cur_sample_idx):
        eval_result = self.dataset.evaluate(outs, meta_info, cur_sample_idx)
        return eval_result

    def _print_eval_result(self, eval_result):
        self.dataset.print_eval_result(eval_result)
        self.logger.info("The evaluation is done successfully.")

