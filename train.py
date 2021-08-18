import torch
from tqdm import tqdm
from config import cfg
from base import Trainer

def main():
    torch.backends.cudnn.benchmark = True
    trainer = Trainer()
    trainer._make_model()
    trainer._make_batch_loader()
    for epoch in range(trainer.start_epoch, cfg.total_epoch):
        for iteration, (inputs, targets, meta_infos) in tqdm(enumerate(trainer.batch_loader)):
            trainer.optimizer.zero_grad()
            loss = trainer.model(inputs, targets)
            sum(loss[k] for k in loss).backward()
            trainer.optimizer.step()
            if not iteration % cfg.print_iter:
                screen = ['[Epoch %d/%d]' % (epoch, cfg.total_epoch),
                          '[Batch %d/%d]' % (iteration, len(trainer.batch_loader)),
                          '[lr %f]' % (trainer.get_lr())]
                screen += ['[%s: %.4f]' % ('loss_' + k, v.detach()) for k, v in loss.items()]
                trainer.logger.info(''.join(screen))
        trainer.schedule.step()
        if not epoch % cfg.save_epoch:
            trainer.save_model(trainer.model, trainer.optimizer, trainer.schedule, epoch)

if __name__ == '__main__':
    main()



