from dataset.dataset import BengaliAIDataset
from dataset.utils import load_data_1295
from dataset.augment_new import *

from model.BengaliClassifier import BengaliClassifier1295
from model.PretrainedCNN_b4 import PretrainedCNN1295
from model.utils import *

import os

from numpy.random.mtrand import RandomState

from pathlib import Path

import torch
from torch.utils.data.dataloader import DataLoader
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Events


if __name__ == "__main__":

    debug=True
    batch_size=64
    device='cuda:1'


    image_path = '/media/data/trung/deep/data/images/' # forder alll image    # phải có / ở cuối 
    train_path = 'train.csv' # file csv
    output_path = 'output-pretrain-1292-1/'  # phải có / ở cuối
    outdir = Path(output_path)
    
    pretrain_path = 'output-pretrain-1292/model_10.pt' # tiếp tục train thì nhớ thêm vào
    optimizer_path = 'output-pretrain-1292/optimizer_10.pt'
    #--- scheduler ---
    lr = 0.001
    patience = 2 #    num bad epoch phải -1 
    factor = float(1/(10**(1/2)))

    #--- input ---
    image_size = (137, 236) # nếu train với size gốc phải để crop = False

    crop = False
    padding = 16

    use_cutout = False

    cutmix_ratio = 0
    cutmix_bien = 0

    auto_aug = False

    # --- load data ---
    train_images,  train_labels = load_data_1295(train_path = train_path)
    test_images = train_images
    test_labels = train_labels
    train_transform = Transform(path = image_path,
                                train=True,
                                mode = 'train',
                                crop = crop,
                                padding = padding,
                                image_size = image_size,
                                use_cutout = use_cutout,
                                auto_aug = auto_aug)

    valid_transform = Transform(path = image_path,
                                train=True,
                                mode = 'valid',
                                image_size=image_size)

    if debug :
        train_dataset = BengaliAIDataset(
            train_images[:100], 
            train_labels[:100], 
            transform=train_transform)

        valid_dataset = BengaliAIDataset(
            train_images[:100], 
            train_labels[:100], 
            transform=valid_transform)
        
        epoch = 10
    else:
        train_dataset = BengaliAIDataset(
            train_images, 
            train_labels, 
            transform=train_transform)

        valid_dataset = BengaliAIDataset(
            train_images, 
            train_labels, 
            transform=valid_transform)

        epoch = 200
    

    print('train_dataset', len(train_dataset), 'valid_dataset', len(valid_dataset))

    # --- Model ---
    device = torch.device(device)

    predictor = PretrainedCNN1295()

    if pretrain_path != '':
        predictor.load_state_dict(torch.load(pretrain_path))

    classifier = BengaliClassifier1295(predictor, 
                                        cutmix_ratio = cutmix_ratio, 
                                        cutmix_bien = cutmix_bien).to(device)

    # --- Training setting ---

    train_loader =  DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)

    if optimizer_path != '':
        optimizer.load_state_dict(torch.load(optimizer_path))

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=factor, patience=patience, min_lr=1e-10)

    trainer = create_trainer(classifier, optimizer, device)
    
    def output_transform(output):
        metric, pred_y, y = output
        return pred_y.cpu(), y.cpu()


    pbar = ProgressBar()
    pbar.attach(trainer, metric_names='all')

    evaluator = create_evaluator(classifier, device)

    def run_evaluator(engine):
        evaluator.run(valid_loader)

    def schedule_lr(engine):
        metrics = engine.state.metrics
        avg_mae = metrics['loss']

        # --- update lr ---
        lr = scheduler.optimizer.param_groups[0]['lr']
        scheduler.step(avg_mae)
        log_report.report('lr', lr)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, run_evaluator)
    # trainer.add_event_handler(Events.EPOCH_COMPLETED, schedule_lr)
    evaluator.add_event_handler(Events.EPOCH_COMPLETED, schedule_lr)
    log_report = LogReport(evaluator, outdir)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, log_report)
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        ModelSnapshotHandler(predictor, optimizer, filepath = output_path))

    trainer.run(train_loader, max_epochs=epoch)

    train_history = log_report.get_dataframe()
    train_history.to_csv(outdir / 'log.csv', index=False)

    train_history