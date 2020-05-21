from dataset.dataset import BengaliAIDataset
from dataset.utils import load_data_trung
from dataset.augment_new import *

from model.BengaliClassifier import BengaliClassifier
from model.PretrainedCNN_b4 import PretrainedCNN, PretrainedCNNAfterPretrain1292
from model.utils import *

import os

from numpy.random.mtrand import RandomState

from pathlib import Path

import torch
from torch.utils.data.dataloader import DataLoader
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Events


if __name__ == "__main__":


    ################################### FREEZE ###############################
    
    debug=False
    batch_size=32
    device='cuda:3'

    image_path = '/media/data/trung/deep/data/images-fold2/' # forder alll image    # phải có / ở cuối 
    train_path = 'fold_2_train.csv' # file csv
    test_path = 'fold_2_test.csv'   # file csv
    output_path = 'output-freeze/'  # phải có / ở cuối 
    outdir = Path(output_path)

    base_model_path = 'output-pretrain-1292_2/model_26.pt'
    pretrain_path = '' # tiếp tục train thì nhớ thêm vào
    # optimizer_path = 'output-pretrain-1292/'
    optimizer_path = ''

    #--- scheduler ---
    lr = 0.001
    patience = 20 #    num bad epoch phải -1 
    factor = float(1/(10**(1/2)))

    #--- input ---
    image_size = (137, 236) # nếu train với size gốc phải để crop = False

    crop = False
    padding = 16

    use_cutout = True

    cutmix_ratio = 1
    cutmix_bien = 0.5

    auto_aug = True

    # --- load data ---
    train_images, test_images, train_labels, test_labels = load_data_trung(train_path = train_path,
                                                                            test_path = test_path)

    train_transform = Transform(path = image_path,
                                train=True,
                                mode = 'train',
                                crop=crop,
                                padding=padding,
                                image_size=image_size,
                                use_cutout = use_cutout,
                                auto_aug = auto_aug)

    valid_transform = Transform(path = image_path,
                                train=True,
                                mode = 'valid',
                                image_size=image_size)

    if debug :
        train_dataset = BengaliAIDataset(
            train_images[:2000], 
            train_labels[:2000], 
            transform=train_transform)

        valid_dataset = BengaliAIDataset(
            test_images[:1000], 
            test_labels[:1000], 
            transform=valid_transform)
        
        epoch = 10
    else:
        train_dataset = BengaliAIDataset(
            train_images, 
            train_labels, 
            transform=train_transform)

        valid_dataset = BengaliAIDataset(
            test_images, 
            test_labels, 
            transform=valid_transform)

        epoch = 10
    

    print('train_dataset', len(train_dataset), 'valid_dataset', len(valid_dataset))

    # --- Model ---
    
    device = torch.device(device)

    predictor = PretrainedCNNAfterPretrain1292(freeze = True, PATH = base_model_path, device = device)

    if pretrain_path != '':
        predictor.load_state_dict(torch.load(pretrain_path, map_location = device))

    classifier = BengaliClassifier(predictor, 
                                    cutmix_ratio = cutmix_ratio, 
                                    cutmix_bien = cutmix_bien).to(device)

    # --- Training setting ---

    train_loader =  DataLoader(train_dataset, batch_size=batch_size * 8, shuffle=True, num_workers=8)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size * 8, shuffle=False, num_workers=8)

    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=factor, patience=patience, min_lr=1e-10)

    trainer = create_trainer(classifier, optimizer, device)
    
    def output_transform(output):
        metric, pred_y, y = output
        return pred_y.cpu(), y.cpu()

    EpochMetric(
        compute_fn=macro_recall,
        output_transform=output_transform
    ).attach(trainer, 'recall')

    pbar = ProgressBar()
    pbar.attach(trainer, metric_names='all')

    evaluator = create_evaluator(classifier, device)
    EpochMetric(
        compute_fn=macro_recall,
        output_transform=output_transform
    ).attach(evaluator, 'recall')

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
    
    ################################### UNFREEZE ###############################
    debug = False
    batch_size=64
    device='cuda:3'

    image_path = '/media/data/trung/deep/data/images-fold2/' # forder alll image    # phải có / ở cuối 
    train_path = 'fold_2_train.csv' # file csv
    test_path = 'fold_2_test.csv'   # file csv
    output_path = 'output-unfreeze/'  # phải có / ở cuối 
    outdir = Path(output_path)

    pretrain_path = 'output-freeze/model_10.pt' # tiếp tục train thì nhớ thêm vào
    optimizer_path = 'output-freeze/optimizer_10.pt'

    #--- scheduler ---
    lr = 0.001

    #--- input ---
    image_size = (137, 236) # nếu train với size gốc phải để crop = False

    crop = False
    padding = 16

    use_cutout = True
    cutmix_ratio = 1
    cutmix_bien = 0.5

    auto_aug = True

    # --- load data ---
    train_images, test_images, train_labels, test_labels = load_data_trung(train_path = train_path,
                                                                            test_path = test_path)

    train_transform = Transform(path = image_path,
                                train=True,
                                mode = 'train',
                                crop=crop,
                                padding=padding,
                                image_size=image_size,
                                use_cutout = use_cutout,
                                auto_aug = auto_aug)

    valid_transform = Transform(path = image_path,
                                train=True,
                                mode = 'valid',
                                image_size=image_size)

    if debug :
        train_dataset = BengaliAIDataset(
            train_images[:2000], 
            train_labels[:2000], 
            transform=train_transform)

        valid_dataset = BengaliAIDataset(
            test_images[:1000], 
            test_labels[:1000], 
            transform=valid_transform)
        
        epoch = 10
    else:
        train_dataset = BengaliAIDataset(
            train_images, 
            train_labels, 
            transform=train_transform)

        valid_dataset = BengaliAIDataset(
            test_images, 
            test_labels, 
            transform=valid_transform)

        epoch = 20
    
    device = torch.device(device)

    print('train_dataset', len(train_dataset), 'valid_dataset', len(valid_dataset))

    predictor = PretrainedCNNAfterPretrain1292()

    if pretrain_path != '':
        predictor.load_state_dict(torch.load(pretrain_path, map_location=device))

    classifier = BengaliClassifier(predictor, 
                                    cutmix_ratio = cutmix_ratio, 
                                    cutmix_bien = cutmix_bien).to(device)

    # --- Training setting ---

    train_loader =  DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    
    if optimizer_path != '':
        optimizer.load_state_dict(torch.load(optimizer_path, map_location = device))


    scheduler = torch.optim.lr_scheduler.MultiStepLR(
                                                    optimizer, 
                                                    milestones =[15, 25, 40, 50] , 
                                                    gamma = float(1)
                                                    )

    trainer = create_trainer(classifier, optimizer, device)
    
    def output_transform(output):
        metric, pred_y, y = output
        return pred_y.cpu(), y.cpu()

    EpochMetric(
        compute_fn=macro_recall,
        output_transform=output_transform
    ).attach(trainer, 'recall')

    pbar = ProgressBar()
    pbar.attach(trainer, metric_names='all')

    evaluator = create_evaluator(classifier, device)
    EpochMetric(
        compute_fn=macro_recall,
        output_transform=output_transform
    ).attach(evaluator, 'recall')

    def run_evaluator(engine):
        evaluator.run(valid_loader)

    def schedule_lr(engine):
        metrics = engine.state.metrics
        avg_mae = metrics['loss']

        # --- update lr ---
        lr = scheduler.optimizer.param_groups[0]['lr']
        scheduler.step()
        log_report.report('lr', lr)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, run_evaluator)

    # trainer.add_event_handler(Events.EPOCH_COMPLETED, schedule_lr)

    evaluator.add_event_handler(Events.EPOCH_COMPLETED, schedule_lr)

    log_report = LogReport(evaluator, outdir)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, log_report)

    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        ModelSnapshotHandler(predictor, optimizer,  filepath = output_path))

    trainer.run(train_loader, max_epochs=epoch)

    train_history = log_report.get_dataframe()
    train_history.to_csv(outdir / 'log.csv', index=False)

    train_history

    