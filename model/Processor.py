import os
from tqdm import tqdm

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
from torch.utils.data.dataloader import DataLoader

from model.utils.LogReport import LogReport
from model.Predictor import Predictor
from model.BengaliClassifier import BengaliClassifier

from dataset.BengaliAIDataset import BengaliAIDataset
from dataset.Transform import Transform


class Processor():

    epoch = 0

    def __init__(self, args):

        self.args = args
        self.device = torch.device(self.args.device)


        # setup dataloader
        train_transform = Transform(train = True,
                                    mode = 'train',
                                    args = args)

        valid_transform = Transform(train=True,
                                    mode = 'valid',
                                    args = args)

        if args.debug:
            indices = list(range(1000))
        else:
            indices = None

        train_dataset = BengaliAIDataset(path = args.train_path, 
                                        transform = train_transform,
                                        indices= indices)

        valid_dataset = BengaliAIDataset(path = args.valid_path, 
                                        transform = valid_transform,
                                        indices = indices)

        self.num_train = len(train_dataset)
        self.num_valid = len(valid_dataset)

        print()
        print('train: {}, valid: {}'.format(self.num_train, self.num_valid))
        print()

        self.train_loader =  DataLoader(train_dataset, 
                                    batch_size = args.batch_size, 
                                    shuffle = True, 
                                    num_workers = args.num_workers)

        self.valid_loader = DataLoader(valid_dataset, 
                                    batch_size = args.batch_size, 
                                    shuffle = False, 
                                    num_workers = args.num_workers)


        predictor = Predictor()
        self.classifier = BengaliClassifier(predictor, 
                                        cutmix_ratio = args.cutmix_ratio, 
                                        cutmix_bien = args.cutmix_bien).to(self.device)

        self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr = args.lr)


    def init_metrics(self):

        self.train_metrics = {'loss' : 0.0, 
                'loss_grapheme' : 0.0, 
                'loss_vowel' : 0.0,
                'loss_consonant' : 0.0}

        self.eval_metrics = {'loss' : 0.0, 
                'loss_grapheme' : 0.0, 
                'loss_vowel' : 0.0,
                'loss_consonant' : 0.0}

    def load_checkpoint(self):

        if os.path.isfile(self.args.checkpoint_path):
            checkpoint = torch.load(self.args.checkpoint_path, map_location=self.device)
            
            self.classifier.predictor.load_state_dict(checkpoint['predictor'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epoch = checkpoint['epoch']

        else:
            raise ValueError('Do NOT exist this checkpoint: {}'.format(self.args.checkpoint_path))
        
    
    def save_checkpoint(self):
        if not os.path.exists(self.args.output_path):
            os.mkdir(self.args.output_path)

        predictor = self.classifier.predictor.state_dict()
        for key in predictor.keys():
            predictor[key] = predictor[key].cpu()

        checkpoint = {'predictor' : predictor, 
                        'optimizer' : self.optimizer, 
                        'epoch' : self.epoch}

        torch.save(checkpoint, 
                    self.args.output_path + 'checkpoint_{}.pth.tar'.format(self.epoch))


    def train(self):
        self.classifier.train()
        self.optimizer.zero_grad()

        for x, y in tqdm(self.train_loader):

            x = x.to(self.device)
            y = y.to(self.device)
            
            loss, metrics, pred_y = self.classifier(x, y)
            
            loss.backward()
            self.optimizer.step()

            for key in self.train_metrics:
                self.train_metrics[key] += metrics[key] * x.shape[0]

    def eval(self):
        self.classifier.eval()

        with torch.no_grad():
            
            for x, y in tqdm(self.valid_loader):

                x = x.to(self.device)
                y = y.to(self.device)

                loss, metrics, pred_y = self.classifier.forward_eval(x, y)

                for key in self.eval_metrics:
                    self.eval_metrics[key] += metrics[key] * x.shape[0]

    def start(self):
        
        if self.args.continue_train:
            self.load_checkpoint()

        self.log_report = LogReport(self.args.output_path)

        while self.epoch < self.args.max_epoch:

            self.epoch += 1
            self.init_metrics()

            print('epoch {}/ {}'.format(self.epoch, self.args.max_epoch))
            self.train()

            self.eval()

            self.log_report.update(epoch = self.epoch, 
                                    train_metrics = self.train_metrics, 
                                    eval_metrics = self.eval_metrics,
                                    num_train = self.num_train,
                                    num_valid = self.num_valid)

            self.save_checkpoint()

        

