from tqdm import tqdm

import torch

from model.utils.LogReport import LogReport

class Trainer():

    def __init__(self, 
                train_loader = None, 
                test_loader = None,
                classifier = None, 
                optimizer = None,
                max_epoch = 20,
                continue_train = False,
                checkpoint_path = '',
                log_path = ''
                ):

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.classifier = classifier
        self.optimizer = optimizer
        self.max_epoch = max_epoch
        self.epoch = 0


        self.continue_train = continue_train
        self.checkpoint_path = checkpoint_path
        if self.continue_train:
            self.load_pretrain()

        self.log_path = log_path
        self.log_report = LogReport(log_path)
    
    def load_checkpoint(self):
        return 
    
    def save_checkpoint(self):
        return


    def train(self):
        self.epoch += 1

        self.classifier.train()
        self.optimizer.zero_grad()
        
        train_loader_bar = tqdm(range(len(self.train_loader)), 
                                desc="Epoch {}".format(epoch+1), position=0)

        train_loader_desc = tqdm(total=0, position=1, bar_format='{desc}')

        for num, (x, y) in enumerate(self.train_loader):

            loss, metrics, pred_y = self.classifier(x, y)
            
            loss.backward()
            self.optimizer.step()

            train_loader_desc.set_description('Epoch: {}/{}, Batch: {}/{}'
                            .format(self.epoch, self.max_epoch, num+1, len(self.train_loader)))

        
        self.log_report.update(self, metrics)

        self.save_checkpoint()

    def test(self):
        self.classifier.eval()

        with torch.no_grad():
            for num, (x, y) in enumerate(self.test_loader):

                loss, metrics, pred_y = self.classifier(x, y)
            


