import os
import json
from time import perf_counter


class LogReport:
    def __init__(self, trainer = None, metric = None, dirpath=None, logger=None):

        self.trainer = trainer
        self.metric = metric

        self.dirpath = str(dirpath) 
        self.history = []

        self.start_time = perf_counter()

    @staticmethod
    def save_json(filepath, params):
        with open(filepath, 'w') as f:
            json.dump(params, f, indent=4)

    def update(self):

        elapsed_time = perf_counter() - self.start_time

        elem = {'epoch': self.trainer.epoch,
                'iteration': self.trainer.iteration}

        elem.update({f'train/{key}': value
                     for key, value in self..metric.keys()})

        elem.update({f'valid/{key}': value
                        for key, value in self.metric.keys()})

        elem['elapsed_time'] = elapsed_time

        self.history.append(elem)

        if self.dirpath:
            save_json(os.path.join(self.dirpath, 'log.json'), self.history)
            self.get_dataframe().to_csv(os.path.join(self.dirpath, 'log.csv'), index=False)


    def get_dataframe(self):
        df = pd.DataFrame(self.history)
        return df