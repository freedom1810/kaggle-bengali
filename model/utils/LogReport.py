import os
import json
from time import perf_counter


class LogReport:
    def __init__(self,  
                dirpath=None
                ):

        self.dirpath = str(dirpath) 
        self.history = []

        self.start_time = perf_counter()



    def update(self, 
                epoch = None, 
                train_metrics = None,
                eval_metrics = None
                ):

        elapsed_time = perf_counter() - self.start_time

        elem = {'epoch': epoch}

        elem.update({f'train/{key}': value
                     for key, value in train_metrics.keys()})

        elem.update({f'valid/{key}': value
                        for key, value in eval_metrics.keys()})

        elem['elapsed_time'] = elapsed_time

        self.history.append(elem)

        if self.dirpath:
            save_json(os.path.join(self.dirpath, 'log.json'), self.history)
            self.get_dataframe().to_csv(os.path.join(self.dirpath, 'log.csv'), index=False)


    def get_dataframe(self):
        df = pd.DataFrame(self.history)
        return df


def save_json(filepath, params):
    with open(filepath, 'w') as f:
        json.dump(params, f, indent=4)