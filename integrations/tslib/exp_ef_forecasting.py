"""EF experiment for Time-Series-Library.

Copy this file to ``Time-Series-Library/exp/`` and register it in ``run.py``
(see the README). Training is TSLib's standard long-term forecasting with
``--pred_len L``, i.e. EF optimizes only the first reasoning block. Testing
then evaluates the same model at every horizon in ``--ef_eval_horizons``.
"""

import os

import torch

from ef import EvolutionaryForecaster
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast


class Exp_EF_Forecast(Exp_Long_Term_Forecast):

    def test(self, setting, test=0):
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        block_len = self.args.pred_len
        horizons = self.args.ef_eval_horizons or [block_len]
        inner = self.model
        self.model = EvolutionaryForecaster(inner, block_len=block_len, label_len=self.args.label_len)
        try:
            for horizon in horizons:
                self.args.pred_len = horizon
                print(f'EF test: block length L={block_len}, evaluation horizon H={horizon}')
                super().test(f'{setting}_H{horizon}', test=0)
        finally:
            self.model = inner
            self.args.pred_len = block_len
