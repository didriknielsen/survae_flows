import torch
import torchtestcase
import unittest
import torch.nn as nn
from torch.optim import SGD
from survae.optim.schedulers import LinearWarmupScheduler


class LinearWarmupSchedulerTest(torchtestcase.TorchTestCase):

    def test_lr(self):
        model = nn.Linear(10,5)
        optimizer = SGD(model.parameters(), lr=0.1)
        scheduler = LinearWarmupScheduler(optimizer, total_epoch=5)

        for i in range(5):
            current_lr = optimizer.param_groups[0]['lr']
            self.assertEqual(current_lr, 0.1 * (i/5))
            optimizer.step()
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        self.assertEqual(current_lr, 0.1)


if __name__ == '__main__':
    unittest.main()
