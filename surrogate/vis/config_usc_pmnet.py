"""
用途:
  保留旧版 USC PMNet 训练配置模板，方便复现实验参数。

直接运行命令:
  无。该文件是配置定义模块，不单独运行。

导出对象与参数:
  config_usc_pmnet_v1()
    返回一份带默认超参数的配置对象。
"""


class config_usc_pmnet_v1:
    def __init__(self):
        self.batch_size = 4
        self.exp_name = "config_usc_pmnet_v1"
        self.num_epochs = 3
        self.val_freq = 1
        self.num_workers = 0

        self.train_ratio = 0.9
        self.validation_ratio = 0.1
        self.test_ratio = 0.1

        self.dataset_settings()
        self.optim_settings()

    def dataset_settings(self):
        self.dataset = "USC"
        self.cityMap = "complete"
        self.sampling = "exclusive"

    def optim_settings(self):
        self.lr = 5e-4
        self.lr_decay = 0.5
        self.step = 10

    def get_train_parameters(self):
        return {
            "exp_name": self.exp_name,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "lr": self.lr,
            "lr_decay": self.lr_decay,
            "step": self.step,
            "sampling": self.sampling,
        }
