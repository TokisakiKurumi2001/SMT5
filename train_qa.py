from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
import pytorch_lightning as pl
from SMT5 import SMT5QADataLoader, LitSMT5
from math import ceil

if __name__ == "__main__":
    wandb_logger = WandbLogger(project="proj_new_dummy")
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # num row MKQA: ~100k
    # num row TSNLI: ~90k
    qa_bz = 64
    num_rows = 101370
    epochs = 20
    hyperparameter = {
        "ckpt": "smt5_ckpt",
        "lr": 1e-3,
        "num_keep_steps": int(0.1 * ceil(num_rows / qa_bz * epochs)),
        "num_training_steps": int(ceil(num_rows / qa_bz * epochs)),
        "name": "qa",
    }
    lit_smt5 = LitSMT5(**hyperparameter)

    # dataloader
    smt5_qa_dataloader = SMT5QADataLoader(ckpt=hyperparameter['ckpt'], query_max_length=100, answer_max_length=50)
    [qa_train_dataloader] = smt5_qa_dataloader.get_dataloader(batch_size=qa_bz, types=["train"])

    # train model
    trainer = pl.Trainer(max_epochs=epochs, devices=[0], accelerator="gpu", logger=wandb_logger, callbacks=[lr_monitor])
    trainer.fit(model=lit_smt5, train_dataloaders=qa_train_dataloader)

    # save model & tokenizer
    lit_smt5.export_model('smt5_model/v1')
