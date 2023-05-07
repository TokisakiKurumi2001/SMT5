from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from SMT5 import SMT5QADataLoader, LitSMT5
from math import ceil

if __name__ == "__main__":
    wandb_logger = WandbLogger(project="proj_new_dummy")

    # num row MKQA: ~100k
    # num row TSNLI: ~90k
    nli_bz = 256
    num_rows = 9000
    hyperparameter = {
        "ckpt": "smt5_mmodel/v1/encoder",
        "mapper_ckpt": "smt5_mmodel/v1/mapper",
        "mode": "train_stage2",
        "lr": 1e-3,
        "num_keep_steps": int(0.1 * ceil(num_rows / nli_bz)),
        "num_training_steps": int(ceil(num_rows / nli_bz)),
        "name": "nli",
    }
    lit_smt5 = LitSMT5(**hyperparameter)

    # dataloader
    smt5_nli_dataloader = SMT5TSNLIDataLoader(ckpt="smt5_ckpt", max_length=128)
    [nli_train_dataloader] = smt5_nli_dataloader.get_dataloader(batch_size=nli_bz, types=["train"])

    # train model
    trainer = pl.Trainer(max_epochs=2, devices=[0], accelerator="gpu", logger=wandb_logger)
    trainer.fit(model=lit_smt5, train_dataloaders=qa_train_dataloader)

    # save model & tokenizer
    lit_smt5.export_model('smt5_model/v1.5')
