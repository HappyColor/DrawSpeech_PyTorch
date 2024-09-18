# mainly from https://github.com/haoheliu/AudioLDM-training-finetuning


import sys

sys.path.append("src")

import os
import wandb

import argparse
import yaml
import torch
from pytorch_lightning.strategies.ddp import DDPStrategy
from drawspeech.utilities.data.dataset import AudioDataset
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from drawspeech.modules.latent_encoder.autoencoder import AutoencoderKL
from pytorch_lightning.callbacks import ModelCheckpoint
from drawspeech.utilities.tools import get_restore_step


def main(configs, exp_group_name, exp_name):
    if "precision" in configs.keys():
        torch.set_float32_matmul_precision(configs["precision"])
    batch_size = configs["model"]["params"]["batchsize"]
    log_path = configs["log_directory"]

    if "dataloader_add_ons" in configs["data"].keys():
        dataloader_add_ons = configs["data"]["dataloader_add_ons"]
    else:
        dataloader_add_ons = []

    dataset = AudioDataset(configs, split="train", add_ons=dataloader_add_ons)

    loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True
    )

    print(
        "The length of the dataset is %s, the length of the dataloader is %s, the batchsize is %s"
        % (len(dataset), len(loader), batch_size)
    )

    val_dataset = AudioDataset(configs, split="val", add_ons=dataloader_add_ons)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=8,
        shuffle=True,
    )

    model = AutoencoderKL(
        ddconfig=configs["model"]["params"]["ddconfig"],
        lossconfig=configs["model"]["params"]["lossconfig"],
        embed_dim=configs["model"]["params"]["embed_dim"],
        image_key=configs["model"]["params"]["image_key"],
        base_learning_rate=configs["model"]["base_learning_rate"],
        subband=configs["model"]["params"]["subband"],
        sampling_rate=configs["preprocessing"]["audio"]["sampling_rate"],
    )

    try:
        config_reload_from_ckpt = configs["reload_from_ckpt"]
    except:
        config_reload_from_ckpt = None

    checkpoint_path = os.path.join(log_path, exp_group_name, exp_name, "checkpoints")

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        monitor="global_step",
        mode="max",
        filename="checkpoint-{global_step:.0f}",
        every_n_train_steps=configs["step"]["save_checkpoint_every_n_steps"],
        save_top_k=configs["step"]["save_top_k"],
        auto_insert_metric_name=False,
        save_last=configs["step"]["save_last"],
    )

    wandb_path = os.path.join(log_path, exp_group_name, exp_name)

    model.set_log_dir(log_path, exp_group_name, exp_name)

    os.makedirs(checkpoint_path, exist_ok=True)

    if len(os.listdir(checkpoint_path)) > 0:
        print("Load checkpoint from path: %s" % checkpoint_path)
        restore_step, n_step = get_restore_step(checkpoint_path)
        resume_from_checkpoint = os.path.join(checkpoint_path, restore_step)
        print("Resume from checkpoint", resume_from_checkpoint)
    elif config_reload_from_ckpt is not None:
        resume_from_checkpoint = config_reload_from_ckpt
        print("Reload ckpt specified in the config file %s" % resume_from_checkpoint)
    else:
        print("Train from scratch")
        resume_from_checkpoint = None

    devices = torch.cuda.device_count()

    wandb_logger = WandbLogger(
        save_dir=wandb_path,
        project=configs["project"],
        config=configs,
        name="%s/%s" % (exp_group_name, exp_name),
    )

    trainer = Trainer(
        accelerator="gpu",
        devices=devices,
        logger=wandb_logger,
        max_steps=configs["step"]["max_steps"],
        limit_val_batches=configs["step"]["limit_val_batches"],
        callbacks=[checkpoint_callback],
        strategy=DDPStrategy(find_unused_parameters=True),
        # val_check_interval=configs["step"]["val_check_interval"],
        check_val_every_n_epoch=configs["step"]["validation_every_n_epochs"],
    )

    # TRAINING
    trainer.fit(model, loader, val_loader, ckpt_path=resume_from_checkpoint)

    # EVALUTION
    # trainer.test(model, test_loader, ckpt_path=resume_from_checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--autoencoder_config",
        type=str,
        required=True,
        help="path to autoencoder config .yam",
    )

    args = parser.parse_args()

    config_yaml = args.autoencoder_config
    exp_name = os.path.basename(config_yaml.split(".")[0])
    exp_group_name = os.path.basename(os.path.dirname(config_yaml))

    config_yaml = os.path.join(config_yaml)

    config_yaml = yaml.load(open(config_yaml, "r"), Loader=yaml.FullLoader)

    main(config_yaml, exp_group_name, exp_name)
