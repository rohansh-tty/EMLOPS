import os
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar, RichModelSummary
from lightning.pytorch.loggers import TensorBoardLogger
from src.datamodules.dogbreed_datamodule import DogBreedDataModule
from src.models.catdog_classifier import CatDogClassifier
from src.utils.logging_utils import setup_logger, task_wrapper
from src.utils.split_dataset import split_dataset

@task_wrapper
def eval(data_module, model, trainer, ckpt):
    trainer.fit(model, data_module)

def main():
    # Set up paths
    base_dir = Path(__file__).resolve().parent.parent / "src"
    data_dir = base_dir / "data"
    log_dir = base_dir / "logs"
    checkpoint_dir = base_dir / "checkpoints"
    
    # Set up logger
    setup_logger(log_dir / "eval_log.log")

    # Initialize DataModule
    data_module = DogBreedDataModule(data_dir, batch_size=32)#, num_workers=4)
    
    # setup dataset
    split_dataset(
    source_dir=data_dir / 'dataset',  # Your original dataset with breed folders
    output_dir=data_dir,      # Where to create the train/val split
    val_split=0.5                   # 50% for validation
)
   
         
    # Initialize Model
    # check if model exists in checkpoint folder
    checkpoints = [m for m in os.listdir(checkpoint_dir)]
    if checkpoints:
        model_file = sorted(checkpoints)[0] # choosing model wiht lowest loss
        model_file_path = os.path.join(checkpoint_dir, model_file)
        model = CatDogClassifier.load_from_checkpoint(model_file_path)
    else:
        model = CatDogClassifier(lr=1e-3)

    # Set up checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="{epoch:02d}-val_loss={val_loss:.2f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
    )

    # Initialize Trainer
    trainer = L.Trainer(
        max_epochs=3,
        callbacks=[
            checkpoint_callback,
            RichProgressBar(),
            RichModelSummary(max_depth=2)
        ],
        default_root_dir=checkpoint_dir,
        accelerator="auto",
        logger=TensorBoardLogger(save_dir=log_dir, name="catdog_classification"),
    )

    # Train and test the model
    eval(data_module, model, trainer, checkpoint_dir)

if __name__ == "__main__":
    main()