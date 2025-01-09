import pytorch_lightning as pl
import torch
from torch import nn
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class MyAwesomeModel(pl.LightningModule):
    """My awesome model."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128, 10)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc1(x)

    def training_step(self, batch, batch_idx):
        """Training step."""
        img, target = batch
        y_pred = self(img)
        loss = self.loss_fn(y_pred, target)
        acc = (target == y_pred.argmax(dim=-1)).float().mean()
        
        # Log scalar metrics
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        
        # Log non-scalar data using wandb directly
        if batch_idx % 100 == 0:  # Log every 100 batches to avoid overhead
            self.logger.experiment.log({
                'logits_dist': wandb.Histogram(y_pred.detach().cpu().numpy())
            })
        
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        img, target = batch
        y_pred = self(img)
        loss = self.loss_fn(y_pred, target)
        acc = (target == y_pred.argmax(dim=-1)).float().mean()
        
        # Log with on_epoch=True to aggregate metrics over the epoch
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_acc', acc, on_epoch=True)

    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.Adam(self.parameters(), lr=1e-3)


def get_data():
    """Prepare MNIST data loaders."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download and load MNIST dataset
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)
    
    # Split training set into train and validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Initialize model
    model = MyAwesomeModel()
    
    # Setup callbacks
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=3,
        verbose=True,
        mode="min"
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath="./models",
        monitor="val_loss",
        mode="min"
    )
    
    # Initialize wandb logger
    wandb_logger = WandbLogger(project="dtu_mlops")
    
    # Setup trainer with all specified requirements
    trainer = Trainer(
        max_epochs=10,  # Limit number of epochs
        limit_train_batches=0.2,  # Use only 20% of training data
        callbacks=[early_stopping_callback, checkpoint_callback],
        logger=wandb_logger,
        default_root_dir="./lightning_logs"  # Sets directory for logs
    )
    
    # Get data loaders
    train_loader, val_loader, test_loader = get_data()
    
    # Train the model
    trainer.fit(model, train_loader, val_loader)