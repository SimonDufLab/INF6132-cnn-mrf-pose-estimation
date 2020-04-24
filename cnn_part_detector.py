import pytorch_lightning as pl
from torch import nn
import torch
import torchvision
from torch.nn import functional as F
from data import load_train_data, load_test_data, to_dataloader, viz_sample
import config as cfg
import utils


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight=False):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints

class CrossELoss(nn.Module):
    def __init__(self, use_target_weight=False):
        super(CrossELoss, self).__init__()

    def forward(self, output, target):
        batch_size = output.size(0)
        num_joints = output.size(1)
        assert num_joints == 10
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_pred = nn.functional.log_softmax(heatmap_pred, dim=1)
            heatmap_gt = heatmaps_gt[idx].squeeze()

            loss += nn.functional.binary_cross_entropy_with_logits(heatmap_pred, heatmap_gt)

        return loss / num_joints


class PoseDetector(pl.LightningModule):

    def __init__(self):
        super(PoseDetector, self).__init__()

        self.model_size = cfg.MODEL_SIZE
        self.output_shape = (60, 90, 10)

        # Layers for full resolution image
        self.fullres_layer1 = nn.Sequential(
            nn.Conv2d(3, self.model_size * 1, 5, stride=1, padding=2),
            nn.BatchNorm2d(self.model_size * 1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU()
        )

        self.fullres_layer2 = nn.Sequential(
            nn.Conv2d(self.model_size * 1, self.model_size * 2, 5, stride=1, padding=2),
            nn.BatchNorm2d(self.model_size * 2),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU()
        )

        self.fullres_layer3 = nn.Sequential(
            nn.Conv2d(self.model_size * 2, self.model_size * 4, 9, stride=1, padding=4),
            nn.BatchNorm2d(self.model_size * 4),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU()
        )

        # Layers for half resolution image
        self.halfres_layer1 = nn.Sequential(
            nn.Conv2d(3, self.model_size * 1, 5, stride=1, padding=2),
            nn.BatchNorm2d(self.model_size * 1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU()
        )

        self.halfres_layer2 = nn.Sequential(
            nn.Conv2d(self.model_size * 1, self.model_size * 2, 5, stride=1, padding=2),
            nn.BatchNorm2d(self.model_size * 2),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU()
        )

        self.halfres_layer3 = nn.Sequential(
            nn.Conv2d(self.model_size * 2, self.model_size * 4, 9, stride=1, padding=4),
            nn.BatchNorm2d(self.model_size * 4),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU()
        )

        # Layers for quarter resolution image
        self.quarterres_layer1 = nn.Sequential(
            nn.Conv2d(3, self.model_size * 1, 5, stride=1, padding=2),
            nn.BatchNorm2d(self.model_size * 1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU()
        )

        self.quarterres_layer2 = nn.Sequential(
            nn.Conv2d(self.model_size * 1, self.model_size * 2, 5, stride=1, padding=2),
            nn.BatchNorm2d(self.model_size * 2),
            nn.MaxPool2d(2, stride=2 , padding =1), #Adding padding so upsample dimension fit
            nn.ReLU()
        )

        self.quarterres_layer3 = nn.Sequential(
            nn.Conv2d(self.model_size * 2, self.model_size * 4, 9, stride=1, padding=4),
            nn.BatchNorm2d(self.model_size * 4),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU()
        )

        # Last common layers
        self.last_layers = nn.Sequential(
            nn.Conv2d(self.model_size * 4, self.model_size * 4, 9, stride=1, padding=4),
            nn.BatchNorm2d(self.model_size * 4),
            nn.ReLU(),
            nn.Conv2d(self.model_size * 4, self.output_shape[2], 9, stride=1, padding=4)
        )

        self.conv_downsample = nn.Sequential(
            nn.Conv2d(3, 3, 3, stride=2, padding=1),
            nn.Conv2d(3, 3, 1, stride=1, padding=0)
        )

        self.conv_upsample = nn.ConvTranspose2d(self.model_size * 4, self.model_size * 4, 3, stride=2, padding=1)

    def forward(self, inputs):
        fullres = inputs
        #halfres = nn.AvgPool2d(2, stride=2, padding=0)(fullres)
        #quarterres = nn.AvgPool2d(2, stride=2, padding=0)(halfres)
        halfres = self.conv_downsample(fullres)
        quarterres = self.conv_downsample(halfres)

        fullres = self.fullres_layer1(fullres)
        fullres = self.fullres_layer2(fullres)
        fullres = self.fullres_layer3(fullres)

        halfres = self.halfres_layer1(halfres)
        halfres = self.halfres_layer2(halfres)
        halfres = self.halfres_layer3(halfres)
        halfres_size = halfres.size()
        halfres = self.conv_upsample(halfres, output_size=fullres.size())

        quarterres = self.quarterres_layer1(quarterres)
        quarterres = self.quarterres_layer2(quarterres)
        quarterres = self.quarterres_layer3(quarterres)
        quarterres = self.conv_upsample(quarterres, output_size=halfres_size)
        quarterres = nn.Conv2d(self.model_size * 4, self.model_size * 4, 1, stride=1, padding=0)(quarterres)
        quarterres = self.conv_upsample(quarterres, output_size=fullres.size())

        output = fullres + halfres + quarterres
        output /= 3  # Take mean of sum

        output = self.last_layers(output)
        return output

    def train_dataloader(self):
        # Load data and create a DataLoader
        X_train, y_train = load_train_data()
        self.train_dataloader = to_dataloader(X_train, y_train, batch_size=cfg.BATCH_SIZE)
        return self.train_dataloader

    # Same validation data as test data for now
    def val_dataloader(self):
        # Load data and create a DataLoader
        X_val, y_val = load_test_data()
        val_dataloader = to_dataloader(X_val, y_val[:, :, :, 0:self.output_shape[2]], batch_size=cfg.BATCH_SIZE)
        return val_dataloader

    def test_dataloader(self):
        # Load data and create a DataLoader
        X_test, y_test = load_test_data()
        test_dataloader = to_dataloader(X_test, y_test, batch_size=cfg.BATCH_SIZE)
        return test_dataloader

    def configure_optimizers(self):
        # Use Adam optimizer to train model
        optimizer = torch.optim.Adam(self.parameters(), lr=cfg.LEARNING_RATE)
        return optimizer

    def loss(self, preds, targets):
        # TODO: find loss function that works

        # Use Softmax2d?
        #softmax = torch.nn.Softmax2d()
        #preds = softmax(preds)
        #print(f"max preds: {preds.max()} | target max: {targets.max()}")

        # Reshape heatmaps?
        #preds = preds.view(-1, self.output_shape[2], self.output_shape[0]*self.output_shape[1])
        #targets = targets.view(-1, self.output_shape[2], self.output_shape[0]*self.output_shape[1])

        # MULTIPLE TRIES, CAN IGNORE
        #loss = F.nll_loss(preds, targets)
        # pytorch function to replicate tensorflow's tf.nn.softmax_cross_entropy_with_logits
        #loss = torch.sum(- targets * F.log_softmax(preds, -1), -1)
        #preds = F.softmax(preds, dim=2)
        #loss = -torch.sum(targets * torch.log(preds), 1)
        #loss = torch.mean(loss)
        #loss_fn = JointsMSELoss()

        loss_fn = CrossELoss()
        #loss_fn = nn.BCEWithLogitsLoss()
        #loss_fn = nn.BCELoss()
        loss = loss_fn(preds, targets)
        return loss

    def training_step(self, batch, batch_idx):
        # Forward pass of the training and loss computation
        inputs, targets = batch
        preds = self.forward(inputs)
        loss = self.loss(preds, targets)
        logs = {"train_loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, batch, batch_idx):
        # Forward pass of the validation
        images, targets = batch
        preds = self.forward(images)
        loss = self.loss(preds, targets)
        return {'val_loss': loss}

    def validation_end(self, outputs):
        # Log validation loss to tensorboard
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, val_batch, batch_idx):
        # Forward pass of the testing
        inputs, targets = val_batch
        preds = self.forward(inputs)
        loss = self.loss(preds, targets)
        return {'test_loss': loss}

    def test_epoch_end(self, outputs):
        # Log test loss to tensorboard
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss}
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}

    def on_epoch_end(self):
        # Test after each epoch on sample image and save image to output dir
        images, targets = next(iter(self.train_dataloader))
        preds = self.forward(images).detach() * 100

        save_folder = self.logger.log_dir

        for i in range(2):
            image, target, pred = images[i], targets[i], preds[i]
            viz_sample(image.permute(1, 2, 0), target.permute(1, 2, 0), f"epoch_{self.current_epoch}_sample_{i}_train", save_dir=save_folder)
            viz_sample(image.permute(1, 2, 0), pred.permute(1, 2, 0), f"epoch_{self.current_epoch}_sample_{i}_preds", save_dir=save_folder)


if __name__ == "__main__":
    # Train model
    model = PoseDetector()
    trainer = pl.Trainer(max_epochs=cfg.EPOCHS, row_log_interval=1)
    trainer.fit(model)
