import pytorch_lightning as pl
from torch import nn
import torch
import torchvision
from torch.nn import functional as F
from torch.autograd import Variable
from data import load_train_data, load_test_data, to_dataloader, viz_sample
import config as cfg
import utils
import numpy as np


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

        return loss #/ num_joints


class PoseDetector(pl.LightningModule):

    def __init__(self, use_spatial_model=True, gpu_cuda = True):
        super(PoseDetector, self).__init__()

        self.model_size = cfg.MODEL_SIZE
        self.output_shape = (60, 90, 10)
        self.use_spatial_model = use_spatial_model
        self.gpu_cuda = gpu_cuda

        # Model joints:
        self.joint_names = ['lsho', 'lelb', 'lwri', 'rsho', 'relb', 'rwri', 'lhip', 'rhip', 'nose', 'torso']
        self.joint_dependence = {}
        ## Assuming there is co-dependence between EVERY joint pairs
        for joint in self.joint_names:
            self.joint_dependence[joint] = [joint_cond for joint_cond in self.joint_names if joint_cond != joint]

        ## Initializing pairwise energies and bias between Joints
        self.pairwise_energies, self.pairwise_biases = {}, {}
        for joint in self.joint_names:#[:n_joints]:
            for cond_joint in self.joint_dependence[joint]:
                #TODO : manage dynamic sizing (in-place of 120,180)
                ## TODO : Check if need to be manually placed on device
                joint_key = joint + '_' + cond_joint
                if self.gpu_cuda:
                    self.pairwise_energies[joint_key] = torch.ones([1,119,179,1], dtype=torch.float32, requires_grad=True, device="cuda")/(119*179)
                    self.pairwise_biases[joint_key] = torch.ones([1,60,90,1], dtype=torch.float32, requires_grad=True, device="cuda")/(60*90)
                else:
                    self.pairwise_energies[joint_key] = torch.ones([1,119,179,1], dtype=torch.float32, requires_grad=True)/(119*179)
                    self.pairwise_biases[joint_key] = torch.ones([1,60,90,1], dtype=torch.float32, requires_grad=True)/(60*90)


        # Layers for full resolution image
        self.fullres_layer1 = nn.Sequential(
            nn.Conv2d(3, self.model_size * 1, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(self.model_size * 1),
            nn.MaxPool2d(2, stride=2)
            #nn.ReLU()
        )

        self.fullres_layer2 = nn.Sequential(
            nn.Conv2d(self.model_size * 1, self.model_size * 2, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(self.model_size * 2),
            nn.MaxPool2d(2, stride=2)
            #nn.ReLU()
        )

        self.fullres_layer3 = nn.Sequential(
            nn.Conv2d(self.model_size * 2, self.model_size * 4, 9, stride=1, padding=4),
            nn.ReLU(),
            nn.BatchNorm2d(self.model_size * 4),
            nn.MaxPool2d(2, stride=2)
            #nn.ReLU()
        )

        # Layers for half resolution image
        self.halfres_layer1 = nn.Sequential(
            nn.Conv2d(3, self.model_size * 1, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(self.model_size * 1),
            nn.MaxPool2d(2, stride=2)
            #nn.ReLU()
        )

        self.halfres_layer2 = nn.Sequential(
            nn.Conv2d(self.model_size * 1, self.model_size * 2, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(self.model_size * 2),
            nn.MaxPool2d(2, stride=2)
            #nn.ReLU()
        )

        self.halfres_layer3 = nn.Sequential(
            nn.Conv2d(self.model_size * 2, self.model_size * 4, 9, stride=1, padding=4),
            nn.ReLU(),
            nn.BatchNorm2d(self.model_size * 4),
            nn.MaxPool2d(2, stride=2)
            #nn.ReLU()
        )

        # Layers for quarter resolution image
        self.quarterres_layer1 = nn.Sequential(
            nn.Conv2d(3, self.model_size * 1, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(self.model_size * 1),
            nn.MaxPool2d(2, stride=2)
            #nn.ReLU()
        )

        self.quarterres_layer2 = nn.Sequential(
            nn.Conv2d(self.model_size * 1, self.model_size * 2, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(self.model_size * 2),
            nn.MaxPool2d(2, stride=2 , padding =1) #Adding padding so upsample dimension fit
            #nn.ReLU()
        )

        self.quarterres_layer3 = nn.Sequential(
            nn.Conv2d(self.model_size * 2, self.model_size * 4, 9, stride=1, padding=4),
            nn.ReLU(),
            nn.BatchNorm2d(self.model_size * 4),
            nn.MaxPool2d(2, stride=2)
            #nn.ReLU()
        )

        # Last common layers
        self.last_layers = nn.Sequential(
            nn.Conv2d(self.model_size * 4, self.model_size * 4, 9, stride=1, padding=4),
            nn.ReLU(),
            nn.BatchNorm2d(self.model_size * 4),
            nn.Conv2d(self.model_size * 4, self.output_shape[2], 9, stride=1, padding=4)
        )

        ## Upsampling and downsampling

        self.conv_downsample = nn.Sequential(
            nn.Conv2d(3, 3, 3, stride=2, padding=1),
            nn.Conv2d(3, 3, 1, stride=1, padding=0)
        )

        self.conv_upsample = nn.ConvTranspose2d(self.model_size * 4, self.model_size * 4, 3, stride=2, padding=1)

        self.conv1_1 = nn.Conv2d(self.model_size * 4, self.model_size * 4, 1, stride=1, padding=0)

        ## Softplus for spatial model
        self.softplus = nn.Softplus(beta=5)

        ## Batchnorm for spatial model
        self.BN_SM = nn.BatchNorm2d(self.output_shape[2])

    def conv_marginal_like(self, prior, likelihood):
        likelihood_shape = likelihood.shape
        prior = prior.permute(0,3,1,2)
        likelihood = likelihood.permute(0,3,1,2)
        likelihood = torch.flip(likelihood, dims=[2,3])

        marginal = F.conv2d(prior, likelihood)
        marginal = marginal.permute(1,2,3,0)
        assert marginal.shape == likelihood_shape
        return marginal

    ## Spatial model
    def spatial_model(self, part_detector_pred):
        hm_logit = torch.stack([F.log_softmax(part_detector_pred[i,:,:,:], dim=1) for i in range(part_detector_pred.shape[0])])
        hm_logit = self.BN_SM(hm_logit)
        hm_logit = hm_logit.permute(0,2,3,1)

        heat_map_hat = []
        for joint_id, joint_name in enumerate(self.joint_names):
            hm = hm_logit[:, :, :, joint_id:joint_id + 1]
            marginal_energy = torch.log(self.softplus(hm) + 1e-6)  #1e-6: numerical stability
            for cond_joint in self.joint_dependence[joint_name]:
                cond_joint_id = np.where(np.array(self.joint_names) == np.array(cond_joint))[0][0]
                prior = self.softplus(self.pairwise_energies[joint_name + '_' + cond_joint])
                likelihood = self.softplus(hm_logit[:, :, :, cond_joint_id:cond_joint_id + 1])
                bias = self.softplus(self.pairwise_biases[joint_name + '_' + cond_joint])
                marginal_energy += torch.log(self.conv_marginal_like(prior, likelihood) + bias + 1e-6)
            heat_map_hat.append(marginal_energy)
        return torch.stack(heat_map_hat, dim=3)[:,:,:,:,0].permute(0,3,1,2)

    def forward(self, inputs):
        ## We need to reattach previous computation of pairwise variables to the new graph
        ## This is needed because the previous graph is discarded between each batches
        for joint in self.joint_names:#[:n_joints]:
            for cond_joint in self.joint_dependence[joint]:
                ## TODO : Check if need to be manually placed on device
                joint_key = joint + '_' + cond_joint
                self.pairwise_energies[joint_key].detach_().requires_grad_(True)
                self.pairwise_biases[joint_key].detach_().requires_grad_(True)


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
        quarterres = self.conv1_1(quarterres)
        quarterres = self.conv_upsample(quarterres, output_size=fullres.size())

        output = fullres + halfres + quarterres
        output /= 3  # Take mean of sum

        output = self.last_layers(output)

        if self.use_spatial_model:
            output_sm = self.spatial_model(output)

        return output, output_sm

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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)
        return [optimizer], [scheduler]
        #return optimizer

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
        loss_cnn = loss_fn(preds[0], targets)
        if self.use_spatial_model:
            loss_sm = loss_fn(preds[1], targets)
        else :
            loss_sm = 0
        loss = loss_cnn + loss_sm
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
        if self.gpu_cuda:
            images = images.to("cuda")
        preds = self.forward(images)[1].detach()
        preds = torch.stack([F.softmax(preds[i,:,:,:], dim=1) for i in range(preds.shape[0])]) * 100
        if self.gpu_cuda:
            preds = preds.cpu()
            images = images.cpu()

        save_folder = self.logger.log_dir

        for i in range(2):
            image, target, pred = images[i], targets[i], preds[i]
            viz_sample(image.permute(1, 2, 0), target.permute(1, 2, 0), f"epoch_{self.current_epoch}_sample_{i}_train", save_dir=save_folder)
            viz_sample(image.permute(1, 2, 0), pred.permute(1, 2, 0), f"epoch_{self.current_epoch}_sample_{i}_preds", save_dir=save_folder)


if __name__ == "__main__":
    # Train model
    model = PoseDetector(gpu_cuda = True)
    if model.gpu_cuda:
        trainer = pl.Trainer(max_epochs=cfg.EPOCHS, row_log_interval=1, gpus=1)
    else:
        trainer = pl.Trainer(max_epochs=cfg.EPOCHS, row_log_interval=1)
    trainer.fit(model)
