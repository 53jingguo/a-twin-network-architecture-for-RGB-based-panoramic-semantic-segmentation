from __future__ import absolute_import, division, print_function
import os

import numpy as np
import time
import json
import tqdm
from collections import Counter

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
torch.manual_seed(100)
torch.cuda.manual_seed(100)
import torch.nn.functional as F

import datasets
from networks import Fuse
from networks import Equi_convnext

class Trainer:
    def __init__(self, settings):
        self.settings = settings
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        self.device = torch.device("cuda")
        # self.gpu_devices = ','.join([str(id) for id in settings.gpu_devices])
        # os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_devices

        self.log_path = os.path.join(self.settings.log_dir, self.settings.model_name)

        # checking the input height and width are multiples of 32
        assert self.settings.height % 32 == 0, "input height must be a multiple of 32"
        assert self.settings.width % 32 == 0, "input width must be a multiple of 32"

        # data
        datasets_dict = {"stanford2d3d": datasets.S2d3dSemgDataset}
        self.dataset = datasets_dict[self.settings.dataset]

        self.train_dataset = self.dataset(self.settings.root, fold='1_train', depth=self.settings.depth, hw=(self.settings.height, self.settings.width), flip=True, rotate=True)
        self.train_loader = DataLoader(self.train_dataset, self.settings.batch_size, True,
                                       num_workers=self.settings.num_workers, pin_memory=True, drop_last=True)
        num_train_samples = len(self.train_dataset)
        self.num_total_steps = num_train_samples // self.settings.batch_size * self.settings.num_epochs

        self.val_dataset = self.dataset(self.settings.root, depth=self.settings.depth, hw=(self.settings.height, self.settings.width), fold='1_valid')
        self.val_loader = DataLoader(self.val_dataset, self.settings.batch_size_test, False,
                                     num_workers=self.settings.num_workers, pin_memory=True, drop_last=False)
        self.invalid_ids = []
        self.label_weight = torch.load('G:/segmentation/networks/label13_weight.pth').float().to(self.device)
        self.label_weight[self.invalid_ids] = 0
        self.label_weight *= (self.settings.num_classes - len(self.invalid_ids)) / self.label_weight.sum()
        self.colors = np.load('G:/Stanford2D3D_sem/colors.npy')
        self.miou_best = 0
        self.marr_best = 0
        # network
        Net_dict = {"fuse512": Fuse}
        dep_Net = Equi_convnext
        Net = Net_dict[self.settings.net]

        self.model = Net(self.settings.height, self.settings.width, invalid_ids=self.invalid_ids,
                         pretrained=self.settings.imagenet_pretrained, num_classes=self.settings.num_classes,
                         )#fusion_type=self.settings.fusion, se_in_fusion=self.settings.se_in_fusion

        self.model.to(self.device)
        self.parameters_to_train = list(self.model.parameters())
        self.optimizer = optim.Adam(self.parameters_to_train, self.settings.learning_rate)

        model_dict = torch.load('G:\dep 4\experiments_1024_f1\panodepth\models\weights_7\model.pth')

        self.depth_model = dep_Net(model_dict['height'], model_dict['width'],
                    max_depth=8)
        self.depth_model.to(self.device)
        model_state_dict = self.depth_model.state_dict()
        self.depth_model.load_state_dict({k: v for k, v in model_dict.items() if k in model_state_dict})
        self.depth_parameters_to_train = list(self.depth_model.parameters())
        self.depth_optimizer = optim.Adam(self.depth_parameters_to_train, self.settings.learning_rate)
        if self.settings.load_weights_dir is not None:
            self.load_model()

        print("Training model named:\n ", self.settings.model_name)
        print("Models and tensorboard events files are saved to:\n", self.settings.log_dir)
        print("Training is using:\n ", self.device)

        self.writers = {}
        for mode in ["train", "test"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))
        self.save_settings()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 1
        self.step = 0
        self.start_time = time.time()
        self.validate()
        for self.epoch in range(1, self.settings.num_epochs+1):
            self.train_one_epoch()
            self.validate()
            if (self.epoch + 1) % self.settings.save_frequency == 0:
                self.save_model_latest()

    def train_one_epoch(self):
        """Run a single epoch of training
        """
        self.model.train()
        self.depth_model.train()
        pbar = tqdm.tqdm(self.train_loader)
        pbar.set_description("Training Epoch_{}".format(self.epoch))
        epoch_losses = Counter()
        for batch_idx, inputs in enumerate(pbar):

            outputs, losses = self.process_batch(inputs)
            if len(losses) == 0:
                continue
            self.optimizer.zero_grad()
            self.depth_optimizer.zero_grad()
            losses["total"].backward()
            self.optimizer.step()
            self.depth_optimizer.step()

            # log less frequently after the first 1000 steps to save time & disk space
            early_phase = batch_idx % self.settings.log_frequency == 0 and self.step < 1000
            late_phase = self.step % 1000 == 0

            if early_phase or late_phase:

                pred = outputs["sem"].detach()
                gt = inputs["sem"]
                mask = (gt >= 0)
                BS = len(inputs['x'])
                for k, v in losses.items():
                    if torch.is_tensor(v):
                        epoch_losses[k] += BS * v.item()
                    else:
                        epoch_losses[k] += BS * v
                self.log("train", inputs, outputs, epoch_losses)
            self.step += 1

    def process_batch(self, inputs):
        for key, ipt in inputs.items():
            if torch.is_tensor(ipt):
                inputs[key] = ipt.to(self.device)

        losses = {}

        rgb = inputs["x"]
        gt_sem = inputs["sem"]
        mask = (gt_sem >= 0).to(self.device)
        if mask.sum() == 0:
            return {},{}
        B, C, H, W = rgb.shape
        dep = self.depth_model(rgb)
        outputs = self.model(rgb, dep['feat4'])

        pred_sem = outputs['sem']
        pred_sem = pred_sem.permute(0, 2, 3, 1)[mask]
        gt = gt_sem[mask]
        losses['acc'] = (pred_sem.argmax(1) == gt).float().mean()
        ce = F.cross_entropy(pred_sem, gt, weight=self.label_weight, reduction='none')
        ce = ce[~torch.isinf(ce) & ~torch.isnan(ce)]
        losses['total.sem'] = ce.mean()
        losses['total'] = sum(v for k, v in losses.items() if k.startswith('total'))
        return outputs, losses

    def validate(self):
        """Validate the model on the validation set
        """
        vis_dir = ''#
        self.model.eval()
        self.depth_model.eval()
        dir = 'G:\\fuse\\experiments\\'
        pbar = tqdm.tqdm(self.val_loader)
        pbar.set_description("testing Epoch_{}".format(self.epoch))
        cm=0
        with torch.no_grad():
            for batch_idx, inputs in enumerate(pbar):
                for key, ipt in inputs.items():
                    if torch.is_tensor(ipt):
                        inputs[key] = ipt.to(self.device)
                rgb = inputs["x"]
                # equi_inputs = equi_inputs.type(torch.cuda.FloatTensor)
                gt_sem = inputs["sem"]
                mask = (gt_sem >= 0).to(self.device)
                if mask.sum() == 0:
                    continue
                B, C, H, W = rgb.shape
                dep = self.depth_model(rgb)
                outputs = self.model(rgb, dep['feat4'])
                pred_sem = outputs["sem"].detach()
                gt = gt_sem[mask]
                pred = pred_sem.argmax(1)[mask]
                if batch_idx % 100 == 0:
                    if vis_dir:
                        import matplotlib.pyplot as plt
                        from imageio import imwrite
                        cmap = (plt.get_cmap('gist_rainbow')(np.arange(self.settings.num_classes) / self.settings.num_classes)[...,:3] * 255).astype(np.uint8)
                        rgb = (inputs['x'][0, :3].permute(1,2,0) * 255).cpu().numpy().astype(np.uint8)
                        imwrite(os.path.join(vis_dir, inputs['fname'][0].strip() + '.rgb.png'), rgb)
                        pre0 = pred_sem[0]
                        vis_sem = cmap[pre0.argmax(0).cpu().numpy()]
                        vis_sem = (rgb * 0.2 + vis_sem * 0.8).astype(np.uint8)
                        imwrite(os.path.join(vis_dir, inputs['fname'][0].strip()), vis_sem)
                        mid = gt_sem[0]
                        vis_sem = cmap[mid.cpu().numpy()]
                        vis_sem = (rgb * 0.2 + vis_sem * 0.8).astype(np.uint8)
                        imwrite(os.path.join(vis_dir, inputs['fname'][0].strip() + '.gt.png'), vis_sem)
                assert gt.min() >= 0 and gt.max() < self.settings.num_classes and pred_sem.shape[1] == self.settings.num_classes
                cm += np.bincount((gt * self.settings.num_classes + pred).cpu().numpy(), minlength=self.settings.num_classes ** 2)
        print('  Summarize  '.center(50, '='))
        cm = cm.reshape(self.settings.num_classes, self.settings.num_classes)
        id2class = np.array(self.val_dataset.ID2CLASS)
        valid_mask = (cm.sum(1) != 0)
        cm = cm[valid_mask][:, valid_mask]
        id2class = id2class[valid_mask]
        inter = np.diag(cm)
        union = cm.sum(0) + cm.sum(1) - inter
        ious = inter / union
        accs = inter / cm.sum(1)
        for name, iou, acc in zip(id2class, ious, accs):
            print(f'{name:20s}:    iou {iou * 100:5.2f}    /    acc {acc * 100:5.2f}')
        print(f'{"Overall":20s}:    iou {ious.mean() * 100:5.2f}    /    acc {accs.mean() * 100:5.2f}')
        np.savez(os.path.join('G:\\fuse\\experiments\\', 'cm.npz'), cm=cm)
        if dir is not None:
            file = os.path.join(dir, "result.txt")
            with open(file, 'a') as f:
                print("\n  " + ("{:>9} | " * 3).format("miou", "macc", "epoch"), file=f)
                print(("&  {: 8.5f} " * 3).format(ious.mean() * 100, accs.mean() * 100, self.epoch), file=f)
        if ious.mean() * 100> self.miou_best:
            self.miou_best = ious.mean() * 100
            self.save_model_best(self.miou_best, accs.mean() * 100)
            print(self.epoch,"miosbest")
        if accs.mean() * 100>self.marr_best:
            self.marr_best = accs.mean()*100
            self.save_model_best_acc(ious.mean() * 100, self.marr_best)
            print(self.epoch,"maccbest")
        del inputs, outputs

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

    def save_settings(self):
        """Save settings to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.settings.__dict__.copy()

        with open(os.path.join(models_dir, 'settings.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        save_path = os.path.join(save_folder, "{}.pth".format("model"))
        to_save = self.model.state_dict()
        to_save['height'] = self.settings.height
        to_save['width'] = self.settings.width
        # save the dataset to train on
        to_save['dataset'] = self.settings.dataset
        to_save['net'] = self.settings.net
        torch.save(to_save, save_path)
        # save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        # torch.save(self.optimizer.state_dict(), save_path)

    def save_model_best(self,miou,macc):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models","best")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        save_path = os.path.join(save_folder, "{}.pth".format("model_iou_best"))
        to_save = self.model.state_dict()
        # save resnet layers - these are needed at prediction time
        to_save['miou'] = miou
        to_save['macc'] = macc
        # save the input sizes
        to_save['height'] = self.settings.height
        to_save['width'] = self.settings.width
        # save the dataset to train on
        to_save['dataset'] = self.settings.dataset
        to_save['net'] = self.settings.net
        torch.save(to_save, save_path)

        depth_save_path = os.path.join(save_folder, "{}.pth".format("depth_model_iou_best"))
        depth_to_save = self.depth_model.state_dict()
        # save resnet layers - these are needed at prediction time
        # depth_to_save['miou'] = miou
        # depth_to_save['macc'] = macc
        # save the input sizes
        depth_to_save['height'] = self.settings.height
        depth_to_save['width'] = self.settings.width
        # save the dataset to train on
        depth_to_save['net'] = 'equi_convnext'
        torch.save(depth_to_save, depth_save_path)
        # save_path = os.path.join(save_folder, "{}.pth".format("adam_iou_best"))
        # torch.save(self.optimizer.state_dict(), save_path)
    def save_model_latest(self,):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models","latest")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        save_path = os.path.join(save_folder, "{}.pth".format("model_latest"))
        to_save = self.model.state_dict()
        # save resnet layers - these are needed at prediction time
        # save the input sizes
        to_save['height'] = self.settings.height
        to_save['width'] = self.settings.width
        # save the dataset to train on
        to_save['dataset'] = self.settings.dataset
        to_save['net'] = self.settings.net
        torch.save(to_save, save_path)
        # save_path = os.path.join(save_folder, "{}.pth".format("adam_latest"))
        # torch.save(self.optimizer.state_dict(), save_path)
        save_path = os.path.join(save_folder, "{}.pth".format("depth_model"))
        to_save = self.depth_model.state_dict()
        # save resnet layers - these are needed at prediction time
        to_save['height'] = self.settings.height
        to_save['width'] = self.settings.width
        to_save['net'] = 'equi_convnext'
        torch.save(to_save, save_path)
    def save_model_best_acc(self,miou,macc):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models","best")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        save_path = os.path.join(save_folder,"{}.pth".format("model_acc_best"))
        to_save = self.model.state_dict()
        # save resnet layers - these are needed at prediction time
        to_save['miou'] = miou
        to_save['macc'] = macc
        # save the input sizes
        to_save['height'] = self.settings.height
        to_save['width'] = self.settings.width
        # save the dataset to train on
        to_save['dataset'] = self.settings.dataset
        to_save['net'] = self.settings.net
        torch.save(to_save, save_path)
        # save_path = os.path.join(save_folder, "{}.pth".format("adam_acc_best"))
        # torch.save(self.optimizer.state_dict(), save_path)
        depth_save_path = os.path.join(save_folder, "{}.pth".format("depth_model_acc_best"))
        depth_to_save = self.depth_model.state_dict()
        depth_to_save['height'] = self.settings.height
        depth_to_save['width'] = self.settings.width
        depth_to_save['net'] = 'equi_convnext'
        torch.save(depth_to_save, depth_save_path)
    def load_model(self):
        """Load model from disk
        """
        self.settings.load_weights_dir = os.path.expanduser(self.settings.load_weights_dir)

        assert os.path.isdir(self.settings.load_weights_dir), \
            "Cannot find folder {}".format(self.settings.load_weights_dir)
        print("loading model from folder {}".format(self.settings.load_weights_dir))

        path = os.path.join(self.settings.load_weights_dir, "{}.pth".format("model"))
        model_dict = self.model.state_dict()
        pretrained_dict = torch.load(path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
        # loading adam state
        # optimizer_load_path = os.path.join(self.settings.load_weights_dir,"{}.pth".format("adam"))
        # if os.path.isfile(optimizer_load_path):
        #     print("Loading Adam weights")
        #     optimizer_dict = torch.load(optimizer_load_path)
        #     self.optimizer.load_state_dict(optimizer_dict)
        # else:
        #     print("Cannot find Adam weights so Adam is randomly initialized")
        path = os.path.join(self.settings.load_weights_dir, "{}.pth".format("depth_model"))
        depth_model_dict = self.depth_model.state_dict()
        pretrained_dict = torch.load(path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in depth_model_dict}
        depth_model_dict.update(pretrained_dict)
        self.depth_model.load_state_dict(depth_model_dict)
    def load_tea(self):
        """Load teacher model from disk
        """
        load_weights_dir = os.path.expanduser("D:\experiments_tea_s2d3d_8\\panodepth\\models\\weights_2\\")
        assert os.path.isdir(load_weights_dir), \
            "Cannot find folder {}".format(load_weights_dir)
        print("loading model from folder {}".format(load_weights_dir))

        path_tea = os.path.join(load_weights_dir, "{}.pth".format("model_tea"))
        model_dict_tea = self.model_tea.state_dict()
        pretrained_dict_tea = torch.load(path_tea)
        pretrained_dict_tea = {k: v for k, v in pretrained_dict_tea.items() if k in model_dict_tea}
        model_dict_tea.update(pretrained_dict_tea)
        self.model_tea.load_state_dict(model_dict_tea)

        optimizer_load_path_tea = os.path.join(load_weights_dir, "{}.pth".format("adam_tea"))
        if os.path.isfile(optimizer_load_path_tea):
            print("Loading Adam weights")
            optimizer_dict_tea = torch.load(optimizer_load_path_tea)
            self.optimizer_tea.load_state_dict(optimizer_dict_tea)
        else:
            print("Cannot find Adam_tea weights so Adam is randomly initialized")
