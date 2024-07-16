from __future__ import absolute_import, division, print_function
import os
import argparse

import numpy as np
import tqdm

import torch
from torch.utils.data import DataLoader
from networks import Fuse, Equi_convnext
import datasets
from metrics import Evaluator
from saver import Saver

parser = argparse.ArgumentParser(description="360 Degree Panorama Depth Estimation Test")

parser.add_argument("--data_path", default="D:\Structured3D\\", type=str, help="path to the dataset.")
parser.add_argument("--dataset", default="struct3d", choices=["stanford2d3d","struct3d"],
                    type=str, help="dataset to evaluate on.")

parser.add_argument("--load_weights_dir",default=r'G:\fuse\experiments\panodepth\models\',
                    type=str, help="folder of model to load")

parser.add_argument("--num_workers", type=int, default=0, help="number of dataloader workers")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--num_classes", type=int, default=40, help="batch size")


parser.add_argument("--median_align", action="store_true", help="if set, apply median alignment in evaluation")
parser.add_argument("--save_samples", default=True, help="if set, save the depth maps and point clouds")

settings = parser.parse_args()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    id2color = np.load("G:\\fuse\\datasets\\colors.npy")
    id2color = id2color[1:]
    load_weights_folder = os.path.expanduser(settings.load_weights_dir)
    model_path = os.path.join(load_weights_folder, "model_iou_best.pth")
    model_dict = torch.load(model_path)

    depth_model_path= os.path.join(load_weights_folder, "depth_model_iou_best.pth")
    depth_model_dict = torch.load(depth_model_path)

    # data
    datasets_dict = {"stanford2d3d": datasets.S2d3dSemgDataset,
                     "struct3d": datasets.Struct3D}
    dataset = datasets_dict[settings.dataset]
    fpath = os.path.join(os.path.dirname(__file__), "datasets", "{}_{}.txt")
    test_file_list = fpath.format(settings.dataset, "test")
    test_dataset = dataset(settings.data_path, test_file_list,
                                        hw=(model_dict['height'], model_dict['width']), )
    test_loader = DataLoader(test_dataset, settings.batch_size, False,
                             num_workers=settings.num_workers, pin_memory=True, drop_last=False)
    num_test_samples = len(test_dataset)
    num_steps = num_test_samples // settings.batch_size
    print("Num. of test samples:", num_test_samples, "Num. of steps:", num_steps, "\n")
    invalid_ids = []
    # network
    Net_dict = {"fuse512": Fuse}
    Net = Net_dict[model_dict['net']]
    dep_Net = Equi_convnext
    depth_model = dep_Net(depth_model_dict['height'], depth_model_dict['width'],max_depth=8)
    depth_model.to(device)
    depth_model_state_dict = depth_model.state_dict()
    depth_model.load_state_dict({k: v for k, v in depth_model_dict.items() if k in depth_model_state_dict})
    depth_model.eval()

    model = Net(equi_h=model_dict['height'],equi_w=model_dict['width'],num_classes=settings.num_classes)
    model.to(device)
    model_state_dict = model.state_dict()
    model.load_state_dict({k: v for k, v in model_dict.items() if k in model_state_dict})
    model.eval()
    print(model_dict['miou'],model_dict['macc'])
    evaluator = Evaluator(settings.median_align)
    evaluator.reset_eval_metrics()
    saver = Saver(load_weights_folder)
    pbar = tqdm.tqdm(test_loader)
    pbar.set_description("Testing")
    cm = 0
    vis_dir = 'G:\\fuse\\vis_struct3d\\'
    with torch.no_grad():
        for batch_idx, inputs in enumerate(pbar):
            rgb = inputs["normalized_rgb"].to(device)
            gt_sem = inputs["sem"].to(device)
            mask = ((gt_sem >= 0) & (gt_sem < settings.num_classes)).to(device)
            if mask.sum() == 0:
                continue
            dep = depth_model(rgb)
            outputs = model(rgb, dep['feat4'])
            pred_sem = outputs["sem"].detach()
            gt = gt_sem[mask]
            pred = pred_sem.argmax(1)[mask]
            if vis_dir:
                import matplotlib.pyplot as plt
                from imageio import imwrite
                rgb = (inputs['rgb'][0, :3].permute(1, 2, 0) * 255).cpu().numpy().astype(np.uint8)
                imwrite(os.path.join(vis_dir, str(inputs['sem_path'][0]) + str(inputs['sem_path'][1]) + str(inputs['sem_path'][2]) + '.rgb.png'), rgb)
                vis_sem = id2color[pred_sem[0].argmax(0).cpu().numpy()]
                vis_sem = (vis_sem).astype(np.uint8)
                imwrite(os.path.join(vis_dir, str(inputs['sem_path'][0]) + str(inputs['sem_path'][1]) + str(inputs['sem_path'][2])+ '.pred.png') , vis_sem)
                gt_sem = gt_sem[0].cpu().numpy()
                gt_sem[gt_sem==255] = 40
                new_row = np.array([0, 0, 0])
                id2color = np.vstack((id2color, new_row))
                vis_sem = id2color[gt_sem]
                vis_sem = (vis_sem).astype(np.uint8)
                imwrite(os.path.join(vis_dir, str(inputs['sem_path'][0]) + str(inputs['sem_path'][1]) + str(inputs['sem_path'][2])  + '.gt.png'), vis_sem)

            assert gt.min() >= 0 and gt.max() < settings.num_classes and pred_sem.shape[
                1] == settings.num_classes
            cm += np.bincount((gt * settings.num_classes + pred).cpu().numpy(),
                              minlength=settings.num_classes ** 2)
        print('  Summarize  '.center(50, '='))
        cm = cm.reshape(settings.num_classes, settings.num_classes)
        id2class = np.array(test_dataset.ID2CLASS)
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


if __name__ == "__main__":
    main()