import argparse
import utils
import random
import numpy as np
import torch
from torchvision import transforms, datasets
import augment
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, SubsetRandomSampler
import models
import copy
import os
import json
from glob import glob
from torch.utils.data import Dataset
from PIL import Image
import tifffile
import torchsat.transforms.transforms_cls as transforms_sat


NETWORKS = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "resnet152": models.resnet152,
    "fresnet18": models.resnet18,
    "fresnet34": models.resnet34,
    "fresnet50": models.resnet50,
    "fresnet101": models.resnet101,
    "fresnet152": models.resnet152,
    "hresnet18": models.resnet18,
    "hresnet34": models.resnet34,
    "hresnet50": models.resnet50,
    "hresnet101": models.resnet101,
    "hresnet152": models.resnet152,
    "vgg11": models.vgg11,
    "vgg13": models.vgg13,
    "vgg16": models.vgg16,
    "vgg19": models.vgg19,
    "mobilenet": models.MobileNetV2,
    "vit": models.VisionTransformer,
}


def long_tailed_dist(data_len, cls_num, imb_type="exp", imb_factor=0.01):
    img_max = data_len / cls_num
    img_num_per_cls = []
    if imb_type == "exp":
        for cls_idx in range(cls_num):
            num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
            img_num_per_cls.append(int(num))
    elif imb_type == "step":
        for cls_idx in range(cls_num // 2):
            img_num_per_cls.append(int(img_max))
        for cls_idx in range(cls_num // 2):
            img_num_per_cls.append(int(img_max * imb_factor))
    else:
        img_num_per_cls.extend([int(img_max)] * cls_num)
    return img_num_per_cls


class FewShot(Dataset):
    def __init__(self, root, split="trainval", transform=None):
        root_imgs = glob(f"{root}/imgs/**/*", recursive=True)
        split_f = open(os.path.join(root, f"{split}.txt"), "r")
        self.data = []
        for line in split_f:
            img_f, cls = line.split(" ")
            img_f = [root_img for root_img in root_imgs if img_f in root_img][0]
            self.data.append((img_f, int(cls)))
        self.transform = transform

    def __getitem__(self, indx):
        img_f, tgt = self.data[indx]
        if ".tif" in img_f:
            img = tifffile.imread(img_f)
            if len(img.shape) == 2:
                img = img[:, :, None]
        else:
            img = Image.open(img_f).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, tgt

    def __len__(self):
        return len(self.data)


class Trainer:
    def __init__(self, args):
        self.args = args
        self.out_dir = args.out_dir

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        self.device, local_rank = utils.setup_device(args.dist)
        if args.dist:
            self.main_thread = True if local_rank == 0 else False
        else:
            self.main_thread = True
        if self.main_thread:
            print(f"\nsetting up device, distributed = {args.dist}")
        print(f" | {self.device}")

        if "cifar" in args.dset:
            if args.pretrained and "vit" in args.net:
                t = [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                ]
            else:
                t = [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                ]
        elif args.dset == "eurosat_rgb":
            t = [
                transforms.RandomCrop(64, padding=8),
                transforms.RandomHorizontalFlip(),
            ]
        elif args.dset == "eurosat_allband":
            t = [
                transforms_sat.Pad(padding=8),
                transforms_sat.RandomCrop(64),
                transforms_sat.RandomHorizontalFlip(),
            ]
        elif args.dset == "clamm":
            t = [
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
            ]
        elif args.dset in ["cub", "isic", "imagenet"]:
            t = [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
            ]
        else:
            raise NotImplementedError(f"args.dset = {args.dset} not implemented.")
        if args.dset in ["eurosat_rgb", "isic"]:
            t.append(transforms.RandomVerticalFlip())
        elif args.dset == "eurosat_allband":
            t.append(transforms_sat.RandomVerticalFlip())
        if args.contrast_aug:
            t.extend(
                [
                    transforms.RandomApply(
                        [
                            transforms.ColorJitter(
                                0.8 * args.color_jitter_strength,
                                0.8 * args.color_jitter_strength,
                                0.8 * args.color_jitter_strength,
                                0.2 * args.color_jitter_strength,
                            )
                        ],
                        p=args.color_jitter_prob,
                    ),
                    transforms.RandomGrayscale(p=args.gray_prob),
                ]
            )
        if args.rand_aug:
            t.extend(
                [
                    augment.RandomAugment(args.n_rand_aug),
                ]
            )
        if args.auto_aug:
            t.extend(
                [
                    augment.Policy(policy=args.auto_aug_policy),
                ]
            )
        if args.custom_aug:
            t.extend(
                [
                    augment.ToNumpy(),
                    augment.CustomAugment.augment_image,
                    transforms.ToPILImage(),
                ]
            )
        if args.blur:
            t.extend(
                [
                    transforms.RandomApply(
                        [augment.GaussianBlur(args.blur_sigma)], p=args.blur_prob
                    ),
                ]
            )
        if args.cutout:
            t.extend(
                [
                    augment.Cutout(cut_len=args.cut_len),
                ]
            )
        if args.dset == "cifar10":
            normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        elif args.dset == "cifar100":
            normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        elif args.dset in ["cub", "eurosat_rgb", "isic", "imagenet"]:
            normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        elif args.dset == "eurosat_allband":
            normalize = transforms_sat.Normalize(mean=(0.5,) * 13, std=(0.5,) * 13)
        elif args.dset == "clamm":
            normalize = transforms.Normalize(mean=(0.5), std=(0.5))
        else:
            raise NotImplementedError(f"args.dset = {args.dset} not implemented.")
        if args.dset == "eurosat_allband":
            train_transform = transforms.Compose(
                [
                    *t,
                    transforms_sat.ToTensor(),
                    normalize,
                ]
            )
        else:
            train_transform = transforms.Compose(
                [
                    *t,
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        if args.dset in ["cifar10", "cifar100", "eurosat_rgb"]:
            if args.pretrained and "vit" in args.net:
                val_transform = transforms.Compose(
                    [
                        transforms.Resize(224),
                        transforms.ToTensor(),
                        normalize,
                    ]
                )
            else:
                val_transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        normalize,
                    ]
                )
        elif args.dset == "eurosat_allband":
            val_transform = transforms.Compose(
                [
                    transforms_sat.ToTensor(),
                    normalize,
                ]
            )
        elif args.dset == "clamm":
            val_transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        elif args.dset in ["cub", "isic", "imagenet"]:
            val_transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        if args.dset == "cifar10":
            train_dset = datasets.CIFAR10(
                root=args.data_root, train=True, transform=train_transform, download=True
            )
            val_dset = datasets.CIFAR10(
                root=args.data_root, train=False, transform=val_transform, download=True
            )
        elif args.dset == "cifar100":
            train_dset = datasets.CIFAR100(
                root=args.data_root, train=True, transform=train_transform, download=True
            )
            val_dset = datasets.CIFAR100(
                root=args.data_root, train=False, transform=val_transform, download=True
            )
        elif args.dset in ["eurosat_rgb", "eurosat_allband", "isic", "clamm", "cub"]:
            train_dset = FewShot(root=args.data_root, split="trainval", transform=train_transform)
            val_dset = FewShot(root=args.data_root, split="test", transform=val_transform)
        elif args.dset == "imagenet":
            train_dset = datasets.ImageFolder(
                root=os.path.join(args.data_root, "train"), transform=train_transform
            )
            val_dset = datasets.ImageFolder(
                root=os.path.join(args.data_root, "val"), transform=train_transform
            )
        else:
            raise NotImplementedError(f"args.dset = {args.dset} not implemented.")
        if args.long_tailed:
            targets = train_dset.targets
            img_num_per_cls = long_tailed_dist(
                len(train_dset),
                len(np.unique(targets)),
                args.long_tailed_type,
                args.long_tailed_factor,
            )
            all_indx = np.arange(len(train_dset))
            data_indx = []
            for cls_id in np.unique(targets):
                indx = all_indx[targets == cls_id]
                data_indx.extend(indx[: img_num_per_cls[cls_id]])
            data_indx = np.array(data_indx)
        else:
            if 0 < args.data_size <= 1:
                data_size = args.data_size
            elif int(args.data_size) == args.data_size:
                data_size = args.data_size / len(train_dset)
            else:
                raise ValueError(
                    f"args.data_size = {args.data_size}, must be float between 0-1 or int > 1"
                )
            if data_size == 1:
                data_indx = np.arange(len(train_dset))
            else:
                _, data_indx = train_test_split(
                    np.arange(len(train_dset)),
                    test_size=data_size,
                    shuffle=True,
                    stratify=train_dset.targets,
                )
        if self.main_thread:
            print(f"setting up dataset, train: {len(data_indx)}, val: {len(val_dset)}")
        train_sampler = SubsetRandomSampler(data_indx)
        self.train_loader = DataLoader(
            train_dset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.n_workers,
        )
        self.val_loader = DataLoader(
            val_dset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers
        )
        self.criterion = torch.nn.CrossEntropyLoss()
        if args.cos_criterion:
            self.cos_criterion = torch.nn.CosineEmbeddingLoss()
        self.metric_meter = utils.AvgMeter()

    def train_epoch(self):
        self.metric_meter.reset()
        self.model.train()
        for indx, (img, target) in enumerate(self.train_loader):
            img, target = img.to(self.device).float(), target.to(self.device)

            if self.args.adv_prop:
                pred, adv_pred = self.model(img, target, adv_prop=True)
                loss = (self.criterion(pred, target) + self.criterion(adv_pred, target)) / 2
            else:
                if self.args.cos_linear == True or self.args.tvmf_linear == True:
                    pred, loss = self.model(img, target)
                else:
                    pred = self.model(img)
                    loss = self.criterion(pred, target)
                    if self.args.cos_criterion:
                        one_hot = torch.nn.functional.one_hot(target, num_classes=pred.shape[-1])
                        loss = 0.1 * loss + self.cos_criterion(
                            pred, one_hot, torch.ones(target.shape).to(target.device)
                        )

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            pred_cls = pred.argmax(dim=1)
            acc = pred_cls.eq(target.view_as(pred_cls)).sum().item() / img.shape[0]

            metrics = {"train loss": loss.item(), "train acc": acc}
            self.metric_meter.add(metrics)
            utils.pbar(indx / len(self.train_loader), msg=self.metric_meter.msg())
        utils.pbar(1, msg=self.metric_meter.msg())

    @torch.no_grad()
    def eval(self):
        self.metric_meter.reset()
        self.model.eval()
        if self.args.adv_prop:
            self.model.apply(utils.to_clean)
        for indx, (img, target) in enumerate(self.val_loader):
            img, target = img.to(self.device).float(), target.to(self.device)

            if self.args.cos_linear == True or self.args.tvmf_linear == True:
                pred, loss = self.model(img, target)
            else:
                pred = self.model(img)
                loss = self.criterion(pred, target)

            pred_cls = pred.argmax(dim=1)
            acc = pred_cls.eq(target.view_as(pred_cls)).sum().item() / img.shape[0]

            metrics = {"val loss": loss.item(), "val acc": acc}
            self.metric_meter.add(metrics)
            utils.pbar(indx / len(self.val_loader), msg=self.metric_meter.msg())
        utils.pbar(1, msg=self.metric_meter.msg())

    def train_imp(self):
        if "vit" in self.args.net:
            if self.args.dset == "cifar10":
                if self.args.pretrained:
                    model = NETWORKS[self.args.net](
                        img_size=224, patch_size=32, pretrained=True, num_classes=10
                    )
                else:
                    model = NETWORKS[self.args.net](
                        img_size=32, patch_size=8, pretrained=False, num_classes=10
                    )
            else:
                raise NotImplementedError(f"args.dset = {self.args.dset} is not implemented")
        else:
            if self.args.dset == "cifar10":
                model = NETWORKS[self.args.net](
                    n_cls=10, pre_conv="small", pretrained=args.pretrained
                )
            elif self.args.dset == "cifar100":
                model = NETWORKS[self.args.net](
                    n_cls=100, pre_conv="small", pretrained=args.pretrained
                )
            elif self.args.dset == "cub":
                model = NETWORKS[self.args.net](
                    n_cls=200, pre_conv="full", pretrained=args.pretrained
                )
            elif self.args.dset == "eurosat_rgb":
                model = NETWORKS[self.args.net](
                    n_cls=10, pre_conv="small", pretrained=args.pretrained
                )
            elif self.args.dset == "eurosat_allband":
                model = NETWORKS[self.args.net](
                    n_cls=10, pre_conv="small", pretrained=args.pretrained, in_dim=13
                )
            elif self.args.dset == "isic":
                model = NETWORKS[self.args.net](
                    n_cls=7, pre_conv="full", pretrained=args.pretrained
                )
            elif self.args.dset == "clamm":
                model = NETWORKS[self.args.net](
                    n_cls=12, pre_conv="full", pretrained=args.pretrained, in_dim=1
                )
            elif self.args.dset == "imagenet":
                model = NETWORKS[self.args.net](
                    n_cls=1000, pre_conv="full", pretrained=args.pretrained
                )
            else:
                raise NotImplementedError(f"args.dset = {args.dset} not implemented.")

        if self.args.cos_linear:
            model.linear = resnet.Cosine(
                model.linear.weight.shape[1], model.linear.weight.shape[0]
            )
        if self.args.tvmf_linear:
            model.linear = resnet.TVMF(model.linear.weight.shape[1], model.linear.weight.shape[0])
        if self.args.adv_prop:
            utils.modify_bn(model)
            setattr(
                model,
                "attacker",
                utils.PGDAttacker(
                    self.args.attack_n_iter, self.args.attack_eps, self.args.attack_step_size, 0.2
                ),
            )
        self.model = model.to(self.device)

        self.optim = utils.setup_optim(self.args, self.model.parameters())
        self.optim, self.lr_sched = utils.setup_lr_sched(self.args, self.optim)

        if os.path.exists(os.path.join(self.args.out_dir, "last_imp.ckpt")):
            if self.args.resume == False:
                raise ValueError(
                    f"directory {self.args.out_dir} already exists, change output directory or use --resume argument"
                )
            ckpt = torch.load(
                os.path.join(self.args.out_dir, "last_imp.ckpt"), map_location=self.device
            )
            self.model.load_state_dict(ckpt["model"])
            self.optim.load_state_dict(ckpt["optim"])
            self.lr_sched.load_state_dict(ckpt["lr_sched"])
            init = ckpt["init"]
            prune_iter_start = ckpt["iter"]
            start_epoch = ckpt["epoch"]
            print(f"\nresuming imp training from iter = {prune_iter_start}, epoch = {start_epoch}")
        else:
            if self.args.resume == True:
                raise ValueError(
                    f"args.resume = true, but no checkpoint found in {self.args.out_dir}"
                )
            os.makedirs(self.args.out_dir, exist_ok=True)
            with open(os.path.join(self.args.out_dir, "args_imp.txt"), "w") as f:
                json.dump(self.args.__dict__, f, indent=4)
            prune_iter_start = 0
            start_epoch = 0
            init = copy.deepcopy(self.model.state_dict())
            print(f"\nstarting imp training from scratch")

        for iter in range(prune_iter_start, self.args.pruning_iters):
            if self.main_thread:
                print(f"pruning state: {iter}")
                if "vit" in self.args.net:
                    print(
                        "remaining weight = ",
                        utils.check_sparsity_vit(self.model, self.args.prune_ff_only),
                    )
                else:
                    print("remaining weight = ", utils.check_sparsity(self.model))
                print("----------------------")

            best_train, best_val = 0, 0
            for epoch in range(start_epoch, self.args.epochs):
                if self.main_thread:
                    print(
                        f"epoch: {epoch}, best train: {round(best_train, 5)}, best val: {round(best_val, 5)}, lr: {round(self.optim.param_groups[0]['lr'], 5)}"
                    )
                    print("---------------")

                self.train_epoch()
                if (
                    iter == 0
                    and self.args.rewind_type == "epoch"
                    and epoch == self.args.rewind_epoch
                ):
                    init = copy.deepcopy(self.model.state_dict())

                if self.main_thread:
                    train_metrics = self.metric_meter.get()
                    self.eval()
                    val_metrics = self.metric_meter.get()

                    if train_metrics["train acc"] > best_train:
                        print(
                            "\x1b[34m"
                            + f"train acc improved from {round(best_train, 5)} to {round(train_metrics['train acc'], 5)}"
                            + "\033[0m"
                        )
                        best_train = train_metrics["train acc"]

                    if val_metrics["val acc"] > best_val:
                        print(
                            "\x1b[33m"
                            + f"val acc improved from {round(best_val, 5)} to {round(val_metrics['val acc'], 5)}"
                            + "\033[0m"
                        )
                        best_val = val_metrics["val acc"]
                        torch.save(
                            {"model": self.model.state_dict(), "init": init},
                            os.path.join(self.args.out_dir, f"best_imp_{iter}.ckpt"),
                        )

                    torch.save(
                        {
                            "model": self.model.state_dict(),
                            "optim": self.optim.state_dict(),
                            "lr_sched": self.lr_sched.state_dict(),
                            "init": init,
                            "iter": iter,
                            "epoch": epoch,
                        },
                        os.path.join(self.args.out_dir, "last_imp.ckpt"),
                    )
                    torch.save(
                        {"model": self.model.state_dict(), "init": init},
                        os.path.join(self.args.out_dir, f"last_imp_{iter}.ckpt"),
                    )

                if epoch < self.args.warmup_epochs:
                    self.optim.param_groups[0]["lr"] = (
                        epoch / self.args.warmup_epochs * self.args.lr
                    )
                else:
                    self.lr_sched.step()

            if iter == 0 and self.args.rewind_type == "pt":
                init = torch.load(
                    os.path.join(self.args.out_dir, f"best_imp_{iter}.ckpt"),
                    map_location=self.device,
                )["model"]

            if self.args.prune_type == "random":
                utils.random_prune(self.model, self.args.prune_rate)
            elif self.args.prune_type == "l1":
                if "vit" in self.args.net:
                    utils.l1_prune_vit(self.model, self.args.prune_rate, self.args.prune_ff_only)
                else:
                    utils.l1_prune(self.model, self.args.prune_rate)
            else:
                raise NotImplementedError(
                    f"args.prune_type = {self.args.prune_type} is not implemented"
                )

            curr_mask = utils.extract_mask(self.model.state_dict())
            if "vit" in self.args.net:
                utils.remove_prune_vit(self.model, self.args.prune_ff_only)
            else:
                utils.remove_prune(self.model)
            self.model.load_state_dict(init)
            if "vit" in self.args.net:
                utils.mask_prune_vit(self.model, curr_mask, self.args.prune_ff_only)
            else:
                utils.mask_prune(self.model, curr_mask)
            self.optim = utils.setup_optim(self.args, self.model.parameters())
            self.optim, self.lr_sched = utils.setup_lr_sched(self.args, self.optim)
            start_epoch = 0
            if self.args.rewind_type:
                for _ in range(self.args.rewind_epoch):
                    self.lr_sched.step()

    def train(self):
        if self.args.dset == "cifar10":
            model = NETWORKS[self.args.net](n_cls=10, pre_conv="small", pretrained=args.pretrained)
        elif self.args.dset == "cifar100":
            model = NETWORKS[self.args.net](
                n_cls=100, pre_conv="small", pretrained=args.pretrained
            )
        elif self.args.dset == "cub":
            model = NETWORKS[self.args.net](n_cls=200, pre_conv="full", pretrained=args.pretrained)
        elif self.args.dset == "eurosat_rgb":
            model = NETWORKS[self.args.net](n_cls=10, pre_conv="small", pretrained=args.pretrained)
        elif self.args.dset == "eurosat_allband":
            model = NETWORKS[self.args.net](
                n_cls=10, pre_conv="small", pretrained=args.pretrained, in_dim=13
            )
        elif self.args.dset == "isic":
            model = NETWORKS[self.args.net](n_cls=7, pre_conv="full", pretrained=args.pretrained)
        elif self.args.dset == "clamm":
            model = NETWORKS[self.args.net](
                n_cls=12, pre_conv="full", pretrained=args.pretrained, in_dim=1
            )
        elif self.args.dset == "imagenet":
            model = NETWORKS[self.args.net](
                n_cls=1000, pre_conv="full", pretrained=args.pretrained
            )
        else:
            raise NotImplementedError(f"args.dset = {args.dset} not implemented.")

        if self.args.cos_linear:
            model.linear = resnet.Cosine(
                model.linear.weight.shape[1], model.linear.weight.shape[0]
            )
        if self.args.tvmf_linear:
            model.linear = resnet.TVMF(model.linear.weight.shape[1], model.linear.weight.shape[0])
        if self.args.adv_prop:
            utils.modify_bn(model)
            setattr(
                model,
                "attacker",
                utils.PGDAttacker(
                    self.args.attack_n_iter, self.args.attack_eps, self.args.attack_step_size, 0.2
                ),
            )
        self.model = model.to(self.device)

        self.optim = utils.setup_optim(self.args, self.model.parameters())
        self.optim, self.lr_sched = utils.setup_lr_sched(self.args, self.optim)

        if self.args.load_ticket:
            if not os.path.exists(self.args.load_ticket):
                raise ValueError(
                    f"args.load_ticket = {self.args.load_ticket}, but no ticket found in {self.args.out_dir}"
                )
            ckpt = torch.load(self.args.load_ticket, map_location=self.device)
            # # load weights except linear
            # init = {key: param for key, param in ckpt["init"].items() if "linear" not in key}
            # self.model.load_state_dict(init, strict=False)
            self.model.load_state_dict(ckpt["init"])
            if int(os.path.basename(self.args.load_ticket).split(".")[0].split("_")[-1]):
                curr_mask = utils.extract_mask(ckpt["model"])
                # # modify mask - random layer sparse
                # new_mask = {}
                # for name, mask in curr_mask.items():
                #     active = mask[mask == 1].shape[0]
                #     total = mask.nelement()
                #     temp = np.random.choice(
                #         [0, 1], size=mask.shape, p=[1 - active / total, active / total]
                #     )
                #     new_mask[name] = torch.tensor(temp).to(self.device)
                # utils.mask_prune(self.model, new_mask)
                utils.mask_prune(self.model, curr_mask)
                print(f"loaded ticket from {self.args.load_ticket}")
                print("remaining weight = ", utils.check_sparsity(self.model))

        # # freeze backbone, train linear classifier
        # for param in self.model.parameters():
        #     param.requires_grad = False
        # for param in self.model.linear.parameters():
        #     param.requires_grad = True

        if os.path.exists(os.path.join(self.args.out_dir, "last.ckpt")):
            if self.args.resume == False:
                raise ValueError(
                    f"directory {self.args.out_dir} already exists, change output directory or use --resume argument"
                )
            ckpt = torch.load(
                os.path.join(self.args.out_dir, "last.ckpt"), map_location=self.device
            )
            self.model.load_state_dict(ckpt["model"])
            self.optim.load_state_dict(ckpt["optim"])
            self.lr_sched.load_state_dict(ckpt["lr_sched"])
            start_epoch = ckpt["epoch"]
            print(f"\nresuming training from epoch = {start_epoch}")
        else:
            if self.args.resume == True:
                raise ValueError(
                    f"args.resume == true, but no checkpoint found in {self.args.out_dir}"
                )
            os.makedirs(self.args.out_dir, exist_ok=True)
            with open(os.path.join(self.args.out_dir, "args.txt"), "w") as f:
                json.dump(self.args.__dict__, f, indent=4)
            start_epoch = 0
            print(f"\nstarting training from scratch")

        best_train, best_val = 0, 0
        for epoch in range(start_epoch, self.args.epochs):
            if self.main_thread:
                print(
                    f"epoch: {epoch}, best train: {round(best_train, 5)}, best val: {round(best_val, 5)}, lr: {round(self.optim.param_groups[0]['lr'], 5)}"
                )
                print("---------------")

            self.train_epoch()
            if self.main_thread:
                train_metrics = self.metric_meter.get()
                self.eval()
                val_metrics = self.metric_meter.get()

                if train_metrics["train acc"] > best_train:
                    print(
                        "\x1b[34m"
                        + f"train acc improved from {round(best_train, 5)} to {round(train_metrics['train acc'], 5)}"
                        + "\033[0m"
                    )
                    best_train = train_metrics["train acc"]

                if val_metrics["val acc"] > best_val:
                    print(
                        "\x1b[33m"
                        + f"val acc improved from {round(best_val, 5)} to {round(val_metrics['val acc'], 5)}"
                        + "\033[0m"
                    )
                    best_val = val_metrics["val acc"]
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(self.args.out_dir, f"best.ckpt"),
                    )

                torch.save(
                    {
                        "model": self.model.state_dict(),
                        "optim": self.optim.state_dict(),
                        "lr_sched": self.lr_sched.state_dict(),
                        "epoch": epoch,
                    },
                    os.path.join(self.args.out_dir, "last.ckpt"),
                )

            if epoch < self.args.warmup_epochs:
                self.optim.param_groups[0]["lr"] = epoch / self.args.warmup_epochs * self.args.lr
            else:
                self.lr_sched.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = utils.add_args(parser)
    args = parser.parse_args()
    utils.print_args(args)

    trainer = Trainer(args)
    if args.mode == "imp":
        trainer.train_imp()
    elif args.mode == "train":
        trainer.train()
    else:
        raise NotImplementedError(f"args.mode = {args.mode} is not implemented")

    if args.dist:
        torch.distributed.destroy_process_group()
