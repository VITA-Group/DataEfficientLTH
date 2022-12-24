import os
import torch
import numpy as np
import torch.nn.utils.prune as prune
import copy
from datetime import datetime
from functools import partial


def print_args(args):
    print("\n---- experiment configuration ----")
    args_ = vars(args)
    for arg, value in args_.items():
        print(f" * {arg} => {value}")
    print("----------------------------------")


def mask_prune_vit(model, mask_dict, prune_ff_only=False):
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            if prune_ff_only and "mlp.fc" in name:
                prune.CustomFromMask.apply(m, "weight", mask=mask_dict[name + ".weight_mask"])
            elif not prune_ff_only and "head" not in name:
                prune.CustomFromMask.apply(m, "weight", mask=mask_dict[name + ".weight_mask"])


def mask_prune(model, mask_dict):
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            prune.CustomFromMask.apply(m, "weight", mask=mask_dict[name + ".weight_mask"])


def extract_mask(model_dict):
    mask_dict = {}
    for key in model_dict.keys():
        if "mask" in key:
            mask_dict[key] = copy.deepcopy(model_dict[key])
    return mask_dict


def remove_prune_vit(model, prune_ff_only=False):
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            if prune_ff_only and "mlp.fc" in name:
                prune.remove(m, "weight")
            elif not prune_ff_only and "head" not in name:
                prune.remove(m, "weight")


def remove_prune(model):
    for _, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            prune.remove(m, "weight")


def l1_prune_vit(model, px, prune_ff_only=False):
    prune_params = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            if prune_ff_only and "mlp.fc" in name:
                prune_params.append((m, "weight"))
            elif not prune_ff_only and "head" not in name:
                prune_params.append((m, "weight"))
    prune_params = tuple(prune_params)
    prune.global_unstructured(
        prune_params,
        pruning_method=prune.L1Unstructured,
        amount=px,
    )


def l1_prune(model, px):
    prune_params = []
    for _, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            prune_params.append((m, "weight"))

    prune_params = tuple(prune_params)
    prune.global_unstructured(
        prune_params,
        pruning_method=prune.L1Unstructured,
        amount=px,
    )


def random_prune(model, px):
    prune_params = []
    for _, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            prune_params.append((m, "weight"))

    prune_params = tuple(prune_params)
    prune.global_unstructured(
        prune_params,
        pruning_method=prune.RandomUnstructured,
        amount=px,
    )


def check_sparsity_vit(model, prune_ff_only=False):
    sum_list = 0
    zero_sum = 0

    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            if prune_ff_only and "mlp.fc" in name:
                sum_list = sum_list + float(m.weight.nelement())
                zero_sum = zero_sum + float(torch.sum(m.weight == 0))
            elif not prune_ff_only and "head" not in name:
                sum_list = sum_list + float(m.weight.nelement())
                zero_sum = zero_sum + float(torch.sum(m.weight == 0))

    return 100 * (1 - zero_sum / sum_list)


def check_sparsity(model):
    sum_list = 0
    zero_sum = 0

    for _, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            sum_list = sum_list + float(m.weight.nelement())
            zero_sum = zero_sum + float(torch.sum(m.weight == 0))
    return 100 * (1 - zero_sum / sum_list)


def add_args(parser):
    parser.add_argument("--mode", type=str, default="train", help="start distributed training")
    parser.add_argument("--dist", action="store_true", help="start distributed training")
    parser.add_argument("--seed", type=int, default=42, help="set experiment seed.")
    parser.add_argument("--dset", type=str, default="cifar10", help="choose dataset")
    parser.add_argument(
        "--contrast_aug",
        action="store_true",
        help="apply contrastive augmentation (source: simclr)",
    )
    parser.add_argument(
        "--color_jitter_strength", type=float, default=0.5, help="color jitter strength"
    )
    parser.add_argument(
        "--color_jitter_prob", type=float, default=0.8, help="color jitter probability"
    )
    parser.add_argument("--gray_prob", type=float, default=0.2, help="gray probability")
    parser.add_argument("--rand_aug", action="store_true", help="apply random augmentation")
    parser.add_argument(
        "--n_rand_aug", type=int, default=4, help="number of sequential random augmentations"
    )
    parser.add_argument("--auto_aug", action="store_true", help="apply auto augmentation")
    parser.add_argument(
        "--auto_aug_policy",
        type=int,
        default=2,
        help="autoaugment policy number (eg: 1: imagenet, 2: cifar)",
    )
    parser.add_argument(
        "--custom_aug",
        action="store_true",
        help="apply custom augmentation (source: imgaug) note: very strong",
    )
    parser.add_argument("--blur", action="store_true", help="apply gaussian blur")
    parser.add_argument("--blur_sigma", type=list, default=[0.1, 2], help="blur sigma")
    parser.add_argument("--blur_prob", type=float, default=0.5, help="blur probability")
    parser.add_argument("--cutout", action="store_true", help="apply cutout")
    parser.add_argument("--cut_len", type=int, default=16, help="cutsize in cutout")
    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/sneezygiraffe/data/cifar10",
        help="dataset directory.",
    )
    # parser.add_argument("--data_root", type=str, required=True, help="dataset directory.")
    parser.add_argument(
        "--data_size",
        type=float,
        default=1.0,
        help="training dataset size (fraction or number of samples).",
    )
    parser.add_argument("--long_tailed", action="store_true", help="long tailed classification")
    parser.add_argument("--long_tailed_type", type=str, default="exp", help="long tailed type")
    parser.add_argument(
        "--long_tailed_factor", type=float, default=0.01, help="long tailed factor"
    )
    parser.add_argument("--batch_size", type=int, default=100, help="batch size.")
    parser.add_argument(
        "--n_workers", type=int, default=4, help="number of workers for dataloading."
    )
    parser.add_argument("--net", type=str, default="resnet18", help="network name")
    parser.add_argument("--in_planes", type=int, default=64, help="resnet init feature size")
    parser.add_argument(
        "--pretrained", action="store_true", help="use pretrained torchvision ckpt"
    )
    parser.add_argument("--optim", type=str, default="sgd", help="optimizer name")
    parser.add_argument("--lr", type=float, default=0.1, help="sgd learning rate.")
    parser.add_argument("--momentum", type=float, default=0.9, help="sgd optimizer momentum.")
    parser.add_argument(
        "--weight_decay", type=float, default=5e-4, help="sgd optimizer weight decay."
    )
    parser.add_argument("--warmup_epochs", type=int, default=0, help="number of lr warmup epochs.")
    parser.add_argument("--lr_sched", type=str, default="cosine", help="lr scheduler name.")
    parser.add_argument(
        "--multi_step_milestones", type=list, default=[100, 150], help="multi step lr milestones."
    )
    parser.add_argument("--multi_step_gamma", type=float, default=0.1, help="multi step lr gamma.")
    parser.add_argument("--resume", action="store_true", help="resume training from checkpoint.")
    parser.add_argument(
        "-o",
        "--out_dir",
        type=str,
        default=f"{datetime.now().strftime('%Y-%m-%d_%H-%M')}",
        help="path to output directory [default: year-month-date_hour-minute].",
    )
    parser.add_argument(
        "--pruning_iters", type=int, default=16, help="number of pruning iterations."
    )
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs.")
    parser.add_argument("--rewind_type", type=str, default="epoch", help="rewind type.")
    parser.add_argument("--rewind_epoch", type=int, default=2, help="rewind epochs.")
    parser.add_argument("--prune_type", type=str, default="l1", help="pruning type.")
    parser.add_argument("--prune_rate", type=float, default=0.2, help="pruning rate.")
    parser.add_argument("--load_ticket", type=str, default="", help="load ticket to train.")
    parser.add_argument(
        "--adv_prop", action="store_true", help="adversarial propagation training scheme."
    )
    parser.add_argument(
        "--cos_criterion", action="store_true", help="use cosine loss with cross entropy."
    )
    parser.add_argument("--cos_linear", action="store_true", help="use cosine linear layer.")
    parser.add_argument("--tvmf_linear", action="store_true", help="use tvmf linear layer.")
    parser.add_argument("--attack_n_iter", type=int, default=1, help="number of attack iters.")
    parser.add_argument("--attack_eps", type=float, default=8 / 255, help="attack epsilon.")
    parser.add_argument("--attack_step_size", type=int, default=2 / 255, help="attack step size.")
    parser.add_argument(
        "--prune_ff_only", action="store_true", help="prune feedforward layers only"
    )

    return parser


def setup_device(dist):
    if dist:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.environ.get("LOCAL_RANK"))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        local_rank = None
        device = torch.device("cuda:0")

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return device, local_rank


def pbar(p=0, msg="", bar_len=20):
    msg = msg.ljust(50)
    block = int(round(bar_len * p))
    text = "\rProgress: [{}] {}% {}".format(
        "\x1b[32m" + "=" * (block - 1) + ">" + "\033[0m" + "-" * (bar_len - block),
        round(p * 100, 2),
        msg,
    )
    print(text, end="")
    if p == 1:
        print()


class AvgMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.metrics = {}

    def add(self, batch_metrics):
        if self.metrics == {}:
            for key, value in batch_metrics.items():
                self.metrics[key] = [value]
        else:
            for key, value in batch_metrics.items():
                self.metrics[key].append(value)

    def get(self):
        return {key: np.mean(value) for key, value in self.metrics.items()}

    def msg(self):
        avg_metrics = {key: np.mean(value) for key, value in self.metrics.items()}
        return "".join(["[{}] {:.5f} ".format(key, value) for key, value in avg_metrics.items()])


def setup_optim(args, params):
    if args.optim == "sgd":
        optim = torch.optim.SGD(
            params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
        )
    elif args.optim == "adam":
        optim = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError(f"args.optim = {args.optim} is not implemented")
    return optim


def setup_lr_sched(args, optim):
    if args.warmup_epochs > 0:
        for group in optim.param_groups:
            group["lr"] = 1e-12 / args.warmup_epochs * group["lr"]
    if args.lr_sched == "cosine":
        lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=args.epochs - args.warmup_epochs
        )
    elif args.lr_sched == "multi_step":
        args.multi_step_milestones = [
            milestone - args.warmup_epochs for milestone in args.multi_step_milestones
        ]
        lr_sched = torch.optim.lr_scheduler.MultiStepLR(
            optim, milestones=args.multi_step_milestones, gamma=args.multi_step_gamma
        )
    else:
        raise NotImplementedError(f"args.lr_sched = {args.lr_sched} not implemented")
    return optim, lr_sched


def to_status(m, status):
    if hasattr(m, "batch_type"):
        m.batch_type = status


to_clean = partial(to_status, status="clean")
to_adv = partial(to_status, status="adv")


class MixBatchNorm2d(torch.nn.BatchNorm2d):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super(MixBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.aux_bn = torch.nn.BatchNorm2d(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )
        self.batch_type = "clean"

    def forward(self, input):
        if self.batch_type == "adv":
            input = self.aux_bn(input)
        elif self.batch_type == "clean":
            input = super(MixBatchNorm2d, self).forward(input)
        else:
            raise NotImplementedError
        return input


def modify_bn(model):
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            modify_bn(module)

        if isinstance(module, torch.nn.BatchNorm2d):
            new_bn = MixBatchNorm2d(
                module.num_features,
                module.eps,
                module.momentum,
                module.affine,
                module.track_running_stats,
            )
            new_bn.load_state_dict(module.state_dict(), strict=False)
            new_bn.aux_bn.load_state_dict(module.state_dict())
            setattr(model, name, new_bn)


class PGDAttacker:
    def __init__(self, n_iter, eps, step_size, start_clean_p=0.0):
        self.n_iter = n_iter
        self.eps = eps
        self.step_size = step_size
        self.start_clean_p = start_clean_p

    def attack(self, img, tgt, model):
        low = torch.clamp(img - self.eps, min=-1.0, max=1.0)
        high = torch.clamp(img + self.eps, min=-1.0, max=1.0)

        noise = torch.empty_like(img, device=img.device).uniform_(-self.eps, self.eps)
        use_noise = (torch.randn([]) > self.start_clean_p).float()
        adv = img + use_noise * noise

        for _ in range(self.n_iter):
            adv.requires_grad = True
            logits = model(adv)
            loss = torch.nn.functional.cross_entropy(logits, tgt)
            grad = torch.autograd.grad(loss, adv, retain_graph=False, create_graph=False)[0]
            adv = adv + torch.sign(grad) * self.step_size
            adv = torch.where(adv > low, adv, low)
            adv = torch.where(adv < high, adv, high).detach()

        return adv
