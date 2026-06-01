# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math
import random
from copy import copy
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.modules import Detect
from ultralytics.nn.tasks import DetectionModel, load_checkpoint
from ultralytics.utils.checks import check_model_file_from_stem
from ultralytics.utils.tal import make_anchors
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.patches import override_configs
from ultralytics.utils.plotting import plot_images, plot_labels
from ultralytics.utils.torch_utils import torch_distributed_zero_first, unwrap_model


class DetectionTrainer(BaseTrainer):
    """A class extending the BaseTrainer class for training based on a detection model.

    This trainer specializes in object detection tasks, handling the specific requirements for training YOLO models for
    object detection including dataset building, data loading, preprocessing, and model configuration.

    Attributes:
        model (DetectionModel): The YOLO detection model being trained.
        data (dict): Dictionary containing dataset information including class names and number of classes.
        loss_names (tuple): Names of the loss components used in training (box_loss, cls_loss, dfl_loss).

    Methods:
        build_dataset: Build YOLO dataset for training or validation.
        get_dataloader: Construct and return dataloader for the specified mode.
        preprocess_batch: Preprocess a batch of images by scaling and converting to float.
        set_model_attributes: Set model attributes based on dataset information.
        get_model: Return a YOLO detection model.
        get_validator: Return a validator for model evaluation.
        label_loss_items: Return a loss dictionary with labeled training loss items.
        progress_string: Return a formatted string of training progress.
        plot_training_samples: Plot training samples with their annotations.
        plot_training_labels: Create a labeled training plot of the YOLO model.
        auto_batch: Calculate optimal batch size based on model memory requirements.

    Examples:
        >>> from ultralytics.models.yolo.detect import DetectionTrainer
        >>> args = dict(model="yolo26n.pt", data="coco8.yaml", epochs=3)
        >>> trainer = DetectionTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None):
        """Initialize a DetectionTrainer object for training YOLO object detection models.

        Args:
            cfg (dict, optional): Default configuration dictionary containing training parameters.
            overrides (dict, optional): Dictionary of parameter overrides for the default configuration.
            _callbacks (dict, optional): Dictionary of callback functions to be executed during training.
        """
        super().__init__(cfg, overrides, _callbacks)

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """Build YOLO Dataset for training or validation.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): 'train' mode or 'val' mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for 'rect' mode.

        Returns:
            (Dataset): YOLO dataset object configured for the specified mode.
        """
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """Construct and return dataloader for the specified mode.

        Args:
            dataset_path (str): Path to the dataset.
            batch_size (int): Number of images per batch.
            rank (int): Process rank for distributed training.
            mode (str): 'train' for training dataloader, 'val' for validation dataloader.

        Returns:
            (DataLoader): PyTorch dataloader object.
        """
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle and not np.all(dataset.batch_shapes == dataset.batch_shapes[0]):
            LOGGER.warning("'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        return build_dataloader(
            dataset,
            batch=batch_size,
            workers=self.args.workers if mode == "train" else self.args.workers * 2,
            shuffle=shuffle,
            rank=rank,
            drop_last=self.args.compile and mode == "train",
        )

    def preprocess_batch(self, batch: dict) -> dict:
        """Preprocess a batch of images by scaling and converting to float.

        Args:
            batch (dict): Dictionary containing batch data with 'img' tensor.

        Returns:
            (dict): Preprocessed batch with normalized images.
        """
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")
        batch["img"] = batch["img"].float() / 255
        if self.args.multi_scale > 0.0:
            imgs = batch["img"]
            sz = (
                random.randrange(
                    int(self.args.imgsz * (1.0 - self.args.multi_scale)),
                    int(self.args.imgsz * (1.0 + self.args.multi_scale) + self.stride),
                )
                // self.stride
                * self.stride
            )  # size
            sf = sz / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
                ]  # new shape (stretched to gs-multiple)
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs
        return batch

    def set_model_attributes(self):
        """Set model attributes based on dataset information."""
        # Nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)
        # self.args.box *= 3 / nl  # scale to layers
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # scale to classes and layers
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        self.model.nc = self.data["nc"]  # attach number of classes to model
        self.model.names = self.data["names"]  # attach class names to model
        self.model.args = self.args  # attach hyperparameters to model
        if getattr(self.model, "end2end"):
            self.model.set_head_attr(max_det=self.args.max_det)

    def set_class_weights(self):
        """Compute and set class weights for handling class imbalance.

        Class weights are computed based on inverse class frequency in the training dataset,
        raised to the power of cls_pw (0 < cls_pw <= 1 dampens, cls_pw > 1 amplifies).
        Final weights are normalized so their mean equals 1.0.
        """
        assert 0 <= self.args.cls_pw <= 1.0, "cls_pw must be in the range [0, 1]"
        if self.args.cls_pw == 0.0:
            return
        classes = np.concatenate([lb["cls"].flatten() for lb in self.train_loader.dataset.labels], 0)
        class_counts = np.bincount(classes.astype(int), minlength=self.data["nc"]).astype(np.float32)
        class_counts = np.where(class_counts == 0, 1.0, class_counts)

        weights = (1.0 / class_counts) ** self.args.cls_pw  # apply power directly
        weights = weights / weights.mean()  # normalize so mean equals 1.0
        self.model.class_weights = torch.from_numpy(weights).to(self.device)
        LOGGER.info(f"Class weights: {self.model.class_weights.cpu().numpy().round(3)}")

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        """Return a YOLO detection model.

        Args:
            cfg (str, optional): Path to model configuration file.
            weights (str, optional): Path to model weights.
            verbose (bool): Whether to display model information.

        Returns:
            (DetectionModel): YOLO detection model.
        """
        model = DetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Return a DetectionValidator for YOLO model validation."""
        head = unwrap_model(self.model).model[-1]
        self.setup_teacher_model()
        base_loss_names = (
            ("box_loss", "cls_loss", "dfl_loss", "aux_loss", "proto_loss")
            if head.__class__.__name__ == "CBRDetect"
            else ("box_loss", "cls_loss", "dfl_loss")
        )
        self.val_loss_names = base_loss_names
        self.loss_names = base_loss_names + (("kd_cls_loss", "kd_box_loss") if self.teacher_model is not None else ())
        return yolo.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def label_loss_items(self, loss_items: list[float] | None = None, prefix: str = "train"):
        """Return a loss dict with labeled training loss items tensor.

        Args:
            loss_items (list[float], optional): List of loss values.
            prefix (str): Prefix for keys in the returned dictionary.

        Returns:
            (dict | list): Dictionary of labeled loss items if loss_items is provided, otherwise list of keys.
        """
        names = self.val_loss_names if prefix == "val" and hasattr(self, "val_loss_names") else self.loss_names
        keys = [f"{prefix}/{x}" for x in names]
        if loss_items is not None:
            loss_items = list(loss_items)
            if len(loss_items) < len(keys):
                loss_items.extend([0.0] * (len(keys) - len(loss_items)))
            elif len(loss_items) > len(keys):
                loss_items = loss_items[: len(keys)]
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        """Return a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def plot_training_samples(self, batch: dict[str, Any], ni: int) -> None:
        """Plot training samples with their annotations.

        Args:
            batch (dict[str, Any]): Dictionary containing batch data.
            ni (int): Batch index used for naming the output file.
        """
        plot_images(
            labels=batch,
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_training_labels(self):
        """Create a labeled training plot of the YOLO model."""
        boxes = np.concatenate([lb["bboxes"] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb["cls"] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)

    def auto_batch(self):
        """Get optimal batch size by calculating memory occupation of model.

        Returns:
            (int): Optimal batch size.
        """
        with override_configs(self.args, overrides={"cache": False}) as self.args:
            train_dataset = self.build_dataset(self.data["train"], mode="train", batch=16)
        max_num_obj = max(len(label["cls"]) for label in train_dataset.labels) * 4  # 4 for mosaic augmentation
        n = len(train_dataset)
        del train_dataset  # free memory
        return super().auto_batch(max_num_obj, dataset_size=n)

    def setup_teacher_model(self):
        """Load and freeze the teacher model for training-time distillation."""
        if hasattr(self, "teacher_model"):
            return self.teacher_model

        teacher = getattr(self.args, "teacher", None)
        self.teacher_model = None
        self._kd_shape_warned = False
        if not teacher:
            return None

        teacher = check_model_file_from_stem(teacher)
        if str(teacher).endswith(".pt"):
            teacher_model, _ = load_checkpoint(teacher, device=self.device, fuse=False)
        else:
            teacher_model = DetectionModel(teacher, nc=self.data["nc"], ch=self.data["channels"], verbose=RANK in {-1, 0})
            teacher_model = teacher_model.to(self.device).eval()

        teacher_nc = getattr(teacher_model.model[-1], "nc", self.data["nc"])
        if teacher_nc != self.data["nc"]:
            raise ValueError(
                f"Teacher model class count ({teacher_nc}) does not match dataset classes ({self.data['nc']})."
            )
        teacher_model.nc = self.data["nc"]
        teacher_model.names = self.data["names"]
        teacher_model.args = self.args
        if getattr(teacher_model, "end2end", False):
            teacher_model.set_head_attr(max_det=self.args.max_det)
        for p in teacher_model.parameters():
            p.requires_grad_(False)
        self.teacher_model = teacher_model.eval()
        LOGGER.info(f"Using teacher model for distillation: {teacher}")
        return self.teacher_model

    def compute_loss(self, batch: dict):
        """Compute student loss and optionally add teacher-student distillation losses."""
        if getattr(self, "teacher_model", None) is None:
            return super().compute_loss(batch)

        student_preds = self.model(batch["img"])
        student_model = unwrap_model(self.model)
        det_loss, det_loss_items = student_model.loss(batch, student_preds)
        with torch.no_grad():
            teacher_preds = self.forward_teacher(batch["img"])
        kd_cls_loss, kd_box_loss = self.compute_distillation_loss(student_preds, teacher_preds)
        kd_loss = kd_cls_loss * self.args.kd_cls + kd_box_loss * self.args.kd_box
        kd_items = det_loss_items.new_tensor([kd_cls_loss.detach(), kd_box_loss.detach()])
        return det_loss + kd_loss, torch.cat((det_loss_items, kd_items))

    def forward_teacher(self, imgs: torch.Tensor):
        """Run the frozen teacher to obtain raw detection outputs for distillation."""
        teacher = unwrap_model(self.teacher_model)
        y = []
        x = imgs
        head = teacher.model[-1]
        for m in teacher.model[:-1]:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in teacher.save else None)

        if head.f != -1:
            x = y[head.f] if isinstance(head.f, int) else [x if j == -1 else y[j] for j in head.f]
        if isinstance(head, Detect):
            return head.forward_head(x, **(head.one2one if getattr(head, "end2end", False) else head.one2many))
        raise TypeError("Teacher distillation currently supports detection heads derived from Detect only.")

    @staticmethod
    def select_distill_branch(preds):
        """Select the branch used for distillation from raw training predictions."""
        preds = preds[1] if isinstance(preds, tuple) else preds
        if isinstance(preds, dict):
            if "one2one" in preds:
                return preds["one2one"]
            if "one2many" in preds and "boxes" not in preds:
                return preds["one2many"]
        return preds

    def compute_distillation_loss(self, student_preds, teacher_preds):
        """Compute classification and box distillation losses from teacher outputs."""
        student_preds = self.select_distill_branch(student_preds)
        teacher_preds = self.select_distill_branch(teacher_preds)
        if (
            student_preds["scores"].shape != teacher_preds["scores"].shape
            or student_preds["boxes"].shape != teacher_preds["boxes"].shape
        ):
            if not self._kd_shape_warned:
                LOGGER.warning("Skipping KD loss because student and teacher prediction shapes do not match.")
                self._kd_shape_warned = True
            zero = student_preds["scores"].new_zeros(())
            return zero, zero

        s_scores = student_preds["scores"].permute(0, 2, 1).contiguous()
        t_scores = teacher_preds["scores"].permute(0, 2, 1).contiguous()
        teacher_prob = t_scores.sigmoid()
        teacher_conf = teacher_prob.max(dim=-1).values
        mask = teacher_conf > self.args.kd_conf
        zero = s_scores.new_zeros(())
        if not mask.any():
            return zero, zero

        temp = self.args.kd_temp
        kd_cls_loss = F.kl_div(
            F.log_softmax(s_scores[mask] / temp, dim=-1),
            F.softmax(t_scores[mask] / temp, dim=-1),
            reduction="batchmean",
        ) * (temp**2)

        criterion = unwrap_model(self.model).criterion
        criterion = criterion.one2one if hasattr(criterion, "one2one") else criterion
        anchor_points, stride_tensor = make_anchors(student_preds["feats"], criterion.stride, 0.5)
        s_boxes = criterion.bbox_decode(anchor_points, student_preds["boxes"].permute(0, 2, 1).contiguous()) * stride_tensor
        t_boxes = criterion.bbox_decode(anchor_points, teacher_preds["boxes"].permute(0, 2, 1).contiguous()) * stride_tensor
        weights = teacher_conf[mask].unsqueeze(-1)
        kd_box_loss = (F.smooth_l1_loss(s_boxes[mask], t_boxes[mask], reduction="none") * weights).sum() / (weights.sum() + 1e-6)
        return kd_cls_loss, kd_box_loss
