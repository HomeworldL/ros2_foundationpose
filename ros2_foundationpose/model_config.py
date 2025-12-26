from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
import rclpy
import torch
from typing import List
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node

package_share_directory = get_package_share_directory("ros2_foundationpose")
CKECKPOINT_ROOT = os.path.join(package_share_directory, "weights")

@dataclass
class ROSInferenceModelConfig(ABC):
    model_name: str
    device: str = "cuda:0"

    # mypy doesn't understand abstractclassmethod, so we use this workaround
    @abstractmethod
    def get_predictor(self):
        pass

    @classmethod
    @abstractmethod
    def from_rosparam(cls):
        pass

    @classmethod
    @abstractmethod
    def from_args(cls):
        pass


@dataclass
class SAMConfig(ROSInferenceModelConfig):
    model_type: str = "vit_t"
    mode: str = "prompt"

    model_checkpoint_root = os.path.join(CKECKPOINT_ROOT, "sam")
    model_checkpoints = {
        "vit_t": os.path.join(model_checkpoint_root, "mobile_sam.pt"),
        "vit_b": os.path.join(model_checkpoint_root, "sam_vit_b.pth"),
        "vit_l": os.path.join(model_checkpoint_root, "sam_vit_l.pth"),
        "vit_h": os.path.join(model_checkpoint_root, "sam_vit_h.pth"),
        "vit_b_hq": os.path.join(model_checkpoint_root, "sam_vit_b_hq.pth"),
        "vit_l_hq": os.path.join(model_checkpoint_root, "sam_vit_l_hq.pth"),
        "vit_h_hq": os.path.join(model_checkpoint_root, "sam_vit_h_hq.pth"),
    }

    def get_predictor(self):
        assert self.model_type in SAMConfig.model_checkpoints
        assert self.mode in ["prompt", "automatic"]
        if "hq" in self.model_type:
            from segment_anything_hq import (
                sam_model_registry,
                SamAutomaticMaskGenerator,
                SamPredictor,
            )
        elif self.model_type == "vit_t":
            from mobile_sam import (
                sam_model_registry,
                SamAutomaticMaskGenerator,
                SamPredictor,
            )
        else:
            from segment_anything import (
                sam_model_registry,
                SamAutomaticMaskGenerator,
                SamPredictor,
            )
        model = sam_model_registry[self.model_type[:5]](checkpoint=self.model_checkpoints[self.model_type])
        model.to(device=self.device).eval()
        return SamPredictor(model) if self.mode == "prompt" else SamAutomaticMaskGenerator(model)

    @classmethod
    def from_args(cls, model_type: str = "vit_t", mode: str = "prompt", device: str = "cuda:0"):
        return cls(model_name="SAM", model_type=model_type, mode=mode, device=device)

    @classmethod
    def from_rosparam(cls, node: Node):
        model_type = node.get_parameter("model_type_sam").value
        mode = node.get_parameter("mode_sam").value
        device = node.get_parameter("device").value
        return cls.from_args(model_type, mode, device)
    
    
@dataclass
class GroundingDINOConfig(ROSInferenceModelConfig):
    model_config = os.path.join(CKECKPOINT_ROOT, "groundingdino/GroundingDINO_SwinT_OGC.py")
    model_checkpoint = os.path.join(CKECKPOINT_ROOT, "groundingdino/groundingdino_swint_ogc.pth")

    def get_predictor(self):
        try:
            from groundingdino.util.inference import Model as GroundingDINOModel
        except ImportError:
            from GroundingDINO.groundingdino.util.inference import (
                Model as GroundingDINOModel,
            )
        return GroundingDINOModel(
            model_config_path=self.model_config,
            model_checkpoint_path=self.model_checkpoint,
            device=self.device,
        )

    @classmethod
    def from_args(cls, device: str = "cuda:0"):
        return cls(model_name="GroundingDINO", device=device)

    @classmethod
    def from_rosparam(cls, node: Node):
        device = node.get_parameter("device").value
        return cls.from_args(device)
