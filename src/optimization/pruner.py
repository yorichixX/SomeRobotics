"""Model pruning module for edge optimization."""

from typing import Tuple, Dict, Any
import os
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


class ModelPruner:
    """Applies weight pruning to PyTorch models and evaluates sparsity."""

    def __init__(self, output_dir: str = "weights"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    @staticmethod
    def calculate_sparsity(model: nn.Module) -> float:
        """Calculate the percentage of zero weights in the model."""
        total_weights = 0
        zero_weights = 0
        for name, param in model.named_parameters():
            if "weight" in name and param.is_floating_point():
                total_weights += param.numel()
                zero_weights += torch.sum(param == 0).item()
        return (zero_weights / max(1, total_weights)) * 100.0

    def prune_l1_unstructured(
        self,
        pytorch_model_path: str,
        amount: float = 0.30,
        output_pruned_path: str = "weights/pruned_model.pt",
    ) -> Tuple[str, Dict[str, Any]]:
        """Apply L1-norm unstructured pruning to all 2D convolutional layers.
        
        Args:
            pytorch_model_path: Path to PyTorch or YOLO checkpoint
            amount: Fraction of connections to prune (e.g. 0.30 = 30%)
            output_pruned_path: Destination for pruned model
            
        Returns:
            Tuple of (saved_path, metrics_dict)
        """
        try:
            from ultralytics import YOLO
            yolo_wrapper = YOLO(pytorch_model_path)
            model = yolo_wrapper.model
        except Exception:
            checkpoint = torch.load(pytorch_model_path, map_location="cpu", weights_only=False)
            model = checkpoint if isinstance(checkpoint, nn.Module) else checkpoint.get("model", checkpoint)

        sparsity_before = self.calculate_sparsity(model)
        pruned_layer_count = 0

        import copy
        model = copy.deepcopy(model)

        # Apply L1 unstructured pruning to Conv2d layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                prune.l1_unstructured(module, name="weight", amount=amount)
                # Make pruning permanent by removing reparameterization hooks
                prune.remove(module, "weight")
                pruned_layer_count += 1

        sparsity_after = self.calculate_sparsity(model)

        # Save pruned model
        torch.save({"model": model}, output_pruned_path)

        metrics = {
            "amount_requested": amount,
            "pruned_conv_layers": pruned_layer_count,
            "sparsity_before_pct": sparsity_before,
            "sparsity_after_pct": sparsity_after,
        }

        print(f"Pruning complete on {pruned_layer_count} Conv2d layers:")
        print(f"  Sparsity Before: {sparsity_before:.2f}%")
        print(f"  Sparsity After:  {sparsity_after:.2f}%")
        print(f"  Saved to:        {output_pruned_path}")

        return output_pruned_path, metrics
