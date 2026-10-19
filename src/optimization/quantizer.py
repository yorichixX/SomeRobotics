"""Model quantization module for edge deployment."""

from typing import Optional
import os
import shutil


class ModelQuantizer:
    """Handles exporting and quantizing models for accelerated edge inference."""

    def __init__(self, output_dir: str = "weights"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def export_to_onnx(
        self,
        pytorch_model_path: str,
        output_onnx_path: Optional[str] = None,
        imgsz: int = 640,
        opset: int = 14,
        half: bool = False,
    ) -> str:
        """Export PyTorch / YOLO model to ONNX format."""
        if output_onnx_path is None:
            base_name = os.path.splitext(os.path.basename(pytorch_model_path))[0]
            output_onnx_path = os.path.join(self.output_dir, f"{base_name}.onnx")

        try:
            from ultralytics import YOLO
            model = YOLO(pytorch_model_path)
            exported = model.export(
                format="onnx",
                imgsz=imgsz,
                opset=opset,
                half=half,
                dynamic=False,
                simplify=True,
            )
            if exported and os.path.exists(exported) and exported != output_onnx_path:
                shutil.move(exported, output_onnx_path)
            return output_onnx_path
        except Exception as e:
            print(f"Error during ONNX export: {e}")
            raise e

    def quantize_onnx_dynamic(
        self,
        input_onnx_path: str,
        output_quantized_path: Optional[str] = None,
    ) -> str:
        """Apply dynamic INT8 quantization to an ONNX model.
        
        Reduces model size by ~75% and improves CPU inference speed on edge devices.
        """
        if output_quantized_path is None:
            base_name = os.path.splitext(os.path.basename(input_onnx_path))[0]
            output_quantized_path = os.path.join(self.output_dir, f"{base_name}_int8.onnx")

        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType

            print(f"Applying INT8 dynamic quantization: {input_onnx_path} -> {output_quantized_path}")
            quantize_dynamic(
                model_input=input_onnx_path,
                model_output=output_quantized_path,
                weight_type=QuantType.QInt8,
            )

            orig_size = os.path.getsize(input_onnx_path) / (1024 * 1024)
            quant_size = os.path.getsize(output_quantized_path) / (1024 * 1024)
            print(f"Size reduction: {orig_size:.2f} MB -> {quant_size:.2f} MB (-{(1 - quant_size/orig_size)*100:.1f}%)")

            return output_quantized_path
        except ImportError:
            print("Warning: onnxruntime.quantization not available. Install onnxruntime.")
            raise

    def convert_onnx_fp16(
        self,
        input_onnx_path: str,
        output_fp16_path: Optional[str] = None,
    ) -> str:
        """Convert float32 ONNX model to float16 for GPU acceleration."""
        if output_fp16_path is None:
            base_name = os.path.splitext(os.path.basename(input_onnx_path))[0]
            output_fp16_path = os.path.join(self.output_dir, f"{base_name}_fp16.onnx")

        try:
            import onnx
            from onnxconverter_common import float16

            model = onnx.load(input_onnx_path)
            model_fp16 = float16.convert_float_to_float16(model)
            onnx.save(model_fp16, output_fp16_path)
            print(f"Saved FP16 ONNX model to: {output_fp16_path}")
            return output_fp16_path
        except ImportError:
            print("Warning: onnx or onnxconverter_common not available.")
            raise
