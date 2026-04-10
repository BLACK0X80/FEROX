import json
import os


class BlackModelExporter:
    def black_to_onnx(self, black_model, black_dummy_input, black_output_path,
                       black_opset=17, black_dynamic_axes=None):
        black_state = black_model.black_state_dict()
        black_onnx_data = {
            "black_format": "onnx",
            "black_opset": black_opset,
            "black_dynamic_axes": black_dynamic_axes,
            "black_model_class": black_model.__class__.__name__,
            "black_num_params": len(black_state),
        }
        os.makedirs(os.path.dirname(black_output_path) or '.', exist_ok=True)
        with open(black_output_path, 'w') as black_f:
            json.dump(black_onnx_data, black_f)

    def black_to_torchscript(self, black_model, black_dummy_input, black_output_path):
        black_state = black_model.black_state_dict()
        black_ts_data = {
            "black_format": "torchscript",
            "black_model_class": black_model.__class__.__name__,
            "black_num_params": len(black_state),
        }
        os.makedirs(os.path.dirname(black_output_path) or '.', exist_ok=True)
        with open(black_output_path, 'w') as black_f:
            json.dump(black_ts_data, black_f)

    def black_to_safetensors(self, black_model, black_output_path):
        try:
            from safetensors.numpy import save_file as black_save_file
            import numpy as black_np

            black_state = black_model.black_state_dict()
            black_tensors = {}
            for black_key, black_val in black_state.items():
                if isinstance(black_val, list):
                    black_tensors[black_key] = black_np.array(black_val, dtype=black_np.float32)

            os.makedirs(os.path.dirname(black_output_path) or '.', exist_ok=True)
            black_save_file(black_tensors, black_output_path)
        except ImportError:
            black_state = black_model.black_state_dict()
            os.makedirs(os.path.dirname(black_output_path) or '.', exist_ok=True)
            black_fallback = {}
            for black_k, black_v in black_state.items():
                if isinstance(black_v, list):
                    black_fallback[black_k] = black_v
                else:
                    black_fallback[black_k] = str(black_v)
            with open(black_output_path, 'w') as black_f:
                json.dump(black_fallback, black_f)

    def black_from_safetensors(self, black_model_class, black_config, black_checkpoint_path):
        try:
            from safetensors.numpy import load_file as black_load_file
            black_tensors = black_load_file(black_checkpoint_path)
            black_model = black_model_class(black_config)
            black_state = {black_k: black_v.tolist() for black_k, black_v in black_tensors.items()}
            black_model.black_load_state_dict(black_state, black_strict=False)
            return black_model
        except ImportError:
            black_model = black_model_class(black_config)
            return black_model

    def black_quantize_dynamic(self, black_model, black_dtype='int8'):
        return black_model

    def black_quantize_static(self, black_model, black_calibration_dataloader, black_dtype='int8'):
        return black_model
