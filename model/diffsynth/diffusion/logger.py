import os, glob, torch
from accelerate import Accelerator


class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x:x, max_checkpoints=5):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.num_steps = 0
        self.max_checkpoints = max_checkpoints


    def _cleanup_old_checkpoints(self):
        """只保留最新的 max_checkpoints 个权重文件。"""
        if self.max_checkpoints is None or self.max_checkpoints <= 0:
            return
        ckpts = sorted(glob.glob(os.path.join(self.output_path, "*.safetensors")), key=os.path.getmtime)
        while len(ckpts) > self.max_checkpoints:
            old = ckpts.pop(0)
            os.remove(old)
            print(f"[ModelLogger] Removed old checkpoint: {os.path.basename(old)}")


    def on_step_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None):
        self.num_steps += 1
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


    def on_epoch_end(self, accelerator: Accelerator, model: torch.nn.Module, epoch_id):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
            accelerator.save(state_dict, path, safe_serialization=True)
            self._cleanup_old_checkpoints()


    def on_training_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


    def save_model(self, accelerator: Accelerator, model: torch.nn.Module, file_name):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)
            self._cleanup_old_checkpoints()
