# umpt_sam/training/phase_scheduler.py
import torch.nn as nn

class PhaseScheduler:
    def __init__(self, train_config):
        """
        train_config là object TrainConfig chứa phase1, phase2, phase3
        được định nghĩa trong Task 1.2 của PLAN (1).md
        """
        self.config = train_config
        
        # Tính mốc thời gian chuyển phase
        self.phase1_end = self.config.phase1.epochs
        self.phase2_end = self.phase1_end + self.config.phase2.epochs
        self.total_epochs = self.phase2_end + self.config.phase3.epochs

    def get_current_phase(self, epoch: int):
        """Trả về PhaseConfig dựa trên epoch hiện tại"""
        if epoch <= self.phase1_end:
            return self.config.phase1
        elif epoch <= self.phase2_end:
            return self.config.phase2
        else:
            return self.config.phase3

    def get_lambda_con(self, epoch: int) -> float:
        """Trả về lambda_con của phase hiện tại"""
        phase = self.get_current_phase(epoch)
        return phase.lambda_con

    def _set_requires_grad(self, module: nn.Module, requires_grad: bool):
        """Helper function bật/tắt gradient cho module"""
        if module is not None:
            for param in module.parameters():
                param.requires_grad = requires_grad

    def apply_phase(self, model: nn.Module, epoch: int, optimizer=None):
        """Freeze/unfreeze đúng component và (tùy chọn) update lr"""
        phase = self.get_current_phase(epoch)
        
        # 1. Đóng băng/Mở đóng băng theo Config Ngày 1
        # Lưu ý: Image Encoder luôn freeze theo thiết kế của SAM
        self._set_requires_grad(model.image_encoder, not phase.freeze_image_encoder)
        self._set_requires_grad(model.prompt_encoder, not phase.freeze_prompt_encoder)
        self._set_requires_grad(model.sam_mask_decoder, not phase.freeze_mask_decoder)
        
        # UPFE và Text Projection luôn trainable theo thiết kế Ngày 3
        self._set_requires_grad(model.upfe_encoder, True)
        if hasattr(model, 'text_projection') and not isinstance(model.text_projection, nn.Identity):
            self._set_requires_grad(model.text_projection, True)

        # 2. Cập nhật Learning Rate (Nếu có truyền optimizer vào)
        if optimizer is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = phase.lr
                
        return phase