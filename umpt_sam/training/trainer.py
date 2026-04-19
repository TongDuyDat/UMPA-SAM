import os
import torch
import torch.nn.functional as F
from datetime import datetime
from tqdm import tqdm
from datetime import datetime
class UMPATrainer:
    def __init__(
        self, 
        model, 
        train_loader, 
        val_loader, 
        test_loader,
        optimizer, 
        scheduler, 
        train_config, 
        composer_loss, 
        evaluate_fn, 
        save_dir="checkpoints", 
        device="cuda",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = train_config
        self.composer_loss = composer_loss
        self.evaluate_fn = evaluate_fn
        
        self.save_dir = save_dir
        self.device = device
        
        self.best_val_dice = 0.0
        self.total_epochs = self.scheduler.total_epochs
        self.scaler = torch.amp.GradScaler(device, enabled=True) 
        
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") # VD: 20260311_164500
        self.save_dir = os.path.join(save_dir, f"run_{run_id}")
        os.makedirs(self.save_dir, exist_ok=True)
        self.log_file_path = os.path.join(self.save_dir, "training_log.txt")
        self._init_log_file()

    def _init_log_file(self):
        start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"BẮT ĐẦU HUẤN LUYỆN LÚC: {start_time} | Tổng số: {self.total_epochs} Epochs\n")
            f.write(f"{'='*60}\n")

    def _init_log_file(self):
        start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        opt_name = self.optimizer.__class__.__name__
        opt_lr = self.optimizer.param_groups[0].get('lr', 'N/A')
        opt_wd = self.optimizer.param_groups[0].get('weight_decay', 'N/A')
        
        train_bs = getattr(self.train_loader, 'batch_size', 'N/A')
        val_bs = getattr(self.val_loader, 'batch_size', 'N/A')
        train_len = len(self.train_loader.dataset) if hasattr(self.train_loader, 'dataset') else 'N/A'
        val_len = len(self.val_loader.dataset) if hasattr(self.val_loader, 'dataset') else 'N/A'
        
        loss_name = self.composer_loss.__class__.__name__
        loss_cfg = getattr(self.composer_loss, 'config_loss', 'N/A')
        
        k_val = getattr(self.config, 'K', 'N/A')
        
        phases_info = ""
        if hasattr(self.scheduler, 'config'):
            cfg = self.scheduler.config
            phases = [getattr(cfg, 'phase1', None), getattr(cfg, 'phase2', None), getattr(cfg, 'phase3', None)]
            
            for i, p in enumerate(phases):
                if p is not None:
                    p_name = getattr(p, 'name', f'Phase {i+1}')
                    p_lr = getattr(p, 'lr', 'N/A')
                    p_lam = getattr(p, 'lambda_con', 'N/A')
                    p_epochs = getattr(p, 'epochs', 'N/A')
                    phases_info += f"    - {p_name} ({p_epochs} epochs) | LR: {p_lr} | lambda_con: {p_lam}\n"
        else:
            phases_info = "    - (Không tìm thấy thông tin chi tiết các Phase)\n"
        log_content = f"""
        {'='*60}
        BẮT ĐẦU HUẤN LUYỆN LÚC: {start_time}
        Thư mục lưu trữ: {self.save_dir}
        {'='*60}
        [CẤU HÌNH HUẤN LUYỆN - HYPERPARAMETERS]

        1. Nhóm Thuật toán (UMPA-SAM):
        - K (Số lượng nhiễu): {k_val}
        - Device: {self.device}
        - Mixed Precision (AMP): Bật (Tối ưu cho GPU A100)

        2. Nhóm Dữ liệu (Data Loaders):
        - Train dataset: {train_len} ảnh | Batch size: {train_bs}
        - Valid dataset: {val_len} ảnh | Batch size: {val_bs}

        3. Nhóm Tối ưu hóa (Optimizer):
        - Loại Optimizer: {opt_name}
        - Base Learning Rate: {opt_lr}
        - Weight Decay: {opt_wd}

        4. Nhóm Hàm Loss (Composer Loss):
        - Loại Loss: {loss_name}
        - Trọng số ban đầu: {loss_cfg}

        5. Nhóm Lịch trình (Scheduler & Phases):
        - Tổng số Epochs: {self.total_epochs}
        - Lộ trình Phases chi tiết:
        {phases_info}
        {'='*60}
        """
        # Ghi vào file
        with open(self.log_file_path, "a", encoding="utf-8") as f:
            f.write(log_content)
        
        print(log_content)

    def _write_log(self, message):
        print(message)
        with open(self.log_file_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    def train_one_epoch(self, epoch, phase):
        self.model.train()
        
        epoch_total_loss = 0.0
        epoch_seg_loss = 0.0
        epoch_con_loss = 0.0

        self.composer_loss.config_loss['consistency_loss_weight'] = phase.lambda_con

        phase_name = getattr(phase, 'name', f'Phase {epoch}')
        pbar = tqdm(self.train_loader, desc=f"Train Epoch {epoch} [{phase_name}]", leave=False)
        
        for batch in pbar:
            images = batch['image'].to(self.device, non_blocking=True)
            gt_masks = batch['mask'].to(self.device, non_blocking=True)  
            
            boxes = batch.get('bbox', None)
            points = batch.get('points', None)
            point_labels = batch.get('point_labels', None)
            captions = batch.get('text', None)
            
            if boxes is not None: boxes = boxes.to(self.device, non_blocking=True)
            if points is not None: points = points.to(self.device, non_blocking=True)
            if point_labels is not None: point_labels = point_labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            # Chạy Forward trong mode tự động ép kiểu (AMP) để tối ưu VRAM/Tốc độ
            with torch.amp.autocast(self.device, enabled=True):
                outputs = self.model(
                    image=images,
                    boxes=boxes,
                    points=points,
                    point_labels=point_labels,
                    captions=captions
                )
                pred_masks = outputs['pred_masks'] 
                
                if pred_masks.shape[-2:] != gt_masks.shape[-2:]:
                    pred_masks = F.interpolate(pred_masks, size=gt_masks.shape[-2:], mode="bilinear", align_corners=False)

                perturbed_masks = None
                if self.config.K > 0 and phase.lambda_con > 0.0:
                    k_outputs = self.model.forward_k_perturbations(
                        image=images,
                        K=self.config.K,
                        boxes=boxes,
                        points=points,
                        point_labels=point_labels,
                        captions=captions
                    )
                    
                    p_masks_list = []
                    for out in k_outputs:
                        p_mask = out['pred_masks']
                        if p_mask.shape[-2:] != gt_masks.shape[-2:]:
                            p_mask = F.interpolate(p_mask, size=gt_masks.shape[-2:], mode="bilinear", align_corners=False)
                        p_masks_list.append(p_mask)
                        
                    perturbed_masks = torch.stack(p_masks_list, dim=1)

                loss_dict = self.composer_loss(
                    pred_masks=pred_masks, 
                    gt_masks=gt_masks, 
                    perturbed_masks=perturbed_masks
                )
                total_loss = loss_dict['total_loss']

            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # [ĐÃ KHỚP LOGIC] Cộng dồn Loss
            epoch_total_loss += total_loss.item()
            epoch_seg_loss += loss_dict['seg_loss'].item()
            con_loss_val = loss_dict['consistency_loss'].item() if isinstance(loss_dict['consistency_loss'], torch.Tensor) else 0.0
            epoch_con_loss += con_loss_val

            pbar.set_postfix({
                "L_tot": f"{total_loss.item():.4f}", 
                "L_seg": f"{loss_dict['seg_loss'].item():.4f}",
                "L_con": f"{con_loss_val:.4f}"
            })

        num_batches = len(self.train_loader)
        return {
            "total_loss": epoch_total_loss / num_batches,
            "seg_loss": epoch_seg_loss / num_batches,
            "consistency_loss": epoch_con_loss / num_batches
        }

    def run(self):
        self._write_log(f"BẮT ĐẦU HUẤN LUYỆN: {self.total_epochs} Epochs | Lưu log tại: {self.log_file_path}")

        for epoch in range(1, self.total_epochs + 1):
            phase = self.scheduler.get_current_phase(epoch)
            self.scheduler.apply_phase(model=self.model, epoch=epoch, optimizer=self.optimizer)
            phase_name = getattr(phase, 'name', 'Phase')
            
            self._write_log(f"\n=== Epoch {epoch}/{self.total_epochs} | Phase: {phase_name} "
                            f"| LR: {phase.lr} | lambda_con: {phase.lambda_con} ===")

            train_info = self.train_one_epoch(epoch, phase)

            val_metrics = self.evaluate_fn(self.model, self.val_loader, device=self.device)
            val_dice = val_metrics.get('dice', 0.0)
            val_miou = val_metrics.get('miou', 0.0)

            log_str = (f"[LOG] | Epoch {epoch:03d} | {phase_name:10s} "
                       f"| L_seg: {train_info['seg_loss']:.4f} "
                       f"| L_con: {train_info['consistency_loss']:.4f} "
                       f"| L_total: {train_info['total_loss']:.4f} "
                       f"| val_dice: {val_dice:.4f} | val_miou: {val_miou:.4f}")
            self._write_log(log_str)

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_dice': val_dice,
                'val_miou': val_miou
            }
            
            torch.save(checkpoint, os.path.join(self.save_dir, "latest_model.pth"))
            
            if val_dice > self.best_val_dice:
                self.best_val_dice = val_dice
                torch.save(checkpoint, os.path.join(self.save_dir, "best_model.pth"))
                self._write_log(f"[NEW BEST] Đã lưu model đỉnh nhất mới với val_dice: {self.best_val_dice:.4f}")

        self._write_log("\n" + "="*60)
        self._write_log("HUẤN LUYỆN HOÀN TẤT! ĐANG NẠP LẠI BEST MODEL...")
        
        best_model_path = os.path.join(self.save_dir, "best_model.pth")
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            best_epoch = checkpoint.get('epoch', '?')
            self._write_log(f"Đã nạp thành công trọng số tốt nhất từ Epoch {best_epoch}")
            
            if self.test_loader is not None:
                self._write_log("ĐANG ĐÁNH GIÁ CHUNG CUỘC TRÊN TẬP TEST ĐỘC LẬP (UNSEEN DATA)...")
                final_metrics = self.evaluate_fn(self.model, self.test_loader, device=self.device)
                self._write_log(f"[KẾT QUẢ TEST CHUNG CUỘC]")
            else:
                self._write_log("[CẢNH BÁO] Không có test_loader. Đang đánh giá lại trên tập Validation...")
                final_metrics = self.evaluate_fn(self.model, self.val_loader, device=self.device)
                self._write_log(f"[KẾT QUẢ VAL CHUNG CUỘC]")
                
            self._write_log(f"   - Final Dice: {final_metrics.get('dice', 0.0):.4f}")
            self._write_log(f"   - Final mIoU: {final_metrics.get('miou', 0.0):.4f}")
        else:
            self._write_log(" [CẢNH BÁO] Không tìm thấy file best_model.pth để đánh giá.")
        
        self._write_log("="*60 + "\n")