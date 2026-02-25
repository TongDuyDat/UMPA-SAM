# Kế hoạch hoàn thành UMPA-SAM — 1 tuần

## Tổng quan kiến trúc

```
UMPA-SAM
├── MPPG  — Multi-Prompt Perturbation Generator     ✅ Hoàn thành
├── UPFE  — Unified Prompt Fusion Encoder           ✅ Hoàn thành
├── MPCL  — Multi-Prompt Consistency Loss           ✅ Hoàn thành
├── DiceLoss                                        ✅ Hoàn thành
└── Pipeline (model + training + eval)              ❌ Chưa làm
```

## Cấu trúc thư mục đích

```
umpt_sam/
├── config/
│   ├── __init__.py
│   ├── model_config.py        ← Ngày 1
│   └── train_config.py        ← Ngày 1
├── data/
│   ├── __init__.py
│   ├── polyp_dataset.py       ← Ngày 2
│   └── transforms.py          ← Ngày 2
├── losses/
│   ├── __init__.py
│   ├── dice_loss.py           ✅ Đã có
│   ├── consistency_loss.py    ✅ Đã có
│   └── loss_composer.py       ← Ngày 4
├── modules/
│   ├── __init__.py
│   ├── modules.py             ✅ Đã có (fix Ngày 1)
│   ├── upf_encoder.py         ✅ Đã có
│   └── umpa_model.py          ← Ngày 3
├── training/
│   ├── __init__.py
│   ├── phase_scheduler.py     ← Ngày 5
│   └── trainer.py             ← Ngày 5
├── evaluate.py                ← Ngày 6
└── umpa_sam.py                ← Ngày 4
```

---

## Ngày 1 — Config + Fix MPPG wrapper

### Mục tiêu
Toàn bộ hyperparameter tập trung một chỗ. Mọi thay đổi từ reviewer chỉ sửa config, không đụng vào logic.

### Task 1.1 — Viết `config/model_config.py`

| Trường | Kiểu | Mô tả |
|--------|------|-------|
| `MPPGConfig.sigma_b` | float | Độ lệch chuẩn nhiễu bbox |
| `MPPGConfig.gamma_range` | tuple | Range nhiễu asymmetric bbox |
| `MPPGConfig.rotation_range` | float | Góc xoay tối đa (độ) |
| `MPPGConfig.sigma_p` | float | Độ lệch chuẩn nhiễu point |
| `MPPGConfig.q_flip` | float | Xác suất flip nhãn point |
| `MPPGConfig.dilate_radius` | int | Bán kính dilation mask |
| `MPPGConfig.erode_radius` | int | Bán kính erosion mask |
| `MPPGConfig.warp_strength` | float | Cường độ warp mask |
| `MPPGConfig.sigma_t` | float | Độ lệch chuẩn nhiễu text embedding |
| `UPFEConfig.embed_dim` | int | Chiều embedding (256) |
| `UPFEConfig.scoring_hidden_dim` | int | Chiều ẩn scoring network |
| `UMPAModelConfig.sam_checkpoint` | str | Đường dẫn SAM checkpoint |
| `UMPAModelConfig.embed_dim` | int | SAM embed dim (256) |
| `UMPAModelConfig.text_embed_dim` | int | CLIP embed dim (512) |
| `UMPAModelConfig.image_size` | int | Kích thước ảnh đầu vào (1024) |

### Task 1.2 — Viết `config/train_config.py`

| Trường | Kiểu | Mô tả |
|--------|------|-------|
| `PhaseConfig.epochs` | int | Số epoch của phase |
| `PhaseConfig.lambda_con` | float | Trọng số L_con (0.0 ở Phase 1&2) |
| `PhaseConfig.freeze_image_encoder` | bool | Freeze ViT hay không |
| `PhaseConfig.freeze_prompt_encoder` | bool | Freeze PromptEncoder hay không |
| `PhaseConfig.freeze_mask_decoder` | bool | Freeze MaskDecoder hay không |
| `PhaseConfig.lr` | float | Learning rate của phase |
| `TrainConfig.batch_size` | int | Kích thước batch |
| `TrainConfig.K` | int | Số perturbed prompts cho MPCL |
| `TrainConfig.lambda_reg` | float | Trọng số L_reg |
| `TrainConfig.phase1` | PhaseConfig | Prompt warm-up |
| `TrainConfig.phase2` | PhaseConfig | Unified adaptation |
| `TrainConfig.phase3` | PhaseConfig | Consistency phase |

**Giá trị mặc định 3 phase:**

| Phase | epochs | lambda_con | freeze_img | freeze_prompt | freeze_decoder | lr |
|-------|--------|-----------|------------|---------------|----------------|----|
| 1 — Warm-up | 5 | 0.0 | ✅ | ✅ | ✅ | 1e-4 |
| 2 — Adaptation | 5 | 0.0 | ✅ | ❌ | ✅ | 5e-5 |
| 3 — Consistency | 10 | 0.5 | ✅ | ❌ | ❌ | 1e-5 |

### Task 1.3 — Fix `modules/modules.py`

`PromptPerturbation` hiện thiếu `MaskPerturbation` và `TextPerturbation` trong constructor và `forward()`. Bổ sung để wrapper gọi đủ 4 loại perturbation.

### Task 1.4 — Chạy lại tests

```bash
pytest tests/test_consistency_loss.py
pytest tests/test_mudules_mask_per.py
```

Bổ sung test nhanh cho `TextPerturbation` và `UPFE` với dummy tensor.

### ✅ Checklist Ngày 1

- [ ] `config/model_config.py` viết xong
- [ ] `config/train_config.py` viết xong
- [ ] `modules/modules.py` fix wrapper đủ 4 perturbation
- [ ] Tất cả tests pass

---

## Ngày 2 — Dataloader

### Mục tiêu
DataLoader chạy được với Kvasir-SEG / CVC-ClinicDB, trả đúng format cho model.

### Task 2.1 — Viết `data/polyp_dataset.py`

Format thư mục dataset:
```
dataset/
├── images/   ← ảnh .jpg
├── masks/    ← binary mask .png cùng tên
└── split/
    ├── train.txt
    └── val.txt
```

Output mỗi sample:
```python
{
    "image":        Tensor (3, H, W),
    "mask":         Tensor (1, H, W),   # binary GT [0,1]
    "bbox":         Tensor (4,),        # (x1,y1,x2,y2) tính từ mask
    "points":       Tensor (N, 2),      # random sample trong foreground
    "point_labels": Tensor (N,),        # 1 = positive
    "text":         str,                # "polyp" hoặc synonym
}
```

Logic tính bbox: lấy bounding rect của vùng foreground mask.
Logic sample point: random 1 điểm trong vùng mask = 1.

### Task 2.2 — Viết `data/transforms.py`

| Transform | Tham số config | Ghi chú |
|-----------|---------------|---------|
| `Resize` | `image_size=1024` | Bắt buộc cho SAM encoder |
| `RandomHorizontalFlip` | `p=0.5` | Áp cả mask + bbox + point |
| `RandomVerticalFlip` | `p=0.5` | Áp cả mask + bbox + point |
| `ColorJitter` | `brightness, contrast, saturation` | Chỉ áp lên ảnh |
| `Normalize` | `mean, std` theo ImageNet | Chuẩn SAM |

Mỗi transform nhận `TransformConfig` — thêm transform mới không sửa dataset.

### Task 2.3 — Collate function

Xử lý padding khi ảnh trong batch khác kích thước sau augmentation.

### Task 2.4 — Kiểm tra

Load 100 sample liên tiếp, print shape, đảm bảo không crash.

### ✅ Checklist Ngày 2

- [ ] `data/polyp_dataset.py` viết xong
- [ ] `data/transforms.py` viết xong với `TransformConfig`
- [ ] Collate function xử lý đúng padding
- [ ] DataLoader chạy ổn định 100 sample không crash

---

## Ngày 3 — `UMPAModel.forward()`

### Mục tiêu
Module trung tâm kết nối SAM + UPFE, chạy end-to-end với dummy data.

### Task 3.1 — Hoàn thiện `modules/umpa_model.py`

**`__init__`** khởi tạo từ `UMPAModelConfig`:

| Component | Nguồn | Trạng thái khi train |
|-----------|-------|---------------------|
| `image_encoder` (SAM ViT) | Load từ checkpoint | Frozen toàn bộ |
| `prompt_encoder` (SAM PromptEncoder) | Load từ checkpoint | Freeze theo phase |
| `mask_decoder` (SAM MaskDecoder) | Load từ checkpoint | Freeze theo phase |
| `text_projection` (Linear 512→256) | Khởi tạo mới | Luôn trainable |
| `upfe` (UnifiedPromptFusionEncoder) | Khởi tạo mới | Luôn trainable |

**`forward()`** — flow dữ liệu:

```
Step 1: image_emb = image_encoder(image)
        → frozen, không tính grad

Step 2: sparse_emb, dense_emb = prompt_encoder(points, boxes, masks)
        sparse_emb: (B, N, 256)
        dense_emb:  (B, 256, H, W)

Step 3: text_emb = text_projection(clip_emb)   nếu có text
        → (B, 256)

Step 4: Gom embeddings dict:
        {box, point, mask, text} → UPFE

Step 5: Efused = upfe(embeddings)
        → (B, 256)

Step 6: new_sparse = cat([sparse_emb, Efused.unsqueeze(1)], dim=1)
        → (B, N+1, 256)
        ↑ inject Efused vào SAM decoder qua sparse tokens

Step 7: masks = mask_decoder(
            image_emb, image_pe,
            sparse_prompt_emb = new_sparse,
            dense_prompt_emb  = dense_emb,
            ...
        )

Return: masks  (B, 1, H, W)
```

### Task 3.2 — Test shape

```python
model = UMPAModel(cfg)
out = model(image, boxes=..., points=..., masks=..., text_emb=...)
assert out.shape == (B, 1, 256, 256)
```

### Task 3.3 — Kiểm tra gradient

Gradient phải chảy qua `upfe` và `text_projection`. `image_encoder` không có grad.

### ✅ Checklist Ngày 3

- [ ] `umpa_model.__init__()` load đúng SAM components từ checkpoint
- [ ] `umpa_model.forward()` chạy đúng 7 bước flow
- [ ] Output shape `(B, 1, 256, 256)` xác nhận
- [ ] `image_encoder` không có gradient
- [ ] `upfe` và `text_projection` có gradient

---

## Ngày 4 — `LossComposer` + `UMPASAM`

### Mục tiêu
Class cấp cao nhất thực hiện K-prompt loop và tính L_total. Thêm loss mới chỉ cần 1 dòng.

### Task 4.1 — Viết `losses/loss_composer.py`

```python
class LossComposer(nn.Module):
    def __init__(self, weight_dict: Dict[str, float]):
        # {"dice": 1.0, "focal": 1.0, "consistency": 0.5, "reg": 0.01}
        ...

    def forward(self, loss_dict: Dict[str, Tensor]) -> Tensor:
        # Chỉ tổng hợp loss có trong weight_dict và weight != 0
        # Loss nào weight=0 hoặc không có trong dict → bỏ qua
        ...
```

**Tính mở rộng:** Reviewer yêu cầu thêm boundary loss → chỉ thêm `"boundary": 0.3` vào `weight_dict`, không sửa `forward()` của model.

### Task 4.2 — Hoàn thiện `umpa_sam.py`

**`__init__`:** Nhận `UMPAModelConfig` và `LossComposer`, khởi tạo `UMPAModel`, `PromptPerturbation`, `MultiPromptConsistencyLoss`.

**`forward(image, prompts, gt_mask, K, lambda_con)`:**
- `K` và `lambda_con` được truyền từ `PhaseScheduler` — không hard-code
- Sinh K masks từ K perturbed prompts
- Tính `loss_dict = {dice, focal, consistency}`
- Return `(total_loss, loss_dict)`

**`predict(image, prompts)`:**
- Inference mode — không dùng MPPG, không tính loss
- Dùng cho eval

### Task 4.3 — Test backward

```python
loss, info = model(image, prompts, gt_mask, K=3, lambda_con=0.5)
loss.backward()
for name, p in model.upfe.named_parameters():
    assert p.grad is not None
```

### ✅ Checklist Ngày 4

- [ ] `losses/loss_composer.py` viết xong, test với dummy dict
- [ ] `umpa_sam.forward()` sinh K masks và tính đủ 3 loss
- [ ] `umpa_sam.predict()` không dùng MPPG
- [ ] Backward pass không lỗi
- [ ] Grad xác nhận tại UPFE parameters

---

## Ngày 5 — Training script 3-phase

### Mục tiêu
Script train hoàn chỉnh. Logic freeze/unfreeze tách biệt khỏi training loop.

### Task 5.1 — Viết `training/phase_scheduler.py`

| Method | Mô tả |
|--------|-------|
| `get_current_phase(epoch)` | Trả về `PhaseConfig` dựa trên epoch hiện tại |
| `apply_phase(phase)` | Freeze/unfreeze đúng component, update lr |
| `get_lambda_con(epoch)` | Trả về `lambda_con` của phase hiện tại |
| `_set_requires_grad(module, value)` | Helper freeze/unfreeze |

### Task 5.2 — Viết `training/trainer.py`

Training loop sạch — chỉ chứa vòng lặp, không chứa logic freeze:

```
for epoch in range(total_epochs):
    phase    = scheduler.get_current_phase(epoch)
    scheduler.apply_phase(phase)          ← freeze/unfreeze ở đây
    lambda_c = phase.lambda_con

    for batch in train_loader:
        loss, info = model(batch, K=cfg.K, lambda_con=lambda_c)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    val_metrics = evaluate(model, val_loader)
    save_checkpoint(model, epoch, val_metrics)
    log(epoch, phase, info, val_metrics)
```

**Checkpoint:** Lưu sau mỗi epoch, giữ `best_model.pth` theo val Dice cao nhất.

**Logging:** Print `epoch | phase | L_seg | L_con | L_total | val_dice | val_miou`.

### ✅ Checklist Ngày 5

- [ ] `training/phase_scheduler.py` viết xong
- [ ] `training/trainer.py` viết xong, loop tách biệt khỏi freeze logic
- [ ] Checkpoint lưu đúng, `best_model.pth` được cập nhật
- [ ] Chạy 1 epoch nhỏ (5 batch) không crash
- [ ] Log in đúng thông tin từng phase

---

## Ngày 6 — Metrics + Evaluation

### Mục tiêu
Vòng lặp validation trả về Dice và mIoU chuẩn theo benchmark polyp segmentation.

### Task 6.1 — Metrics tận dụng từ SAM3

| Metric | Nguồn | Cần làm |
|--------|-------|---------|
| Dice | `umpt_sam/losses/dice_loss.py` | Wrap lại cho eval |
| mIoU | `sam3/train/loss/loss_fns.py:segment_miou` | Wrap lại |
| Focal Loss | `sam3/train/loss/sigmoid_focal_loss.py` | Dùng thẳng |

### Task 6.2 — Viết `evaluate.py`

```python
@torch.no_grad()
def evaluate(model, val_loader, threshold=0.5) -> Dict[str, float]:
    # Dùng model.predict() — không MPPG, không loss
    # Threshold sigmoid tại 0.5
    # Trả về dict → dễ thêm metric mới
    return {"dice": ..., "miou": ...}
```

### Task 6.3 — Tích hợp vào Trainer

Gọi `evaluate()` sau mỗi epoch, dùng val Dice để lưu best checkpoint.

### ✅ Checklist Ngày 6

- [ ] `evaluate.py` viết xong
- [ ] Dice và mIoU ra số hợp lý (không NaN, không 0)
- [ ] Tích hợp vào Trainer, best checkpoint lưu theo val Dice
- [ ] Thêm metric mới chỉ cần thêm vào dict trả về

---

## Ngày 7 — Smoke test + Debug

### Mục tiêu
Toàn bộ pipeline chạy end-to-end từ load data → train 3 epoch → eval → checkpoint.

### Task 7.1 — Smoke test

```bash
python -m umpt_sam.training.trainer \
    --data_root /path/to/Kvasir-SEG \
    --sam_checkpoint /path/to/sam3.pth \
    --phase1_epochs 1 \
    --phase2_epochs 1 \
    --phase3_epochs 1 \
    --batch_size 2
```

### Task 7.2 — Checklist các điểm hay lỗi

| Điểm nguy hiểm | Vấn đề thường gặp | Cách kiểm tra |
|---------------|------------------|--------------|
| Inject Efused vào sparse_emb | Shape mismatch dim 1 | Print shape trước `cat` |
| Frozen module chặn gradient | Grad = None ở UPFE | `assert p.grad is not None` |
| MPCL khi K=1 | Không có cặp i≠j → NaN | Guard `if K < 2: return 0` |
| TextPerturbation khi text=None | AttributeError | Check None trước khi process |
| Mask resize về đúng GT size | Shape không khớp khi tính loss | Interpolate mask về GT size |

### Task 7.3 — Fix và retest

Buổi chiều Ngày 7 dành hoàn toàn để fix bug từ smoke test.

### ✅ Checklist Ngày 7

- [ ] Smoke test 3 epoch không crash
- [ ] Loss giảm dần qua 3 phase
- [ ] Val Dice được print đúng cuối mỗi epoch
- [ ] Checkpoint `best_model.pth` được lưu
- [ ] Tất cả 5 điểm nguy hiểm đã kiểm tra

---

## Tóm tắt tiến độ toàn tuần

| Ngày | Việc chính | File chính | Độ khó | Hoàn thành |
|------|-----------|------------|--------|-----------|
| 1 | Config dataclass + fix MPPG | `config/*.py`, `modules/modules.py` | Dễ | ☐ |
| 2 | Dataloader polyp | `data/polyp_dataset.py`, `data/transforms.py` | Dễ–TB | ☐ |
| 3 | UMPAModel forward | `modules/umpa_model.py` | **Khó nhất** | ☐ |
| 4 | LossComposer + UMPASAM | `losses/loss_composer.py`, `umpa_sam.py` | Trung bình | ☐ |
| 5 | Training 3-phase | `training/phase_scheduler.py`, `training/trainer.py` | Trung bình | ☐ |
| 6 | Metrics + Eval | `evaluate.py` | Dễ | ☐ |
| 7 | Smoke test + Debug | Tất cả | Trung bình | ☐ |

---

## Tính mở rộng khi có yêu cầu mới từ reviewer

| Yêu cầu reviewer | Sửa ở đâu | Ước tính |
|-----------------|-----------|---------|
| Đổi λ_con | `config/train_config.py` Phase3 | 1 dòng |
| Thêm boundary loss | `losses/loss_composer.py` weight_dict | ~10 dòng |
| Thêm augmentation mới | `data/transforms.py` | ~15 dòng |
| Thêm dataset mới | Subclass `PolypDataset` | ~30 dòng |
| Đổi K số prompt | `config/train_config.py` | 1 dòng |
| Thêm metric mới | `evaluate.py` dict | ~10 dòng |
| Đổi perturbation strategy | Subclass `PromptPerturbation` | ~20 dòng |
