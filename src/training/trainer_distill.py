# Training loop with teacher-student
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.distillation.loss import DistillationLoss
from src.utils.logger import init_wandb

class DistillationTrainer:
    def __init__(self, cfg, student, teacher, train_dataset):
        self.cfg = cfg
        self.student = student.to(cfg.train.device)
        self.teacher = teacher.to(cfg.train.device).eval()
        self.criterion = DistillationLoss(alpha=cfg.train.feature_loss_weight)
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.student.parameters()),
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=cfg.train.batch_size, shuffle=True
        )
        init_wandb(cfg)

    def train(self):
        for epoch in range(1, self.cfg.train.epochs + 1):
            self.student.train()
            epoch_loss = 0.0
            for i, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch}")):
                img = batch["image"].to(self.cfg.train.device)
                prompt = batch["prompt"]
                gt_mask = batch["mask"].to(self.cfg.train.device)

                # Teacher forward (no grad)
                with torch.no_grad():
                    teacher_out = self.teacher(img, **prompt)
                    teacher_mask = teacher_out[0]
                    teacher_feat = self.teacher.get_image_features(img)

                # Student forward
                student_out = self.student(img, **prompt)
                student_mask = student_out[0]
                student_feat = self.student.get_image_features(img)

                # Loss
                loss = self.criterion(student_feat, teacher_feat, student_mask, teacher_mask)

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

                if i % self.cfg.train.log_every == 0:
                    wandb.log({
                        "loss": loss.item(),
                        "feat_loss": loss.item() * self.cfg.train.feature_loss_weight,
                        "mask_loss": loss.item() * (1 - self.cfg.train.feature_loss_weight)
                    })

            print(f"Epoch {epoch} Loss: {epoch_loss / len(self.train_loader):.4f}")
            torch.save(self.student.state_dict(), f"{self.cfg.paths.checkpoints}/minisam_epoch{epoch}.pth")