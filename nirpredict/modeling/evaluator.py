import torch
import torchmetrics

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from nirpredict.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate(nir_model, test_nir_loader, criterion, activation, device):
    nir_model.eval()

    test_loss = 0.0
    mae_metric = torchmetrics.MeanAbsoluteError().to(device)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    with torch.no_grad():
        for rgb_test_batch, nir_test_batch in test_nir_loader:
            rgb_test_batch = rgb_test_batch.to(device)
            nir_test_batch = nir_test_batch.to(device)

            logits = nir_model(rgb_test_batch)

            if activation == 'sigmoid':
                nir_pred = torch.sigmoid(logits)

            elif activation == 'tanh':
                nir_pred = (torch.tanh(logits) + 1) / 2  # Rescale to 0-1

            elif activation == 'sigmoid-scaled':
                scaling_factor = 5
                scaled_logits = logits * scaling_factor
                nir_pred = torch.sigmoid(scaled_logits)

            elif activation == 'sigmoid-normalized':
                nir_pred = torch.sigmoid(logits)  # Forward pass
                nir_pred = (nir_pred - nir_pred.min()) / (nir_pred.max() - nir_pred.min())

            else: # default activation 'relu'
                nir_pred = torch.clamp(torch.relu(logits), 0, 1)

            combined_loss, mse_loss, ssim_loss = criterion(nir_pred, nir_test_batch)
            test_loss += combined_loss.item()

            mae_metric.update(nir_pred, nir_test_batch)
            psnr_metric.update(nir_pred, nir_test_batch)
            ssim_metric.update(nir_pred, nir_test_batch)

    test_loss /= len(test_nir_loader)
    avg_mae = mae_metric.compute().item()

    psnr_value = psnr_metric.compute()
    avg_psnr = psnr_value.mean().item() if isinstance(psnr_value, torch.Tensor) else sum(psnr_value) / len(psnr_value)

    ssim_value = ssim_metric.compute()
    avg_ssim = ssim_value.mean().item() if isinstance(ssim_value, torch.Tensor) else sum(ssim_value) / len(ssim_value)

    mae_metric.reset()
    psnr_metric.reset()
    ssim_metric.reset()

    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Mean Absolute Error (MAE): {avg_mae:.4f}")
    logger.info(f"Peak Signal-to-Noise Ratio (PSNR): {avg_psnr:.4f} dB")
    logger.info(f"Structural Similarity Index (SSIM): {avg_ssim:.4f}")