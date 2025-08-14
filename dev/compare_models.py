import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from latent_resizer import WanLatentResizer
from retrain_with_real_data import RealLatentDataset
from sklearn.metrics import mean_squared_error
import seaborn as sns

def calculate_metrics(pred, target):
    """Calculate various image quality metrics."""
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    
    # MSE
    mse = mean_squared_error(target_np.flatten(), pred_np.flatten())
    
    # PSNR
    psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
    
    # SSIM (simplified version)
    def ssim_channel(x, y):
        mu_x = np.mean(x)
        mu_y = np.mean(y)
        sigma_x = np.var(x)
        sigma_y = np.var(y)
        sigma_xy = np.mean((x - mu_x) * (y - mu_y))
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / \
               ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2))
        return ssim
    
    # Average SSIM across channels
    ssim_values = []
    for c in range(pred_np.shape[1]):
        ssim_c = ssim_channel(pred_np[0, c], target_np[0, c])
        ssim_values.append(ssim_c)
    ssim = np.mean(ssim_values)
    
    return mse, psnr, ssim

def compare_models():
    """Compare old synthetic model vs new real data model."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running comparison on: {device}")
    
    # Load test dataset
    try:
        dataset = RealLatentDataset()
        # Use last 100 samples for testing
        test_indices = list(range(len(dataset) - 100, len(dataset)))
        test_data = torch.utils.data.Subset(dataset, test_indices)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    except FileNotFoundError:
        print("❌ Real dataset not found! Please run create_real_dataset.py first")
        return
    
    # Load models
    models = {}
    
    # Original synthetic model
    if Path("models/wan2.2_resizer_best.pt").exists():
        model_synthetic = WanLatentResizer(in_channels=16, out_channels=16, hidden_dim=256)
        model_synthetic.load_state_dict(torch.load("models/wan2.2_resizer_best.pt", map_location=device))
        model_synthetic = model_synthetic.to(device).eval()
        models['Synthetic (Old)'] = model_synthetic
        print("✅ Loaded synthetic model")
    
    # New real data model
    if Path("models/wan2.2_resizer_real_best.pt").exists():
        model_real = WanLatentResizer(in_channels=16, out_channels=16, hidden_dim=256)
        model_real.load_state_dict(torch.load("models/wan2.2_resizer_real_best.pt", map_location=device))
        model_real = model_real.to(device).eval()
        models['Real Data (New)'] = model_real
        print("✅ Loaded real data model")
    else:
        print("❌ Real data model not found! Please train it first with retrain_with_real_data.py")
    
    if not models:
        print("❌ No models found to compare!")
        return
    
    # Run comparison
    results = {name: {'mse': [], 'psnr': [], 'ssim': []} for name in models.keys()}
    results['Bilinear'] = {'mse': [], 'psnr': [], 'ssim': []}
    
    print(f"\n🔍 Testing on {len(test_loader)} samples...")
    
    with torch.no_grad():
        for i, (lr_input, hr_target) in enumerate(test_loader):
            if i >= 50:  # Test on 50 samples
                break
                
            lr_input = lr_input.to(device)
            hr_target = hr_target.to(device)
            
            # Bilinear baseline
            bilinear = F.interpolate(lr_input, size=hr_target.shape[-2:], mode='bilinear', align_corners=False)
            mse, psnr, ssim = calculate_metrics(bilinear, hr_target)
            results['Bilinear']['mse'].append(mse)
            results['Bilinear']['psnr'].append(psnr)
            results['Bilinear']['ssim'].append(ssim)
            
            # Test each model
            for name, model in models.items():
                pred = model(lr_input)
                mse, psnr, ssim = calculate_metrics(pred, hr_target)
                results[name]['mse'].append(mse)
                results[name]['psnr'].append(psnr)
                results[name]['ssim'].append(ssim)
    
    # Calculate averages
    print(f"\n📊 **COMPARISON RESULTS**")
    print("=" * 60)
    
    for name, metrics in results.items():
        avg_mse = np.mean(metrics['mse'])
        avg_psnr = np.mean(metrics['psnr'])
        avg_ssim = np.mean(metrics['ssim'])
        
        print(f"\n**{name}:**")
        print(f"  MSE:  {avg_mse:.6f}")
        print(f"  PSNR: {avg_psnr:.2f} dB")
        print(f"  SSIM: {avg_ssim:.4f}")
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # MSE comparison
    mse_data = [results[name]['mse'] for name in results.keys()]
    axes[0, 0].boxplot(mse_data, labels=list(results.keys()))
    axes[0, 0].set_title('MSE Comparison (Lower is Better)')
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # PSNR comparison
    psnr_data = [results[name]['psnr'] for name in results.keys()]
    axes[0, 1].boxplot(psnr_data, labels=list(results.keys()))
    axes[0, 1].set_title('PSNR Comparison (Higher is Better)')
    axes[0, 1].set_ylabel('PSNR (dB)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # SSIM comparison
    ssim_data = [results[name]['ssim'] for name in results.keys()]
    axes[1, 0].boxplot(ssim_data, labels=list(results.keys()))
    axes[1, 0].set_title('SSIM Comparison (Higher is Better)')
    axes[1, 0].set_ylabel('SSIM')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Average metrics bar chart
    methods = list(results.keys())
    avg_metrics = {
        'MSE': [np.mean(results[name]['mse']) for name in methods],
        'PSNR': [np.mean(results[name]['psnr']) for name in methods],
        'SSIM': [np.mean(results[name]['ssim']) for name in methods]
    }
    
    x = np.arange(len(methods))
    width = 0.25
    
    axes[1, 1].bar(x - width, avg_metrics['MSE'], width, label='MSE', alpha=0.7)
    axes[1, 1].bar(x, [p/10 for p in avg_metrics['PSNR']], width, label='PSNR/10', alpha=0.7)
    axes[1, 1].bar(x + width, avg_metrics['SSIM'], width, label='SSIM', alpha=0.7)
    
    axes[1, 1].set_title('Average Metrics Comparison')
    axes[1, 1].set_ylabel('Normalized Values')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(methods, rotation=45)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calculate improvements
    if 'Real Data (New)' in results and 'Synthetic (Old)' in results:
        print(f"\n🚀 **IMPROVEMENT ANALYSIS**")
        print("=" * 40)
        
        old_mse = np.mean(results['Synthetic (Old)']['mse'])
        new_mse = np.mean(results['Real Data (New)']['mse'])
        mse_improvement = ((old_mse - new_mse) / old_mse) * 100
        
        old_psnr = np.mean(results['Synthetic (Old)']['psnr'])
        new_psnr = np.mean(results['Real Data (New)']['psnr'])
        psnr_improvement = ((new_psnr - old_psnr) / old_psnr) * 100
        
        old_ssim = np.mean(results['Synthetic (Old)']['ssim'])
        new_ssim = np.mean(results['Real Data (New)']['ssim'])
        ssim_improvement = ((new_ssim - old_ssim) / old_ssim) * 100
        
        print(f"MSE Improvement:  {mse_improvement:+.1f}%")
        print(f"PSNR Improvement: {psnr_improvement:+.1f}%")
        print(f"SSIM Improvement: {ssim_improvement:+.1f}%")
        
        if ssim_improvement > 10:
            print(f"\n🎉 Significant improvement with real data training!")
        elif ssim_improvement > 0:
            print(f"\n✅ Moderate improvement with real data training")
        else:
            print(f"\n⚠️  Real data model needs more training or different architecture")
    
    print(f"\n📈 Comparison plots saved as: model_comparison.png")

if __name__ == "__main__":
    compare_models()
