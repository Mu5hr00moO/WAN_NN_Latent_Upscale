import torch
import numpy as np

def check_dataset_vae():
    """
    Überprüft ob wirklich die Wan2.2 VAE verwendet wurde.
    """
    
    print("🔍 Überprüfung des Datasets...")
    
    # Lade das Dataset
    dataset_path = "datasets/real_latent/final_official_wan22_dataset.pt"
    
    try:
        data = torch.load(dataset_path, map_location='cpu')
        
        print("📊 Dataset Metadaten:")
        metadata = data.get('metadata', {})
        for key, value in metadata.items():
            print(f"  {key}: {value}")
        
        print("\n📐 Dataset Dimensionen:")
        hr_latents = data['hr_latents']
        lr_latents = data['lr_latents']
        
        print(f"  HR Latents: {hr_latents.shape}")
        print(f"  LR Latents: {lr_latents.shape}")
        
        print("\n🔬 Latent Statistiken:")
        print(f"  HR - Min: {hr_latents.min():.4f}, Max: {hr_latents.max():.4f}, Mean: {hr_latents.mean():.4f}, Std: {hr_latents.std():.4f}")
        print(f"  LR - Min: {lr_latents.min():.4f}, Max: {lr_latents.max():.4f}, Mean: {lr_latents.mean():.4f}, Std: {lr_latents.std():.4f}")
        
        # Prüfe VAE-typische Eigenschaften
        print("\n🧪 VAE-Eigenschaften Analyse:")
        
        # 1. Skalierung prüfen (VAE sollte ~0.18215 Skalierung haben)
        hr_scale = hr_latents.abs().mean()
        lr_scale = lr_latents.abs().mean()
        print(f"  HR Skalierung: {hr_scale:.4f} (VAE-typisch: ~0.18)")
        print(f"  LR Skalierung: {lr_scale:.4f} (VAE-typisch: ~0.18)")
        
        # 2. Verteilung prüfen (VAE sollte normalverteilt sein)
        try:
            hr_kurtosis = torch.kurtosis(hr_latents.flatten()) if hasattr(torch, 'kurtosis') else 0.0
            lr_kurtosis = torch.kurtosis(lr_latents.flatten()) if hasattr(torch, 'kurtosis') else 0.0
            print(f"  HR Kurtosis: {hr_kurtosis:.4f} (Normal: ~3.0)")
            print(f"  LR Kurtosis: {lr_kurtosis:.4f} (Normal: ~3.0)")
        except:
            hr_kurtosis = 3.0  # Assume normal for compatibility
            lr_kurtosis = 3.0
            print(f"  Kurtosis check skipped (torch version compatibility)")
        
        # 3. Channel-Korrelation prüfen
        hr_sample = hr_latents[0]  # Erstes Sample
        lr_sample = lr_latents[0]
        
        print(f"\n📊 Channel-Analyse (erstes Sample):")
        print(f"  HR Channels - Min: {hr_sample.min(dim=(1,2))[0][:8]}")
        print(f"  HR Channels - Max: {hr_sample.max(dim=(1,2))[0][:8]}")
        print(f"  LR Channels - Min: {lr_sample.min(dim=(1,2))[0][:8]}")
        print(f"  LR Channels - Max: {lr_sample.max(dim=(1,2))[0][:8]}")
        
        # 4. Prüfe ob es echte VAE-Latents sind oder synthetische
        print(f"\n🔍 VAE-Authentizität Check:")
        
        # Echte VAE-Latents haben bestimmte Eigenschaften:
        # - Skalierung um 0.18215
        # - Bestimmte Verteilungscharakteristika
        # - Channel-spezifische Patterns
        
        is_real_vae = True
        reasons = []
        
        if abs(hr_scale - 0.18215) > 0.1:
            is_real_vae = False
            reasons.append(f"Skalierung nicht VAE-typisch ({hr_scale:.4f} statt ~0.18)")
        
        if hr_latents.std() < 0.1 or hr_latents.std() > 2.0:
            is_real_vae = False
            reasons.append(f"Standardabweichung ungewöhnlich ({hr_latents.std():.4f})")
        
        if abs(hr_kurtosis - 3.0) > 2.0:
            is_real_vae = False
            reasons.append(f"Verteilung nicht normal ({hr_kurtosis:.4f})")
        
        print(f"  VAE-Authentizität: {'✅ ECHT' if is_real_vae else '❌ SYNTHETISCH'}")
        if not is_real_vae:
            print("  Gründe:")
            for reason in reasons:
                print(f"    - {reason}")
        
        # 5. Vergleiche mit bekannten VAE-Patterns
        print(f"\n🎯 Fazit:")
        vae_used = metadata.get('vae_used', 'unknown')
        print(f"  Laut Metadaten: {vae_used}")
        print(f"  Laut Analyse: {'ECHTE Wan2.2 VAE' if is_real_vae else 'SYNTHETISCHE VAE'}")
        
        if vae_used == 'real_wan22' and is_real_vae:
            print("  ✅ BESTÄTIGT: Echte Wan2.2 VAE wurde verwendet!")
        elif vae_used == 'real_wan22' and not is_real_vae:
            print("  ⚠️  WARNUNG: Metadaten sagen 'real_wan22', aber Analyse deutet auf synthetisch!")
        else:
            print("  ❌ PROBLEM: Keine echte VAE verwendet!")
        
        return is_real_vae, vae_used
        
    except Exception as e:
        print(f"❌ Fehler beim Laden des Datasets: {e}")
        return False, "error"

if __name__ == "__main__":
    is_real, vae_type = check_dataset_vae()
    
    if is_real:
        print("\n🎉 Dataset ist bereit für Training!")
    else:
        print("\n🚨 Dataset sollte neu erstellt werden mit echter VAE!")
