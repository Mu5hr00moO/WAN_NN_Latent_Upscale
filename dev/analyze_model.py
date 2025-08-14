import torch

# Analysiere unser Wan2.2 Modell genauer
model_path = 'models/wan2.2_resizer_best.pt'
state_dict = torch.load(model_path, map_location='cpu')

print('Wan2.2 Model Layer Analysis:')
print('First few layers:')
for i, (key, tensor) in enumerate(list(state_dict.items())[:10]):
    print(f'{key}: {tensor.shape}')

print('\nLast few layers:')
for key, tensor in list(state_dict.items())[-5:]:
    print(f'{key}: {tensor.shape}')

print(f'\nTotal layers: {len(state_dict)}')

# Check if it matches our WanLatentResizer architecture
expected_keys = [
    'encoder.0.weight', 'encoder.0.bias',
    'encoder.2.weight', 'encoder.2.bias',
    'encoder.4.weight', 'encoder.4.bias',
    'upsampler.0.weight', 'upsampler.0.bias',
    'upsampler.2.weight', 'upsampler.2.bias',
    'upsampler.4.weight', 'upsampler.4.bias',
    'skip_conv.weight', 'skip_conv.bias'
]

print('\nChecking if it matches our WanLatentResizer:')
matches = all(key in state_dict for key in expected_keys)
print(f'Matches WanLatentResizer: {matches}')

if matches:
    print('Architecture details:')
    print(f'encoder.0 (first conv): {state_dict["encoder.0.weight"].shape}')
    print(f'encoder.2 (second conv): {state_dict["encoder.2.weight"].shape}')
    print(f'encoder.4 (third conv): {state_dict["encoder.4.weight"].shape}')
    print(f'upsampler.0 (transpose): {state_dict["upsampler.0.weight"].shape}')
    print(f'skip_conv: {state_dict["skip_conv.weight"].shape}')
