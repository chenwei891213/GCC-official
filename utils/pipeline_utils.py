import torch
import torch.nn.functional as F

def create_laplacian_pyramid_kernel(device):
    """Create Gaussian kernel for Laplacian pyramid
    
    Args:
        device: Computation device
        
    Returns:
        Gaussian kernel tensor
    """
    
    kernel = torch.tensor([
        [0.0625, 0.125, 0.0625],
        [0.125, 0.25, 0.125],
        [0.0625, 0.125, 0.0625]
    ], device=device).unsqueeze(0).unsqueeze(0)
    
    return kernel

def apply_laplacian_highpass(latents, level=1):
    """Apply Laplacian pyramid high-frequency extraction to VAE latents"""
    device = latents.device
    batch_size, channels, height, width = latents.size()
    
    # Create Gaussian kernel for grouped convolution
    base_kernel = create_laplacian_pyramid_kernel(device).to(dtype=latents.dtype)
    conv_kernel = base_kernel.repeat(channels, 1, 1, 1)  # [C, 1, 3, 3]
    
    highpass_latents = torch.zeros_like(latents)
    current_tensor = latents
    
    for l in range(level):
        # Gaussian blur (independent per channel)
        blurred = F.conv2d(
            current_tensor, 
            conv_kernel, 
            padding=1, 
            groups=channels
        )
        
        # Calculate high-frequency component
        highfreq = current_tensor - blurred
        
        # Downsample for next level
        current_tensor = F.avg_pool2d(blurred, kernel_size=2)
        
        # Accumulate high-frequency components
        if l == 0:
            highpass_latents = highfreq
        else:
            # Upsample back to original resolution and add
            highpass_latents += F.interpolate(
                highfreq, 
                size=(height, width), 
                mode='bilinear', 
                align_corners=False
            )
    
    return highpass_latents

def encode_vae_mean(vae, org_image):
    with torch.no_grad():  
        h = vae.encoder(org_image)
        moments = vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        # scale latent
        latent = mean * vae.config.scaling_factor
    return latent
