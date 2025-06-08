import torch
from diffusers import DDIMScheduler, UNet2DConditionModel
from .pipeline import OneStepLaplacianInpaintPipeline

class DiffusionInpaintPipeline:
    """Test inference pipeline wrapper"""
    
    def __init__(self, model_path, device="cuda"):
        self.model_path = model_path
        self.device = device
        self.scheduler = None
        self.unet = None
        self.pipeline = None
        self.load_model()
        
    def load_model(self, base_model="stabilityai/stable-diffusion-2-inpainting"):
        """Load model"""
        # Set up scheduler
        self.scheduler = DDIMScheduler.from_pretrained(
            base_model,
            subfolder="scheduler",
            timestep_spacing="trailing",
            prediction_type="v_prediction"
        )
        self.scheduler.set_timesteps(1)
        
        # Load UNet model
        self.unet = UNet2DConditionModel.from_pretrained(
            self.model_path,
            subfolder="unet",
            torch_dtype=torch.float16
        )
        
        # Create pipeline
        self.pipeline = OneStepLaplacianInpaintPipeline.from_pretrained(
            base_model,     
            torch_dtype=torch.float16,
            scheduler=self.scheduler,
            unet=self.unet
        )
        self.pipeline.to(self.device)
        return self.pipeline
    
    def run_inference(self, images, masks, prompts, seed=42):
        """Run inference"""
        return self.pipeline(
            prompts,
            image=images,
            generator=torch.Generator(device="cpu").manual_seed(seed),
            num_inference_steps=1,
            mask_image=masks,
            guidance_scale=0,
        )[0]