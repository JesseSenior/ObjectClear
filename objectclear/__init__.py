from .models import CLIPImageEncoder, PostfuseModule
from .pipelines import ObjectClearPipeline
from .utils import attention_guided_fusion, resize_by_short_side
import torch
from typing import Optional
from PIL import Image


class ObjectClear:
    """
    High-level API for ObjectClear object removal.
    
    This class simplifies the usage of ObjectClearPipeline by providing
    a convenient interface for object removal from images.
    
    Example:
        >>> from objectclear import ObjectClear
        >>> from PIL import Image
        >>> 
        >>> # Initialize
        >>> oc = ObjectClear(device='cuda', use_fp16=True)
        >>> 
        >>> # Load image and mask
        >>> image = Image.open('input.jpg')
        >>> mask = Image.open('mask.png').convert('L')
        >>> 
        >>> # Perform inference
        >>> result = oc.inference(image, mask)
        >>> output = result.images[0]
        >>> output.save('output.jpg')
    """
    
    def __init__(
        self,
        device: str = "cuda",
        cache_dir: Optional[str] = None,
        use_fp16: bool = False,
        seed: int = 42,
        steps: int = 20,
        guidance_scale: float = 2.5,
        enable_agf: bool = True,
        model_name: str = "jixin0101/ObjectClear",
    ):
        """
        Initialize ObjectClear instance.
        
        Args:
            device: Device to run inference on ("cuda" or "cpu")
            cache_dir: Path to model directory for caching weights
            use_fp16: Use float16 precision for inference
            seed: Random seed for reproducibility
            steps: Default number of diffusion inference steps
            guidance_scale: Default classifier-free guidance scale
            enable_agf: Enable Attention Guided Fusion
            model_name: Pretrained model name
        """
        # Set device
        if device == "cuda":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Configure dtype
        self.torch_dtype = torch.float16 if use_fp16 else torch.float32
        self.variant = "fp16" if use_fp16 else None
        
        # Initialize generator
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Store default parameters
        self.default_steps = steps
        self.default_guidance_scale = guidance_scale
        self.enable_agf = enable_agf
        
        # Load pipeline
        self.pipeline = ObjectClearPipeline.from_pretrained_with_custom_modules(
            model_name,
            torch_dtype=self.torch_dtype,
            apply_attention_guided_fusion=enable_agf,
            cache_dir=cache_dir,
            variant=self.variant,
        )
        self.pipeline.to(self.device)
    
    def inference(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: Optional[str] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        generator: Optional[torch.Generator] = None,
        return_attn_map: bool = False,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ):
        """
        Perform object removal inference.
        
        Args:
            image: Input PIL Image (RGB)
            mask: Binary mask PIL Image (L, white=remove, black=keep)
            prompt: Text prompt (default: "remove the instance of object")
            num_inference_steps: Number of diffusion steps (overrides default)
            guidance_scale: CFG scale (overrides default)
            generator: torch.Generator (overrides default)
            return_attn_map: Return attention maps
            height: Image height (auto-detected if None)
            width: Image width (auto-detected if None)
        
        Returns:
            ObjectClearPipelineOutput with:
                - images: List of PIL images
                - attns: Optional attention maps
        """
        # Set default prompt
        if prompt is None:
            prompt = "remove the instance of object"
        
        # Resize image and mask (shorter side = 512)
        image_resized = resize_by_short_side(image, 512, resample=Image.BICUBIC)
        mask_resized = resize_by_short_side(mask, 512, resample=Image.NEAREST)
        
        # Auto-detect dimensions
        if height is None or width is None:
            width, height = image_resized.size
        
        # Use provided parameters or defaults
        steps = num_inference_steps if num_inference_steps is not None else self.default_steps
        scale = guidance_scale if guidance_scale is not None else self.default_guidance_scale
        gen = generator if generator is not None else self.generator
        
        # Run inference
        result = self.pipeline(
            prompt=prompt,
            image=image_resized,
            mask_image=mask_resized,
            generator=gen,
            num_inference_steps=steps,
            guidance_scale=scale,
            height=height,
            width=width,
            return_attn_map=return_attn_map,
        )
        
        return result
    
    def to(self, device):
        """
        Move the pipeline to a different device.
        
        Args:
            device: Target device (str or torch.device)
        """
        self.device = torch.device(device)
        self.pipeline.to(self.device)


__all__ = [
    "CLIPImageEncoder",
    "PostfuseModule", 
    "ObjectClearPipeline",
    "ObjectClear",
    "attention_guided_fusion",
    "resize_by_short_side",
]