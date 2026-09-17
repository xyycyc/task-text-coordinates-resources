"""Frozen FP32 image/text encoders with explicit endpoint features."""
from pathlib import Path
import torch
import torch.nn.functional as F


class _ProcessorTransform:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, image):
        return self.processor(images=image.convert("RGB"), return_tensors="pt")["pixel_values"][0]


class Encoder:
    def __init__(self, family, classnames, template, device="cpu", checkpoint=None, download_root=None):
        from torchvision import transforms as T
        self.family, self.device = family, torch.device(device)
        self.prompts = [template.format(name.replace("_", " ")) for name in classnames]
        self.captured, self.hooks = {}, []
        if family == "clip":
            import clip
            self.model, _ = clip.load(checkpoint or "ViT-B/16", device=self.device,
                                      jit=False, download_root=download_root)
            self.model.float().eval().requires_grad_(False)
            self.w0 = self.model.visual.proj.detach().cpu().clone()
            if self.w0.shape != (768, 512):
                raise ValueError("Expected OpenAI CLIP ViT-B/16")
            self.hooks.append(self.model.visual.ln_post.register_forward_hook(
                lambda module, args, output: self.captured.update(h=output.detach())))
            with torch.inference_mode():
                text = self.model.encode_text(clip.tokenize(self.prompts).to(self.device))
            self.scale = 100.
            self.checkpoint_name = "OpenAI CLIP ViT-B/16"
            mode = T.InterpolationMode.BICUBIC
            mean, std = (.48145466, .4578275, .40821073), (.26862954, .26130258, .27577711)
            evaluation = [T.Resize(224, interpolation=mode), T.CenterCrop(224)]
        elif family == "siglip2":
            from transformers import AutoImageProcessor, AutoTokenizer, SiglipModel
            source = checkpoint or "google/siglip2-base-patch16-224"
            local = Path(source).is_dir()
            processor = AutoImageProcessor.from_pretrained(source, local_files_only=local,
                                                          cache_dir=download_root)
            tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=local,
                                                     cache_dir=download_root, use_fast=True)
            self.prompts = [s.lower() for s in self.prompts]
            tokens = tokenizer(self.prompts, padding="max_length", truncation=True,
                               max_length=64, return_attention_mask=False, return_tensors="pt")
            self.model = SiglipModel.from_pretrained(source, local_files_only=local,
                cache_dir=download_root, torch_dtype=torch.float32).to(self.device).eval().requires_grad_(False)
            head = self.model.vision_model.head
            self.w0 = head.mlp.fc2.weight.detach().T.cpu().contiguous().clone()
            if self.w0.shape != (3072, 768):
                raise ValueError("Expected fixed-resolution SigLIP 2 Base/16 at 224 pixels")
            self.hooks.append(head.mlp.fc2.register_forward_pre_hook(
                lambda module, args: self.captured.update(h=args[0].detach())))
            with torch.inference_mode():
                text = self.model.text_model(input_ids=tokens["input_ids"].to(self.device)).pooler_output
            self.scale = float(self.model.logit_scale.detach().exp())
            self.checkpoint_name = "google/siglip2-base-patch16-224"
            mode = T.InterpolationMode.BILINEAR
            mean = std = (.5, .5, .5)
            evaluation = [T.Resize((224, 224), interpolation=mode)]
        else:
            raise ValueError("Encoder must be clip or siglip2")
        self.text = F.normalize(text.float(), dim=-1).cpu()
        self.train_transform = T.Compose([
            T.RandomResizedCrop(224, scale=(.5, 1.), ratio=(.75, 4/3), interpolation=mode),
            T.RandomHorizontalFlip(.5), T.ToTensor(), T.Normalize(mean, std)])
        self.eval_transform = T.Compose(evaluation + [T.ToTensor(), T.Normalize(mean, std)])
        if family == "siglip2":
            self.eval_transform = _ProcessorTransform(processor)

    @torch.inference_mode()
    def encode(self, images):
        self.captured.clear()
        pixels = images.to(self.device, dtype=torch.float32)
        if self.family == "clip":
            z0 = self.model.encode_image(pixels)
            h = self.captured.pop("h")
        else:
            z0 = self.model.vision_model(pixel_values=pixels).pooler_output
            h = self.captured.pop("h")[:, 0]
        if not torch.isfinite(z0).all() or not torch.isfinite(h).all():
            raise FloatingPointError("Nonfinite encoder output")
        return h.float().cpu().contiguous(), z0.float().cpu().contiguous()

    def close(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.model = None
