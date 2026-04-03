from .clip_encoder import CLIPVisionTower
from .multipath_encoder_wapper import MultiPathCLIPVisionTower

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(
        vision_tower_cfg,
        "mm_vision_tower",
        getattr(vision_tower_cfg, "vision_tower", None),
    )
    is_multipath_encoder = getattr(vision_tower_cfg, 'is_multipath_encoder', False)
    if is_multipath_encoder:
        print("build MultiPathCLIPVisionTower ")
        return MultiPathCLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
        
    elif (
        vision_tower.startswith("openai")
        or vision_tower.startswith("laion")
        or "clip" in vision_tower
    ):
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f"Unknown vision tower: {vision_tower}")
