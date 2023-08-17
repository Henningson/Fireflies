import torch
import utils_torch

class Camera:
    id = 0

    MITSUBA_CAMERA_KEYS = {
        'fov': 'x_fov',
        'to_world': 'to_world',

        
        }


    def __init__(self, to_world: torch.tensor, fov: float, near_clip: float = 0.01, far_clip: float = 1000.0, device: torch.cuda.device = torch.device("cuda")):
        self.device = device
        
        self._to_world = to_world.to(self.device)
        self._origin = self._to_world[0:3, 3]
        self._perspective = utils_torch.build_projection_matrix(fov, near_clip, far_clip).to(self.device)
        self._near_clip = near_clip
        self._far_clip = far_clip
        self._fov = fov
        
        self._mitsuba_key = self.generate_mitsuba_key()
        Camera.id += 1


    def mitsuba_key(self, key: str):
        return self._mitsuba_key + "." + Camera.MITSUBA_CAMERA_KEYS[key]

    def mitsuba_base_key(self) -> str:
        return self._mitsuba_key

    def near_clip(self) -> float:
        return self._near_clip
    
    def generate_mitsuba_key(self) -> str:
        if Camera.id == 0:
            return "PerspectiveCamera"
        
        return "PerspectiveCamera_{0}".format(id)

    def far_clip(self) -> float:
        return self._far_clip
    
    def origin(self) -> torch.tensor:
        return self._origin
    
    def fov(self) -> torch.tensor:
        return self._fov
    
    def to_world(self) -> torch.tensor:
        return self._to_world