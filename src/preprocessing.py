

import numpy as np
import cv2
import pydicom
import pywt
from typing import Type
from abc import abstractmethod, ABC
import numpy as np
from src.utils.yaml_config_reader import CONFIG
from cv2.typing import Size
import copy

def load_dicom_pixels(path: str) -> np.ndarray:
    """Extract raw pixel array from a DICOM file. """
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return arr




class ImagePreprocessing(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def resize(self, input_image: np.ndarray) -> np.ndarray:
        pass

class NaiveResize(ImagePreprocessing):
    def __init__(self, size: int = CONFIG.target_size) -> None:
        super().__init__()
        self.size = size
    
    def resize(self, input_image: np.ndarray) -> np.ndarray:
        resized = cv2.resize(input_image, (self.size, self.size), interpolation=cv2.INTER_CUBIC)
        return np.clip(resized, 0.0, 1.0).astype(np.float32)

class ClaheResize(ImagePreprocessing):
    def __init__(self, 
                 size: int = CONFIG.target_size,
                 clip_limit: float = CONFIG.clahe.clip_limit,
                 tile_grid: int = CONFIG.clahe.tile_grid
                 ) -> None:
        super().__init__()
        self.size = size
        self.clip_limit = clip_limit
        self.tile_grid = tile_grid

    def resize(self, input_image: np.ndarray) -> np.ndarray:
        arr_uint8 = (input_image).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=(self.tile_grid, self.tile_grid))
        equalized = clahe.apply(arr_uint8)
        equalized_f = equalized.astype(np.float32) / 255.0
        resized = cv2.resize(equalized_f, (self.size, self.size), interpolation=cv2.INTER_CUBIC)
        return np.clip(resized, 0.0, 1.0).astype(np.float32)

class HaarResize(ImagePreprocessing):
    def __init__(self, 
                 size: int = CONFIG.target_size
                 ) -> None:
        super().__init__()
        self.size = size

    def resize(self, input_image: np.ndarray) -> np.ndarray:
        # Ensure input is float32 for initial resize
        input_image = input_image.astype(np.float32) 
        working = cv2.resize(input_image, (2 * self.size, 2 * self.size), interpolation=cv2.INTER_CUBIC)
        
        coeffs = pywt.dwt2(working, "haar")
        ll, (lh, hl, hh) = coeffs
        
        # Ensure coeffs are real (if input was complex, take real or abs)
        ll = np.real(ll).astype(np.float32)
        lh = np.real(lh).astype(np.float32)
        

        ll_resized = cv2.resize(ll,  (self.size, self.size), interpolation=cv2.INTER_CUBIC)
        lh_resized = cv2.resize(lh,  (self.size, self.size), interpolation=cv2.INTER_CUBIC)
        
        _norm = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
        stacked = np.stack([_norm(ll_resized), _norm(lh_resized)], axis=0)
        return stacked.astype(np.float32)


PREPROCESSING_REGISTRY: dict[str, Type[ImagePreprocessing]] = {
    "naive": NaiveResize,
    "clahe": ClaheResize,
    "dwt": HaarResize,
}

class ResizeComp:

    def __init__(self):
        self.registry = copy.copy(PREPROCESSING_REGISTRY)

    def set_clahe_settings(self, clahe: Type[ClaheResize]):
        self.registry["clahe"] = clahe

    def preprocess(self, input_image: np.ndarray, condition_type: str) -> np.ndarray:
        if condition_type not in self.registry:
            raise ValueError(f"Unknown condition '{condition_type}'")
        
        processor = self.registry[condition_type]
        
        # Check if it is a class (needs instantiation) or already an instance
        if isinstance(processor, type):
            processor = processor()
            
        return processor.resize(input_image)