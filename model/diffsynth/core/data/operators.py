import torch, torchvision, imageio, os, random
import imageio.v3 as iio
from PIL import Image


class DataProcessingPipeline:
    def __init__(self, operators=None):
        self.operators: list[DataProcessingOperator] = [] if operators is None else operators
        
    def __call__(self, data):
        for operator in self.operators:
            data = operator(data)
        return data
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline(self.operators + pipe.operators)


class DataProcessingOperator:
    def __call__(self, data):
        raise NotImplementedError("DataProcessingOperator cannot be called directly.")
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline([self]).__rshift__(pipe)


class DataProcessingOperatorRaw(DataProcessingOperator):
    def __call__(self, data):
        return data


class ToInt(DataProcessingOperator):
    def __call__(self, data):
        return int(data)


class ToFloat(DataProcessingOperator):
    def __call__(self, data):
        return float(data)


class ToStr(DataProcessingOperator):
    def __init__(self, none_value=""):
        self.none_value = none_value
    
    def __call__(self, data):
        if data is None: data = self.none_value
        return str(data)


class LoadImage(DataProcessingOperator):
    def __init__(self, convert_RGB=True, convert_RGBA=False):
        self.convert_RGB = convert_RGB
        self.convert_RGBA = convert_RGBA
    
    def __call__(self, data: str):
        image = Image.open(data)
        if self.convert_RGB: image = image.convert("RGB")
        if self.convert_RGBA: image = image.convert("RGBA")
        return image


class ImageCropAndResize(DataProcessingOperator):
    def __init__(self, height=None, width=None, max_pixels=None, height_division_factor=1, width_division_factor=1):
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor

    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    def get_height_width(self, image):
        if self.height is None or self.width is None:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    def __call__(self, data: Image.Image):
        image = self.crop_and_resize(data, *self.get_height_width(data))
        return image


class ToList(DataProcessingOperator):
    def __call__(self, data):
        return [data]
    

class LoadVideo(DataProcessingOperator):
    def __init__(
        self,
        num_frames=81,
        time_division_factor=4,
        time_division_remainder=1,
        frame_processor=lambda x: x,
        source_fps=None,
        target_fps=None,
        min_duration_sec=None,
        max_duration_sec=None,
        random_clip=True,
        return_frame_indices=False,
    ):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.frame_processor = frame_processor
        self.source_fps = source_fps
        self.target_fps = target_fps
        self.min_duration_sec = min_duration_sec
        self.max_duration_sec = max_duration_sec
        self.random_clip = random_clip and (min_duration_sec is not None and max_duration_sec is not None and source_fps is not None and target_fps is not None)
        self.return_frame_indices = return_frame_indices

    def get_num_frames(self, reader):
        num_frames = self.num_frames
        if int(reader.count_frames()) < num_frames:
            num_frames = int(reader.count_frames())
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames

    def get_frame_indices_random_clip(self, reader):
        """按 source_fps/target_fps 下采样，并在 [min_duration_sec, max_duration_sec] 内随机截取一段。"""
        total_frames = int(reader.count_frames())
        if total_frames <= 0:
            return []
        src_fps = float(self.source_fps)
        tgt_fps = float(self.target_fps)
        video_duration_sec = total_frames / src_fps
        duration_sec = random.uniform(self.min_duration_sec, self.max_duration_sec)
        duration_sec = min(duration_sec, video_duration_sec)
        if duration_sec <= 0:
            return []
        max_start_sec = video_duration_sec - duration_sec
        start_sec = random.uniform(0, max(0, max_start_sec))
        start_frame = int(start_sec * src_fps)
        num_frames_out = int(duration_sec * tgt_fps)
        while num_frames_out > 1 and num_frames_out % self.time_division_factor != self.time_division_remainder:
            num_frames_out -= 1
        if num_frames_out <= 0:
            num_frames_out = 1
        step = src_fps / tgt_fps
        frame_indices = []
        for i in range(num_frames_out):
            idx = min(total_frames - 1, int(round(start_frame + i * step)))
            frame_indices.append(idx)
        return frame_indices

    def __call__(self, data: str):
        reader = imageio.get_reader(data)
        if self.random_clip:
            frame_indices = self.get_frame_indices_random_clip(reader)
        else:
            num_frames = self.get_num_frames(reader)
            if self.source_fps is not None and self.target_fps is not None:
                step = self.source_fps / self.target_fps
                total_frames = int(reader.count_frames())
                frame_indices = [min(total_frames - 1, int(round(i * step))) for i in range(num_frames)]
            else:
                frame_indices = list(range(num_frames))
        frames = []
        for frame_id in frame_indices:
            frame = reader.get_data(frame_id)
            frame = Image.fromarray(frame)
            frame = self.frame_processor(frame)
            frames.append(frame)
        reader.close()
        if self.return_frame_indices:
            return {"frames": frames, "frame_indices": frame_indices}
        return frames


class SequencialProcess(DataProcessingOperator):
    def __init__(self, operator=lambda x: x):
        self.operator = operator
        
    def __call__(self, data):
        return [self.operator(i) for i in data]


class LoadGIF(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor
        
    def get_num_frames(self, path):
        num_frames = self.num_frames
        images = iio.imread(path, mode="RGB")
        if len(images) < num_frames:
            num_frames = len(images)
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
        
    def __call__(self, data: str):
        num_frames = self.get_num_frames(data)
        frames = []
        images = iio.imread(data, mode="RGB")
        for img in images:
            frame = Image.fromarray(img)
            frame = self.frame_processor(frame)
            frames.append(frame)
            if len(frames) >= num_frames:
                break
        return frames


class RouteByExtensionName(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data: str):
        file_ext_name = data.split(".")[-1].lower()
        for ext_names, operator in self.operator_map:
            if ext_names is None or file_ext_name in ext_names:
                return operator(data)
        raise ValueError(f"Unsupported file: {data}")


class RouteByType(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data):
        for dtype, operator in self.operator_map:
            if dtype is None or isinstance(data, dtype):
                return operator(data)
        raise ValueError(f"Unsupported data: {data}")


class LoadTorchPickle(DataProcessingOperator):
    def __init__(self, map_location="cpu"):
        self.map_location = map_location
        
    def __call__(self, data):
        return torch.load(data, map_location=self.map_location, weights_only=False)


class ToAbsolutePath(DataProcessingOperator):
    def __init__(self, base_path=""):
        self.base_path = base_path
        
    def __call__(self, data):
        return os.path.join(self.base_path, data)


class LoadAudio(DataProcessingOperator):
    def __init__(self, sr=16000):
        self.sr = sr
    def __call__(self, data: str):
        import librosa
        input_audio, sample_rate = librosa.load(data, sr=self.sr)
        return input_audio


class LoadCameraC2W(DataProcessingOperator):
    """Load camera-to-world matrices from NPZ file.

    Returns numpy array [N, 4, 4] float32 (raw c2w, normalization done in pipeline unit).
    """
    def __init__(self, key=None):
        self.key = key

    def __call__(self, data: str):
        import numpy as np
        npz = np.load(data)
        fallback_keys = ("data", "c2w", "cam_c2w", "poses", "cameras", "camera")
        if self.key is not None:
            arr = npz[self.key]
        else:
            arr = None
            for k in fallback_keys:
                if k in npz:
                    arr = npz[k]
                    break
            if arr is None:
                arr = npz[list(npz.keys())[0]]
        arr = arr.astype(np.float32)
        if arr.ndim == 2:
            arr = arr.reshape(-1, 4, 4)
        elif arr.ndim == 3 and arr.shape[-2:] == (3, 4):
            F = arr.shape[0]
            full = np.tile(np.eye(4, dtype=np.float32), (F, 1, 1))
            full[:, :3, :] = arr
            arr = full
        return arr
