from .operators import *
import torch, json, pandas, random


class UnifiedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        repeat=1,
        data_file_keys=tuple(),
        main_data_operator=lambda x: x,
        special_operator_map=None,
        ti2v_prob=None,
    ):
        self.base_path = base_path
        self.metadata_path = metadata_path
        self.repeat = repeat
        self.data_file_keys = data_file_keys
        self.main_data_operator = main_data_operator
        self.cached_data_operator = LoadTorchPickle()
        self.special_operator_map = {} if special_operator_map is None else special_operator_map
        self.ti2v_prob = ti2v_prob
        self.data = []
        self.cached_data = []
        self.load_from_cache = metadata_path is None
        self.load_metadata(metadata_path)

    @staticmethod
    def default_image_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor)),
            (list, SequencialProcess(ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor))),
        ])
    
    @staticmethod
    def default_video_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        num_frames=81, time_division_factor=4, time_division_remainder=1,
        source_fps=None, target_fps=None, min_duration_sec=None, max_duration_sec=None, random_clip=True,
        return_frame_indices=False,
    ):
        load_video_kw = dict(
            num_frames=num_frames,
            time_division_factor=time_division_factor,
            time_division_remainder=time_division_remainder,
            frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
        )
        if source_fps is not None:
            load_video_kw["source_fps"] = source_fps
        if target_fps is not None:
            load_video_kw["target_fps"] = target_fps
        if min_duration_sec is not None:
            load_video_kw["min_duration_sec"] = min_duration_sec
        if max_duration_sec is not None:
            load_video_kw["max_duration_sec"] = max_duration_sec
        load_video_kw["random_clip"] = random_clip
        load_video_kw["return_frame_indices"] = return_frame_indices
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> RouteByExtensionName(operator_map=[
                (("jpg", "jpeg", "png", "webp"), LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor) >> ToList()),
                (("gif",), LoadGIF(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                )),
                (("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"), LoadVideo(**load_video_kw)),
            ])),
        ])
        
    def search_for_cached_data_files(self, path):
        for file_name in os.listdir(path):
            subpath = os.path.join(path, file_name)
            if os.path.isdir(subpath):
                self.search_for_cached_data_files(subpath)
            elif subpath.endswith(".pth"):
                self.cached_data.append(subpath)
    
    def load_metadata(self, metadata_path):
        if metadata_path is None:
            print("No metadata_path. Searching for cached data files.")
            self.search_for_cached_data_files(self.base_path)
            print(f"{len(self.cached_data)} cached data files found.")
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        elif metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, 'r') as f:
                for line in f:
                    metadata.append(json.loads(line.strip()))
            self.data = metadata
        else:
            metadata = pandas.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]

    def __getitem__(self, data_id):
        max_retries = 5
        for attempt in range(max_retries):
            try:
                return self._load_item(data_id)
            except Exception as e:
                print(f"[UnifiedDataset] Error loading sample {data_id}: {e}. "
                      f"Retry {attempt + 1}/{max_retries} with a random sample.")
                data_id = random.randint(0, len(self.data) - 1)
        # last attempt, let it raise if it fails
        return self._load_item(data_id)

    def _load_item(self, data_id):
        if self.load_from_cache:
            data = self.cached_data[data_id % len(self.cached_data)]
            data = self.cached_data_operator(data)
        else:
            data = self.data[data_id % len(self.data)].copy()
            for key in self.data_file_keys:
                if key in data:
                    if key in self.special_operator_map:
                        data[key] = self.special_operator_map[key](data[key])
                    elif key in self.data_file_keys:
                        result = self.main_data_operator(data[key])
                        # LoadVideo with return_frame_indices=True returns a dict
                        if isinstance(result, dict) and "frames" in result:
                            data["frame_indices"] = result["frame_indices"]
                            data[key] = result["frames"]
                        else:
                            data[key] = result
        if self.ti2v_prob is not None and "prompt" in data:
            data["task_type"] = "ti2v" if random.random() < self.ti2v_prob else "t2v"
        return data

    def __len__(self):
        if self.load_from_cache:
            return len(self.cached_data) * self.repeat
        else:
            return len(self.data) * self.repeat
        
    def check_data_equal(self, data1, data2):
        # Debug only
        if len(data1) != len(data2):
            return False
        for k in data1:
            if data1[k] != data2[k]:
                return False
        return True
