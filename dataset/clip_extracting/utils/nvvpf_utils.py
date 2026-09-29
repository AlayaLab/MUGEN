import io
from typing import List, Union

import cvcuda
import nvcv
import PyNvVideoCodec as nvvc
import torch

from utils.nvcodec_utils import (
    NVVCVideoDecoder,
    NVVCVideoEncoder,
)
from utils.sampler_utils import EMDownSampler

pixel_format_to_cvcuda_code = {
    nvvc.Pixel_Format.YUV444: cvcuda.ColorConversion.YUV2RGB,
    nvvc.Pixel_Format.NV12: cvcuda.ColorConversion.YUV2RGB_NV12,
}


class VideoBatchDecoder:
    def __init__(
        self,
        width: int,
        height: int,
        fps: Union[int, float],
        batch_size: int,
        device_id: int,
        cuda_ctx,
        cuda_stream,
    ):
        self.device_id = device_id
        self.cuda_ctx = cuda_ctx
        self.cuda_stream = cuda_stream

        self.width = width
        self.height = height
        self.fps = fps
        self.batch_size = batch_size

        self.decoder = None
        self.sampler = None
        self.cvcuda_colorconversion = None

        self.batch_idx = 0

    def initialize(self, filename: str) -> None:
        self.decoder = NVVCVideoDecoder(filename, self.device_id, self.cuda_ctx, self.cuda_stream)

        self.sampler = EMDownSampler(self.decoder.fps, self.fps)

        self.cvcuda_colorconversion = pixel_format_to_cvcuda_code.get(self.decoder.pixelFormat)
        if self.cvcuda_colorconversion is None:
            raise ValueError(f"Unsupported pixel format: {self.decoder.pixelFormat}")

    def process(self, cvcuda_YUVtensor: List[nvcv.Tensor]) -> nvcv.Tensor:
        cvcuda_YUVtensor = cvcuda.stack(cvcuda_YUVtensor)

        if cvcuda_YUVtensor.layout != "NHWC":
            raise ValueError("Unexpected tensor layout, NHWC expected.")

        cvcuda_RGBtensor = cvcuda.cvtcolor(cvcuda_YUVtensor, self.cvcuda_colorconversion)

        cvcuda_RGBtensor = cvcuda.hq_resize(
            cvcuda_RGBtensor,
            (
                self.height,
                self.width,
            ),
            interpolation=cvcuda.Interp.LANCZOS,
        )
        return cvcuda_RGBtensor

    def __iter__(self):
        cvcuda_YUVtensor = []
        for flag, frame in zip(self.sampler, self.decoder):
            if flag:
                _cvcuda_YUVtensor = nvcv.as_tensor(nvcv.as_image(frame.nvcv_image(), nvcv.Format.U8))
                if _cvcuda_YUVtensor.layout == "NCHW":
                    cvcuda_YUVtensor.append(cvcuda.reformat(_cvcuda_YUVtensor, "NHWC"))
                else:
                    raise ValueError("Unexpected tensor layout, NCHW expected.")
            if len(cvcuda_YUVtensor) == self.batch_size:
                yield self.process(cvcuda_YUVtensor)
                cvcuda_YUVtensor.clear()
        if len(cvcuda_YUVtensor) > 0:
            yield self.process(cvcuda_YUVtensor)
            cvcuda_YUVtensor.clear()

    def finish(self):
        self.decoder.finish()

        del self.decoder


class VideoMemoryEncoder:
    def __init__(
        self,
        width: int,
        height: int,
        fps: Union[int, float],
        batch_size: int,
        device_id: int,
        cuda_ctx,
        cuda_stream,
    ):
        self.device_id = device_id
        self.cuda_ctx = cuda_ctx
        self.cuda_stream = cuda_stream

        self.width = width
        self.height = height
        self.fps = fps
        self.batch_size = batch_size
        assert batch_size == 1

        self.file = None

        self.encoder = None

    def initialize(self):
        self.file = io.BytesIO()

        self.encoder = NVVCVideoEncoder(
            self.file, self.device_id, self.width, self.height, self.fps, self.cuda_ctx, self.cuda_stream
        )

    def __call__(self, cvcuda_RGBtensor):
        cvcuda_YUVtensor = cvcuda.cvtcolor(
            cvcuda_RGBtensor,
            cvcuda.ColorConversion.RGB2YUV_NV12,
        )
        cvcuda_YUVtensor = cvcuda.reformat(cvcuda_YUVtensor, "NCHW")

        self.encoder(torch.as_tensor(cvcuda_YUVtensor.cuda(), device=f"cuda:{self.device_id}").squeeze(0, 1))

    def finish(self):
        self.encoder.finish()

        file = self.file.getvalue()
        self.file.close()

        del self.file
        del self.encoder
        return file
