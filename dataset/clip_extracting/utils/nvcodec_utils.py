import logging
from fractions import Fraction

import PyNvVideoCodec as nvvc


class NVVCVideoDecoder:
    def __init__(self, enc_file, device_id, cuda_ctx, cuda_stream):
        """
        Create instance of HW-accelerated video decoder.
        :param enc_file: Full path to the MP4 file that needs to be decoded.
        :param device_id: id of video card which will be used for decoding & processing.
        :param cuda_ctx: A cuda context object.
        """
        self.device_id = device_id
        self.cuda_ctx = cuda_ctx
        self.enc_file = enc_file
        self.cuda_stream = cuda_stream
        self.nvDemux = nvvc.PyNvDemuxer(self.enc_file)
        self.nvDec = nvvc.CreateDecoder(
            gpuid=device_id,
            codec=self.nvDemux.GetNvCodecId(),
            cudacontext=self.cuda_ctx.handle,
            cudastream=self.cuda_stream.handle,
            usedevicememory=1,
        )

        self.width = self.nvDemux.Width()
        self.height = self.nvDemux.Height()
        self.fps = self.nvDemux.FrameRate()
        self.pixelFormat = self.nvDec.GetPixelFormat()

        self.frame_idx = 0

        logging.info(f"Width={self.width}, Height={self.height}, FrameRate={self.fps}, PixelFormat={self.pixelFormat}.")

    def __iter__(self):
        for packet in self.nvDemux:
            for frame in self.nvDec.Decode(packet):
                self.frame_idx += 1
                yield frame

    def finish(self):
        logging.info(f"Finish decode {self.frame_idx} frames.")


class NVVCVideoEncoder:
    def __init__(
        self,
        enc_file,
        device_id,
        width,
        height,
        fps,
        cuda_ctx,
        cuda_stream,
    ):
        self.device_id = device_id
        self.fps = round(Fraction(fps), 6)
        self.enc_file = enc_file
        self.cuda_ctx = cuda_ctx
        self.cuda_stream = cuda_stream

        self.nvEnc = nvvc.CreateEncoder(
            width,
            height,
            fmt="NV12",
            usecpuinputbuffer=0,
            codec="hevc",
            fps=fps,
            initqp="0,0,0",
            gop=int(fps // 2),
            tuning_info="high_quality",
            preset="P7",
            maxbitrate="60M",
            vbvinit="120M",
            vbvbufsize="120M",
            rc="vbr",
            temporalaq=1,
            aq=1,
            colorspace="bt709",
            cudastream=cuda_stream.handle,
        )

        self.frame_idx = 0

    def __call__(self, frame):
        bitstream = self.nvEnc.Encode(frame)
        self.frame_idx += 1

        bitstream = bytearray(bitstream)

        self.enc_file.write(bitstream)

    def finish(self):
        bitstream = self.nvEnc.EndEncode()

        if bitstream:
            bitstream = bytearray(bitstream)

            self.enc_file.write(bitstream)

        del self.nvEnc
