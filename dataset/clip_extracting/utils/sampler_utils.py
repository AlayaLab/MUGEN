# https://superuser.com/questions/843292/ffmpeg-how-does-ffmpeg-decide-which-frames-to-pick-when-fps-value-is-specified
from typing import Union

import bitarray


class RecurrentBitQueue:
    def __init__(self, maxsize: int, buf_size: int = 65536):
        buf_size = max(buf_size, maxsize)

        self._buf = bitarray.bitarray(buf_size)
        self._buf_lptr = 0
        self._buf_rptr = 0
        self._buf_size = buf_size

        self.maxsize = maxsize

    def push(self, x: bool) -> None:
        if self._buf_rptr == self._buf_size:
            self._buf[0 : self._buf_rptr - self._buf_lptr] = self._buf[self._buf_lptr : self._buf_rptr]
            self._buf_lptr, self._buf_rptr = 0, self._buf_rptr - self._buf_lptr

        self._buf[self._buf_rptr] = x
        self._buf_rptr += 1

        if self._buf_rptr - self._buf_lptr > self.maxsize:
            self._buf_lptr += 1

    def size(self) -> int:
        return self._buf_rptr - self._buf_lptr

    def count(self) -> int:
        return self._buf.count(1, self._buf_lptr, self._buf_rptr)


class EMDownSampler:
    def __init__(self, s_fps: Union[int, float], t_fps: Union[int, float]):
        self._upsampling = s_fps < t_fps

        self.s_fps = s_fps
        self.t_fps = t_fps

        self.qf = [RecurrentBitQueue(round(s_fps * s) - 1) for s in [0.5, 1.0, 3.0]]
        self._frame_idx = 0

    def test_and_set(self) -> bool:
        if self._upsampling:
            return True
        # test
        _margins_a, _margins_b = [], []
        for q in self.qf:
            _tmp = self.s_fps * q.count() - self.t_fps * (q.size() + 1)
            _margins_a.append(abs(_tmp))
            _margins_b.append(abs(_tmp + self.s_fps))
        _res = sum(_margins_a) > sum(_margins_b)

        # set
        for q in self.qf:
            q.push(_res)

        return _res

    def __iter__(self):
        while True:
            yield self.test_and_set()
            self._frame_idx += 1
