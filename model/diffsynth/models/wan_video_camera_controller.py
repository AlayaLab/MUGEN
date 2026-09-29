import torch
import torch.nn as nn
import numpy as np
import math
from einops import rearrange
import os
from typing import Optional, Union
from typing_extensions import Literal

class SimpleAdapter(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, num_residual_blocks=1, downscale_factor=8):
        super(SimpleAdapter, self).__init__()

        # Pixel Unshuffle: reduce spatial dimensions by `downscale_factor`
        # (8 for Wan2.1-VAE pipelines, 16 for Wan2.2-VAE / TI2V-5B pipelines)
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor=downscale_factor)

        # Convolution: reduce spatial dimensions by a factor
        #  of 2 (without overlap)
        self.conv = nn.Conv2d(in_dim * downscale_factor * downscale_factor, out_dim, kernel_size=kernel_size, stride=stride, padding=0)

        # Residual blocks for feature extraction
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(out_dim) for _ in range(num_residual_blocks)]
        )

    def forward(self, x):
        # Reshape to merge the frame dimension into batch
        bs, c, f, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(bs * f, c, h, w)

        # Pixel Unshuffle operation
        x_unshuffled = self.pixel_unshuffle(x)

        # Convolution operation
        x_conv = self.conv(x_unshuffled)

        # Feature extraction with residual blocks
        out = self.residual_blocks(x_conv)

        # Reshape to restore original bf dimension
        out = out.view(bs, f, out.size(1), out.size(2), out.size(3))

        # Permute dimensions to reorder (if needed), e.g., swap channels and feature frames
        out = out.permute(0, 2, 1, 3, 4)

        return out
    
    def process_camera_coordinates(
        self,
        direction_or_cam_c2w,
        length_or_num_frames=None,
        height: int = None,
        width: int = None,
        speed_or_speed_scale=None,
        origin=None,
    ):
        """Dispatch by first arg type:
          str        → pinhole + enum direction (legacy 5B-Fun-CC API) → ray_condition
          ndarray/Tensor [F,4,4] → panoramic                            → pano_ray_condition
        Returns Plücker 6D embedding [F, H, W, 6].
        """
        if isinstance(direction_or_cam_c2w, (np.ndarray, torch.Tensor)):
            cam_c2w = direction_or_cam_c2w
            num_frames = length_or_num_frames if length_or_num_frames is not None else len(cam_c2w)
            speed_scale = speed_or_speed_scale if speed_or_speed_scale is not None else 1.0
            cams = [Pano_Camera(cam_c2w[i], speed_scale) for i in range(num_frames)]
            return process_pano_pose_file(cams, width, height)

        direction = direction_or_cam_c2w
        length = length_or_num_frames
        speed = speed_or_speed_scale if speed_or_speed_scale is not None else (1 / 54)
        if origin is None:
            origin = (0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0)
        coordinates = generate_camera_coordinates(direction, length, speed, origin)
        return process_pose_file(coordinates, width, height)



class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return out
    
class Camera(object):
    """Copied from https://github.com/hehao13/CameraCtrl/blob/main/inference.py
    """
    def __init__(self, entry):
        fx, fy, cx, cy = entry[1:5]
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        w2c_mat = np.array(entry[7:]).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4)
        w2c_mat_4x4[:3, :] = w2c_mat
        self.w2c_mat = w2c_mat_4x4
        self.c2w_mat = np.linalg.inv(w2c_mat_4x4)


class Pano_Camera(object):
    """Panoramic (ERP) camera. No intrinsics K — uses only c2w.

    Per-pixel direction:
        lon  = 2π·(u/(W-1) − 0.5)     ∈ [-π, π]
        lat  = π·(0.5 − v/(H-1))      ∈ [π/2, -π/2]
        d_cam   = (cos lat·sin lon, sin lat, cos lat·cos lon)
        d_world = R · d_cam,   m = o × d_world,   where R, o are from c2w.
    Plücker 6D = (d_world, m).
    """
    def __init__(self, cam_c2w, speed_scale: float = 1.0):
        c2w = np.asarray(cam_c2w, dtype=np.float32).copy()
        if abs(speed_scale - 1.0) > 1e-6:
            c2w[:3, 3] *= speed_scale
        self.c2w_mat = c2w
        self.w2c_mat = np.linalg.inv(c2w)


def get_relative_pose(cam_params):
    """Copied from https://github.com/hehao13/CameraCtrl/blob/main/inference.py
    """
    abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
    abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
    cam_to_origin = 0
    target_cam_c2w = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, -cam_to_origin],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    abs2rel = target_cam_c2w @ abs_w2cs[0]
    ret_poses = [target_cam_c2w, ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
    ret_poses = np.array(ret_poses, dtype=np.float32)
    return ret_poses

def custom_meshgrid(*args):
    # torch>=2.0.0 only
    return torch.meshgrid(*args, indexing='ij')


def ray_condition(K, c2w, H, W, device):
    """Copied from https://github.com/hehao13/CameraCtrl/blob/main/inference.py
    """
    # c2w: B, V, 4, 4
    # K: B, V, 4

    B = K.shape[0]

    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    i = i.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5  # [B, HxW]
    j = j.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5  # [B, HxW]

    fx, fy, cx, cy = K.chunk(4, dim=-1)  # B,V, 1

    zs = torch.ones_like(i)  # [B, HxW]
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)  # B, V, HW, 3
    directions = directions / directions.norm(dim=-1, keepdim=True)  # B, V, HW, 3

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)  # B, V, 3, HW
    rays_o = c2w[..., :3, 3]  # B, V, 3
    rays_o = rays_o[:, :, None].expand_as(rays_d)  # B, V, 3, HW
    # c2w @ dirctions
    rays_dxo = torch.linalg.cross(rays_o, rays_d)
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)  # B, V, H, W, 6
    # plucker = plucker.permute(0, 1, 4, 2, 3)
    return plucker


def pano_ray_condition(c2w: torch.Tensor, H: int, W: int, device=None) -> torch.Tensor:
    """Panoramic Plücker 6D ray condition. Each ERP pixel maps directly to a
    spherical direction (no K). Returns [B, V, H, W, 6] = (d_world, o × d_world).
    """
    if not isinstance(c2w, torch.Tensor):
        c2w = torch.as_tensor(c2w, dtype=torch.float32)
    if device is None:
        device = c2w.device
    c2w = c2w.to(device=device, dtype=torch.float32)
    if c2w.dim() == 3:
        c2w = c2w.unsqueeze(0)
    B, V = c2w.shape[:2]

    grid_v, grid_u = custom_meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
    )
    lon = (grid_u / max(W - 1, 1) - 0.5) * 2 * math.pi
    lat = (0.5 - grid_v / max(H - 1, 1)) * math.pi
    d_cam = torch.stack(
        [torch.cos(lat) * torch.sin(lon),
         torch.sin(lat),
         torch.cos(lat) * torch.cos(lon)],
        dim=-1,
    )

    R = c2w[..., :3, :3]
    d_cam_b = d_cam.unsqueeze(0).unsqueeze(0).expand(B, V, H, W, 3)
    d_world = torch.einsum("bvij,bvhwj->bvhwi", R, d_cam_b)

    o = c2w[..., :3, 3].view(B, V, 1, 1, 3).expand(B, V, H, W, 3)
    m = torch.cross(o, d_world, dim=-1)

    return torch.cat([d_world, m], dim=-1)


def process_pano_pose_file(cam_params, width: int, height: int, device='cpu', return_poses: bool = False):
    """cam_params: list of Pano_Camera. Returns [F, H, W, 6]."""
    if return_poses:
        return cam_params
    c2ws = get_relative_pose(cam_params)
    c2ws = torch.as_tensor(c2ws)[None]
    plucker = pano_ray_condition(c2ws, H=height, W=width, device=torch.device(device))
    return plucker[0]


def process_pose_file(cam_params, width=672, height=384, original_pose_width=1280, original_pose_height=720, device='cpu', return_poses=False):
    if return_poses:
        return cam_params
    else:
        cam_params = [Camera(cam_param) for cam_param in cam_params]

        sample_wh_ratio = width / height
        pose_wh_ratio = original_pose_width / original_pose_height  # Assuming placeholder ratios, change as needed

        if pose_wh_ratio > sample_wh_ratio:
            resized_ori_w = height * pose_wh_ratio
            for cam_param in cam_params:
                cam_param.fx = resized_ori_w * cam_param.fx / width
        else:
            resized_ori_h = width / pose_wh_ratio
            for cam_param in cam_params:
                cam_param.fy = resized_ori_h * cam_param.fy / height

        intrinsic = np.asarray([[cam_param.fx * width,
                                cam_param.fy * height,
                                cam_param.cx * width,
                                cam_param.cy * height]
                                for cam_param in cam_params], dtype=np.float32)

        K = torch.as_tensor(intrinsic)[None]  # [1, 1, 4]
        c2ws = get_relative_pose(cam_params)  # Assuming this function is defined elsewhere
        c2ws = torch.as_tensor(c2ws)[None]  # [1, n_frame, 4, 4]
        plucker_embedding = ray_condition(K, c2ws, height, width, device=device)[0].permute(0, 3, 1, 2).contiguous()  # V, 6, H, W
        plucker_embedding = plucker_embedding[None]
        plucker_embedding = rearrange(plucker_embedding, "b f c h w -> b f h w c")[0]
        return plucker_embedding



def generate_camera_coordinates(
    direction: Literal["Left", "Right", "Up", "Down", "LeftUp", "LeftDown", "RightUp", "RightDown", "In", "Out"],
    length: int,
    speed: float = 1/54,
    origin=(0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0)
):
    coordinates = [list(origin)]
    while len(coordinates) < length:
        coor = coordinates[-1].copy()
        if "Left" in direction:
            coor[9] += speed
        if "Right" in direction:
            coor[9] -= speed
        if "Up" in direction:
            coor[13] += speed
        if "Down" in direction:
            coor[13] -= speed
        if "In" in direction:
            coor[18] -= speed
        if "Out" in direction:
            coor[18] += speed
        coordinates.append(coor)
    return coordinates
