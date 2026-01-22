"""Utility module for loading PLY files in Gaussian Splatting format."""

import struct
from typing import Dict

import numpy as np
import torch


def load_ply(ply_path: str) -> Dict[str, torch.Tensor]:
    """Load Gaussian splatting parameters from a binary PLY file.
    
    Args:
        ply_path: Path to the PLY file
        
    Returns:
        Dictionary containing:
            - means: torch.Tensor (N, 3) - Gaussian positions
            - scales: torch.Tensor (N, 3) - Gaussian scales (in log space)
            - quats: torch.Tensor (N, 4) - Gaussian quaternions
            - opacities: torch.Tensor (N,) - Gaussian opacities (in sigmoid space)
            - sh0: torch.Tensor (N, 1, 3) - SH DC coefficients
            - shN: torch.Tensor (N, K, 3) - SH rest coefficients
    """
    with open(ply_path, "rb") as f:
        # Read header
        header_lines = []
        while True:
            line = f.readline()
            header_lines.append(line)
            if b"end_header" in line:
                break
        
        header = b"".join(header_lines).decode("ascii", errors="ignore")
        
        # Parse header to find properties and vertex count
        num_vertices = 0
        properties = []
        for line in header.split("\n"):
            line = line.strip()
            if line.startswith("element vertex"):
                num_vertices = int(line.split()[-1])
            elif line.startswith("property float"):
                prop_name = line.split()[-1]
                properties.append(prop_name)
        
        # Determine SH degree from number of f_rest properties
        f_rest_props = [p for p in properties if p.startswith("f_rest_")]
        num_f_rest = len(f_rest_props)
        # f_rest has shape (K, 3) where K = (sh_degree + 1)^2 - 1
        # So num_f_rest = K * 3 = ((sh_degree + 1)^2 - 1) * 3
        # Solving: (sh_degree + 1)^2 - 1 = num_f_rest / 3
        # (sh_degree + 1)^2 = num_f_rest / 3 + 1
        # sh_degree + 1 = sqrt(num_f_rest / 3 + 1)
        # sh_degree = sqrt(num_f_rest / 3 + 1) - 1
        if num_f_rest > 0:
            sh_degree = int(np.sqrt(num_f_rest / 3 + 1) - 1)
            num_sh_coeffs = (sh_degree + 1) ** 2
            num_sh_rest = num_sh_coeffs - 1  # Excluding DC
        else:
            sh_degree = 0
            num_sh_rest = 0
        
        # Read binary data
        # Each vertex has: x, y, z, f_dc_0, f_dc_1, f_dc_2, f_rest_*, opacity, scale_0, scale_1, scale_2, rot_0, rot_1, rot_2, rot_3
        num_floats_per_vertex = len(properties)
        data = np.fromfile(f, dtype=np.float32, count=num_vertices * num_floats_per_vertex)
        data = data.reshape(num_vertices, num_floats_per_vertex)
        
        # Extract data based on property order from header
        # Find indices for each property type
        prop_indices = {}
        idx = 0
        for prop in properties:
            if prop in ["x", "y", "z"]:
                if "xyz" not in prop_indices:
                    prop_indices["xyz"] = []
                prop_indices["xyz"].append((prop, idx))
            elif prop.startswith("f_dc_"):
                if "f_dc" not in prop_indices:
                    prop_indices["f_dc"] = []
                prop_indices["f_dc"].append((prop, idx))
            elif prop.startswith("f_rest_"):
                if "f_rest" not in prop_indices:
                    prop_indices["f_rest"] = []
                prop_indices["f_rest"].append((prop, idx))
            elif prop == "opacity":
                prop_indices["opacity"] = idx
            elif prop.startswith("scale_"):
                if "scale" not in prop_indices:
                    prop_indices["scale"] = []
                prop_indices["scale"].append((prop, idx))
            elif prop.startswith("rot_"):
                if "rot" not in prop_indices:
                    prop_indices["rot"] = []
                prop_indices["rot"].append((prop, idx))
            idx += 1
        
        # Extract means (x, y, z) - ensure correct order
        xyz_props = sorted(prop_indices["xyz"], key=lambda x: x[0])  # Sort by property name
        xyz_indices = [idx for _, idx in xyz_props]
        means = data[:, xyz_indices]
        
        # Extract f_dc - ensure correct order (f_dc_0, f_dc_1, f_dc_2)
        f_dc_props = sorted(prop_indices["f_dc"], key=lambda x: int(x[0].split("_")[-1]))
        f_dc_indices = [idx for _, idx in f_dc_props]
        f_dc = data[:, f_dc_indices]
        
        # Extract f_rest - ensure correct order
        if "f_rest" in prop_indices:
            f_rest_props = sorted(prop_indices["f_rest"], key=lambda x: int(x[0].split("_")[-1]))
            f_rest_indices = [idx for _, idx in f_rest_props]
            f_rest = data[:, f_rest_indices]
        else:
            f_rest = np.zeros((num_vertices, 0), dtype=np.float32)
        
        # Extract opacity
        opacities = data[:, prop_indices["opacity"]]
        
        # Extract scales - ensure correct order (scale_0, scale_1, scale_2)
        scale_props = sorted(prop_indices["scale"], key=lambda x: int(x[0].split("_")[-1]))
        scale_indices = [idx for _, idx in scale_props]
        scales = data[:, scale_indices]
        
        # Extract quats/rotations - ensure correct order (rot_0, rot_1, rot_2, rot_3)
        rot_props = sorted(prop_indices["rot"], key=lambda x: int(x[0].split("_")[-1]))
        rot_indices = [idx for _, idx in rot_props]
        quats = data[:, rot_indices]
        
        # Convert to torch tensors
        means = torch.from_numpy(means).float()
        scales = torch.from_numpy(scales).float()
        quats = torch.from_numpy(quats).float()
        opacities = torch.from_numpy(opacities).float()
        
        # Reshape SH coefficients
        # f_dc: (N, 3) -> (N, 1, 3)
        sh0 = torch.from_numpy(f_dc).float().unsqueeze(1)  # (N, 1, 3)
        
        # f_rest: (N, K*3) -> (N, K, 3) where K = num_sh_rest
        if num_sh_rest > 0:
            shN = torch.from_numpy(f_rest).float().reshape(num_vertices, num_sh_rest, 3)  # (N, K, 3)
        else:
            shN = torch.zeros((num_vertices, 0, 3), dtype=torch.float32)
        
        return {
            "means": means,
            "scales": scales,
            "quats": quats,
            "opacities": opacities,
            "sh0": sh0,
            "shN": shN,
        }
