from datetime import time
import torch
import torch.nn.functional as F
import numpy as np
import os
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from torch_dbscan import dbscan

def compute_foreground_mask(x0_pred, channel=0, tau=0.1):
    x0_pred = x0_pred[channel] # channel 0
    f, h, w = x0_pred.shape[0], x0_pred.shape[1], x0_pred.shape[2]
    x0_pred = x0_pred.view(f, h*w)

    # step 1: normalize tensor to [0, 1] 
    min_val = x0_pred.min(axis=1).values.unsqueeze(-1)
    max_val = x0_pred.max(axis=1).values.unsqueeze(-1)
    x0_pred = (x0_pred - min_val) / (max_val - min_val)
    
    # step 2: take foreground as relu(abs(x0_pred - median) - tau)
    med_val = x0_pred.median(axis=1).values.unsqueeze(-1)
    x0_pred = F.relu((x0_pred - med_val).abs() - tau)

    return x0_pred.view(f, h, w)

def compute_objects_masks(fg_mask, top_p=0.08, min_samples=10, eps_scale=1.5):
    f, h, w = fg_mask.shape[0], fg_mask.shape[1], fg_mask.shape[2]
    device = fg_mask.device
    dtype = fg_mask.dtype

    # step 1: select top-p foreground pixels
    fg_mask_flat = fg_mask.view(-1)
    thres = torch.quantile(fg_mask_flat, 1 - top_p)
    sel_mask = (fg_mask_flat >= thres).to(torch.int32)
    idx = torch.nonzero(sel_mask, as_tuple=False).squeeze(1)

    # unravel indices
    tt = idx // (h * w)
    yy = (idx % (h * w)) // w
    xx = (idx % (h * w)) % w

    # points in 3D space
    pts = torch.stack([xx, yy, tt], dim=1).to(device=device, dtype=dtype)
    wts = fg_mask_flat[sel_mask].to(device=device, dtype=dtype)

    print(pts.shape)

    # step 2: auto-tune eps from kNN distances
    k = min(min_samples, max(pts.shape[0]-1, 1)) if len(pts) > min_samples else 1
    D = torch.cdist(pts, pts)
    
    if k >= 1:
        knn_d, _ = torch.topk(D, k=k+1, largest=False)
        base = torch.median(knn_d[:, -1])
        eps = torch.clamp(eps_scale * base, min=torch.tensor(1.0, device=device, dtype=dtype)).item()
    else:
        eps = 3.0

    # step 3: cluster using DBSCAN
    neighborhoods = D < eps
    neighbor_counts = torch.sum(neighborhoods, dim=1)
    core_points = neighbor_counts >= min_samples

    labels = torch.full((pts.shape[0],), -1, dtype=torch.int32, device=device)
    cluster_id = 0
    for i in range(pts.shape[0]):
        if core_points[i] and labels[i] == -1:
            # Start a new cluster
            labels[i] = cluster_id
            cluster_members = [i]
            while cluster_members:
                new_members = []
                for member in cluster_members:
                    neighbors = torch.where(neighborhoods[member] & (labels == -1))[0]
                    labels[neighbors] = cluster_id
                    new_members.extend(neighbors.tolist())
                cluster_members = new_members
            cluster_id += 1

    # step 4: create an f**w image
    mask = torch.full((f, h, w), -1, dtype=torch.int32, device=device)
        
    valid = labels >= 0
    if valid.any():
        mask[tt[valid], yy[valid], xx[valid]] = labels[valid]

    return mask

def compute_physical_properties(fg_mask):
    masses = compute_object_masses(fg_mask) # [f, num_objects]
    edge_mass = count_pixels_that_touch_edge(fg_mask, masses) # [f, num_objects]
    com_positions = compute_center_of_mass_positions(fg_mask, masses) # [f, num_objects, 2]
    com_velocities = compute_center_of_mass_velocities(com_positions) # [f, num_objects, 2]

    return masses, edge_mass, com_positions, com_velocities

def compute_object_masses(fg_mask):
    f = fg_mask.shape[0]
    num_objects = fg_mask.max().item() + 1
    masses = torch.zeros((f, num_objects), device=fg_mask.device)
    for i in range(f):
        for j in range(num_objects):
            frame_mask = fg_mask[i]
            masses[i, j] = (frame_mask == j).sum()
    return masses

def count_pixels_that_touch_edge(fg_mask, masses):
    f = fg_mask.shape[0]
    num_objects = fg_mask.max().item() + 1
    edge_pixels = torch.zeros((f, num_objects), dtype=torch.float32, device=fg_mask.device)
    for i in range(f):
        for j in range(num_objects):
            frame_mask = fg_mask[i]
            if masses[i, j] > 0:
                pixels = (frame_mask == j).nonzero(as_tuple=True)
                pixels_0 = (pixels[0] == 0).float() + (pixels[0] == frame_mask.shape[0] - 1).float()
                pixels_1 = (pixels[1] == 0).float() + (pixels[1] == frame_mask.shape[1] - 1).float()
                edge_pixels[i, j] = (pixels_0 + pixels_1).sum()
    return edge_pixels

def compute_center_of_mass_positions(fg_mask, masses):
    f = fg_mask.shape[0]
    num_objects = fg_mask.max().item() + 1
    com_positions = torch.zeros((f, num_objects, 2), device=fg_mask.device)
    for i in range(f):
        for j in range(num_objects):
            frame_mask = fg_mask[i]
            if masses[i, j] > 0:
                com_positions[i, j, 0] = (frame_mask == j).nonzero(as_tuple=True)[1].float().mean()
                com_positions[i, j, 1] = (frame_mask == j).nonzero(as_tuple=True)[0].float().mean()
    return com_positions

def compute_center_of_mass_velocities(com_positions):
    # Compute velocities as the difference between consecutive frames
    return torch.diff(com_positions, dim=0, append=com_positions[-1:])

def compute_losses_aux(masses, edge_mass, com_velocities):
    losses = {}
    mass_diff = torch.abs(torch.diff(masses, dim=0)) # [f-1, num_objects]
    edge_mass = edge_mass[:-1]  # [f-1, num_objects]
    mass_diff = F.relu(mass_diff - edge_mass) # if mass_diff < edge_mass, treat mass_diff as 0. Otherwise, treat as is.
    losses['mass_1'] = mass_diff.sum() # [1], should be zero
    losses['mass_2'] = torch.var(mass_diff, dim=0).sum() # [1], should be small
    losses['mass_3'] = torch.var(masses, dim=0).sum() # [1], should be small for each object
    losses['mass_4'] = torch.abs(mass_diff).sum() # [1], should be small for each object

    momentum = masses.unsqueeze(-1) * com_velocities # [f, num_objects, 2]
    momentum = momentum[:-1]  # remove last frame which is zero velocity
    momentum = momentum.permute(0, 2, 1) # [f-1, 2, num_objects]
    momentum = momentum.sum(dim=-1) # [f-1, 2]
    losses['momentum_1'] = torch.var(momentum, dim=0).sum() # [1], should be small on every axis
    losses['momentum_2'] = torch.abs(torch.diff(momentum, dim=0)).sum() # [1], should be small on every axis

    kinetic_energy = 0.5 * masses * torch.norm(com_velocities, dim=-1)**2 # [f, num_objects]
    kinetic_energy = kinetic_energy[:-1]  # remove last frame which is zero velocity
    kinetic_energy = kinetic_energy.sum(dim=-1) # [f-1]
    losses['kinetic_energy_1'] = torch.var(kinetic_energy, dim=0) # [1], should be small
    losses['kinetic_energy_2'] = torch.abs(torch.diff(kinetic_energy, dim=0)).sum() # [1], should be small

    return losses

def compute_losses(x0_pred):
    start_time = time.time()
    output = compute_foreground_mask(x0_pred, tau=0.1)
    output = compute_objects_masks(output)
    masses, edge_mass, com_positions, com_velocities = compute_physical_properties(output)

    losses = compute_losses_aux(masses, edge_mass, com_velocities)
    end_time = time.time()

    print(f"compute_losses took {end_time - start_time:.1f} seconds")
    return losses

__all__ = ['compute_losses']