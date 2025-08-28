import torch
import torch.nn.functional as F
import numpy as np
import os
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

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

def compute_objects_masks(fg_mask, top_p=0.2, kappa=16.0, bg_logit=-2.0, min_samples=5, eps_scale=1.5):
    f, h, w = fg_mask.shape[0], fg_mask.shape[1], fg_mask.shape[2]

    fg_mask = fg_mask.numpy()
    
    tt, yy, xx = np.meshgrid(np.arange(f), np.arange(h), np.arange(w), indexing='ij')

    # step 1: select top-p foreground pixels
    fg_mask_flat = fg_mask.reshape(-1)
    thres = np.quantile(fg_mask_flat, 1 - top_p)
    mask_bools = (fg_mask_flat >= thres)
    xs = xx.reshape(-1)[mask_bools].astype(np.float32)
    ys = yy.reshape(-1)[mask_bools].astype(np.float32)
    ts = tt.reshape(-1)[mask_bools].astype(np.float32)
    ws = fg_mask_flat[mask_bools].astype(np.float32)
    pts = np.stack([xs, ys, ts], axis=1)

    # step 2: auto-tune eps from kNN distances
    k = min(5, len(pts)-1) if len(pts) > 5 else 1
    if k >= 1:
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(pts)
        distances, _ = nbrs.kneighbors(pts)
        base = np.median(distances[:, -1])
        eps = max(1.0, eps_scale * base)
    else:
        eps = 3.0

    # # step 3: cluster using DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(pts, sample_weight=ws)
    labels = db.labels_

    # step 4: create an f**w image
    mask = -1 * np.ones((f, h, w), dtype=np.int32)
    for i in range(len(labels)):
        if labels[i] != -1:
            x, y, t = int(xs[i]), int(ys[i]), int(ts[i])
            mask[t, y, x] = labels[i]

    return torch.tensor(mask)

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
    output = compute_foreground_mask(x0_pred, tau=0.1)
    output = compute_objects_masks(output)
    masses, edge_mass, com_positions, com_velocities = compute_physical_properties(output)

    return compute_losses_aux(masses, edge_mass, com_velocities)

__all__ = ['compute_losses']