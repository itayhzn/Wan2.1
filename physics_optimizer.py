import gc
import torch
import torch.utils.checkpoint as checkpoint
import types
import physics_invariants
from utils import log_losses


class GradientCheckpointer:
    """
    Adds gradient checkpointing to models that don't have native support.
    Ensures proper gradient flow.
    """
    def __init__(self, model):
        self.model = model
        self.original_forward = model.forward
        self.is_active = False
    
    def enable(self):
        """Enable gradient checkpointing by patching the forward method"""
        if self.is_active:
            return
            
        def checkpointed_forward(self_model, latent_model_input, t, **kwargs):
            """Wrapped forward function with gradient checkpointing"""
            # Get the first element of latent_model_input
            latent = latent_model_input[0]
            
            def custom_forward(x, timestep):
                # Create list format expected by original forward
                latent_list = [x]
                
                # Call original forward
                output = self.original_forward(latent_list, t=timestep, **kwargs)
                
                # Properly handle output format
                if isinstance(output, (list, tuple)):
                    result = output[0]
                else:
                    result = output
                    
                return result
                
            # Apply checkpointing with proper preserve_rng_state flag
            return checkpoint.checkpoint(
                custom_forward, 
                latent,
                t,
                preserve_rng_state=True  # Important for consistent results
            )
            
        # Replace the forward method
        self.model.forward = types.MethodType(checkpointed_forward, self.model)
        self.is_active = True
        
    def disable(self):
        """Restore original forward method"""
        if not self.is_active:
            return
            
        # Restore original forward method
        self.model.forward = self.original_forward
        self.is_active = False

class Optimizer:
    def __init__(self, 
                 iterations=1,
                 lr=0.03,
                 diffusion_steps_to_optimize=[],
                 use_softmax_mean=True,
                 temperature=10.0,
                 model=None,
                 arg_c=None,
                 arg_null=None,
                 guide_scale=None,
                 loss_name=None,
                 encoded_params=None):
        self.iterations = iterations
        self.lr = lr
        self.diffusion_steps_to_optimize = diffusion_steps_to_optimize
        self.use_softmax_mean = use_softmax_mean
        self.temperature = temperature
        self.loss_name = loss_name
        self.model = model
        self.arg_c = arg_c
        self.arg_null = arg_null
        self.guide_scale = guide_scale
        self.encoded_params = encoded_params

    def print_shapes(self, x, name, counter=0):
        # debugging
        if isinstance(x, list):
            print(f"1{counter} - {name} is a list, {name}[0].shape: {x[0].shape}")
        if isinstance(x, torch.Tensor):
            print(f"1{counter} - {name} is a tensor, {name}.shape: {x.shape}")

    def optimize(self, latents, t, t_idx, noise_pred, sigma):
        # debugging
        self.print_shapes(latents, 'latents', 0)
        self.print_shapes(noise_pred, 'noise_pred', 1)
        x0_pred = latents[0] - sigma * noise_pred
        self.print_shapes(x0_pred, 'x0_pred', 2)
        losses = physics_invariants.compute_losses(x0_pred)
        log_losses(self.encoded_params+'.txt', losses, t_idx)

        if t_idx not in self.diffusion_steps_to_optimize or \
           self.loss_name is None or \
           self.loss_name not in losses.keys():
            return latents
        
        # should optimize if got here

        checkpointer = GradientCheckpointer(self.model)
        checkpointer.enable()

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        latent = latents[0].detach().clone().requires_grad_(True)
        self.print_shapes(latent, 'latent', 3)
        optimizer = torch.optim.Adam([latent], lr=self.lr)

        with torch.enable_grad():
            for i in range(self.iterations):
                optimizer.zero_grad()
                # Compute loss and gradients
                
                latent_model_input = [latent]
                self.print_shapes(latent_model_input, 'latent_model_input', 4)
                timestep = [t]
                timestep = torch.stack(timestep)

                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, **self.arg_c)[0]
                self.print_shapes(noise_pred_cond, 'noise_pred_cond', 5)
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, **self.arg_null)[0]
                self.print_shapes(noise_pred_uncond, 'noise_pred_uncond', 6)

                noise_pred = noise_pred_uncond + self.guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                self.print_shapes(noise_pred, 'noise_pred', 7)
                x0_pred = latent - sigma * noise_pred
                self.print_shapes(x0_pred, 'x0_pred', 8)

                losses = physics_invariants.compute_losses(x0_pred)
                loss = losses[self.loss_name]

                loss.backward(retain_graph=False)
                optimizer.step()

        checkpointer.disable()

        return [latent.detach()]
