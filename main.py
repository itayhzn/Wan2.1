import os
# if using Apple MPS, fall back to CPU for unsupported ops
from datetime import datetime
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2

from wan.modules.vae import WanVAE

def read_file(file_path):
    """
    Read a text file and return its content as a list of lines.
    """
    with open(file_path, "r") as f:
        lines = f.readlines()
    return [line.strip() for line in lines if line.strip()]


if __name__ == "__main__":
    experiment_name = "combos-07"

    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    prompts =  [ read_file("videojam_prompts.txt")[5] ]

    seeds = [ '11' ]

    subject_prompts = [ read_file("videojam_subjects.txt")[5] ]

    edit_prompts =  ["A cat."] * len(prompts) 

    # redirect output to a file
    with open(f"jobs-out-err/{datetime_str}_{experiment_name}.out", "w") as f:
        os.dup2(f.fileno(), 1)
    # # redirect error output to a file
    with open(f"jobs-out-err/{datetime_str}_{experiment_name}.err", "w") as f:
        os.dup2(f.fileno(), 2)

    for cross_attn_option in [7,0,1]: #range(5):
        for timestep in [0,2,4,6,8]: # [0, 11, 2]: #range(0, 20, 2):
            for self_attn_option in [5,6,0,1,2]: #range(13):
                print(f"Running with timestep {timestep}, self_attn_option {self_attn_option}, cross_attn_option {cross_attn_option}")
                os.system(f"""python generate.py --task t2v-1.3B --size 832*480 --ckpt_dir ./Wan2.1-T2V-1.3B --prompts "{'" "'.join(prompts)}" --seeds {' '.join(seeds)} --paired_generation "True" --subject_prompts "{'" "'.join(subject_prompts)}" --edit_prompts "{'" "'.join(edit_prompts)}" --experiment_name "{experiment_name}" --timestep_for_edit {timestep} --self_attn_option {self_attn_option} --cross_attn_option {cross_attn_option} """)

