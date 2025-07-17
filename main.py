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
    experiment_name = "onemask_selfqkmasked"

    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # prompts = [
    #     "A fish swims forward in a steady line, its tail swaying side to side as it propels itself.",
    #     "A red octopus moving its tentacles around.",
    #     "A jellyfish swimming in shallow water. The jellyfish has a translucent body with a distinctive pattern of white circles and lines. It appears to be swimming just below the surface of the water, which is dark and murky due to the presence of algae or other aquatic plants.",
    #     "A dog playing with an orange ball with blue stripes. The dog picks up the ball and holds it in its mouth, conveying a sense of playfulness and energy. Throughout the video, the dog is seen playing with the ball, capturing the joy and excitement of the moment.",
    #     "Athletic man doing gymnastics elements on horizontal bar in city park. Male sportsmen perform strength exercises outdoors.",
    #     "A small dog playing with a red ball on a hardwood floor.",
    #     "A white kitten playing with a ball."
    # ]
    prompts = # read_file("videojam_prompts.txt")
    [ 
        "A small dog playing with a red ball on a hardwood floor."
    ]

    seeds = [ '1024' ]

    subject_prompts =  # read_file("videojam_subjects.txt")
    [
        'dog'
    ]

    edit_prompt =  "cat" # "A cat." # "A red octopus moving its tentacles around."

    # redirect output to a file
    # with open(f"jobs-out-err/{datetime_str}_{experiment_name}.out", "w") as f:
    #     os.dup2(f.fileno(), 1)
    # # redirect error output to a file
    # with open(f"jobs-out-err/{datetime_str}_{experiment_name}.err", "w") as f:
    #     os.dup2(f.fileno(), 2)

    os.system(f"""python generate.py --task t2v-1.3B --size 832*480 --ckpt_dir ./Wan2.1-T2V-1.3B --prompts "{'" "'.join(prompts)}" --seeds {' '.join(seeds)} --paired_generation "True" --subject_prompts "{'" "'.join(subject_prompts)}" --edit_prompt "{edit_prompt}" --experiment_name "{experiment_name}" """)

