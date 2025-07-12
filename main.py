import os
import torch
from datetime import datetime

def read_videojam_prompts():
    """
    Read prompts from the videojam_prompts.txt file.
    Returns a list of prompts.
    """
    with open("videojam_prompts.txt", "r") as f:
        prompts = f.readlines()
    return [prompt.strip() for prompt in prompts if prompt.strip()]

if __name__ == "__main__":
    experiment_name = "selfx2eqx1"

    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    prompts = read_videojam_prompts()
    [
        "Two ibexes navigating a rocky hillside. They are walking down a steep slope covered in small rocks and dirt. In the background, there are more rocks and some greenery visible through an opening in the rocks.",
        "Athletic man doing gymnastics elements on horizontal bar in city park. Male sportsmen perform strength exercises outdoors.",
        "A small dog playing with a red ball on a hardwood floor.",
        "A white kitten playing with a ball.",
        "A woman enjoying the fun of hula hooping."
    ]

    seeds = [ '1024' ]

    addit_prompt =  "A cat." # "A cat." # "A red octopus moving its tentacles around."

    # redirect output to a file
    with open(f"jobs-out-err/{datetime_str}_{experiment_name}.out", "w") as f:
        os.dup2(f.fileno(), 1)
    # redirect error output to a file
    with open(f"jobs-out-err/{datetime_str}_{experiment_name}.err", "w") as f:
        os.dup2(f.fileno(), 2)

    os.system(f"""python generate.py --task t2v-1.3B --size 832*480 --ckpt_dir ./Wan2.1-T2V-1.3B --prompts "{'" "'.join(prompts)}" --seeds {' '.join(seeds)} --paired_generation "True" --addit_prompt "{addit_prompt}" --experiment_name "{experiment_name}" """)

    # os.system(f"""python generate.py --task t2v-1.3B --size 832*480 --ckpt_dir ./Wan2.1-T2V-1.3B --prompts "{'" "'.join(prompts)}" --seeds {' '.join(seeds)} --paired_generation "True" --addit_prompt "{addit_prompt}" --experiment_name "{experiment_name}" """)