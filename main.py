import os
import torch
from datetime import datetime

if __name__ == "__main__":
    experiment_name = "baseline"

    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    prompts = [
        "A fish swims forward in a steady line, its tail swaying side to side as it propels itself.",
        "A red octopus moving its tentacles around.",
        "A jellyfish swimming in shallow water. The jellyfish has a translucent body with a distinctive pattern of white circles and lines. It appears to be swimming just below the surface of the water, which is dark and murky due to the presence of algae or other aquatic plants.",
        "A dog playing with an orange ball with blue stripes. The dog picks up the ball and holds it in its mouth, conveying a sense of playfulness and energy. Throughout the video, the dog is seen playing with the ball, capturing the joy and excitement of the moment.",
        "A small dog playing with a red ball on a hardwood floor.",
        "A white kitten playing with a ball."
    ]

    seeds = [ '1024' ]

    addit_prompt = "A red octopus moving its tentacles around."

    # redirect output to a file
    with open(f"jobs-out-err/{experiment_name}_{datetime_str}.out", "w") as f:
        os.dup2(f.fileno(), 1)
    # redirect error output to a file
    with open(f"jobs-out-err/{experiment_name}_{datetime_str}.err", "w") as f:
        os.dup2(f.fileno(), 2)

    os.system(f"""python generate.py --task t2v-1.3B --size 832*480 --ckpt_dir ./Wan2.1-T2V-1.3B --prompts "{'" "'.join(prompts)}" --seeds {' '.join(seeds)} --experiment_name "{experiment_name}" """)

    # os.system(f"""python generate.py --task t2v-1.3B --size 832*480 --ckpt_dir ./Wan2.1-T2V-1.3B --prompts "{'" "'.join(prompts)}" --seeds {' '.join(seeds)} --paired_generation "True" --addit_prompt "{addit_prompt}" --experiment_name "{experiment_name}" """)