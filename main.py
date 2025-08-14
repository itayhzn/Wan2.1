import os
import torch

if __name__ == "__main__":
    # redirect output to a file
    # with open("output.txt", "w") as f:
    #     os.dup2(f.fileno(), 1)
    # # redirect error output to a file
    # with open("error.txt", "w") as f:
    #     os.dup2(f.fileno(), 2)

    input_paths = [ "datasets/woman_jumping_on_horse.mp4" ]
    subject_prompts = ["the horse jumping"]
    edit_prompts = ["a blue zebra"]
    base_seed = 1024
    experiment_name = "maskgen_noedit"

    # os.system(f"""python generate.py --task t2v-1.3B --size 832*480 --ckpt_dir ./Wan2.1-T2V-1.3B --prompts "{'" "'.join(edit_prompts)}" --seeds {base_seed} --experiment_name "{experiment_name}"  """)

    os.system("""python generate.py --task t2v-1.3B --size 832*480 --ckpt_dir ./Wan2.1-T2V-1.3B --prompts "A woman performing an intricate dance on stage, illuminated by a single spotlight in the first frame." --seeds 1024 """)


    # os.system(f"""python generate.py --task t2v-1.3B --size 832*480 --ckpt_dir ./Wan2.1-T2V-1.3B --edit_mode "True" --input_paths "{'" "'.join(input_paths)}" --subject_prompts "{'" "'.join(subject_prompts)}" --edit_prompts "{'" "'.join(edit_prompts)}" --base_seed {base_seed} --experiment_name "{experiment_name}" """)