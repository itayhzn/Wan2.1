import os
import torch

if __name__ == "__main__":
    # redirect output to a file
    # with open("output.txt", "w") as f:
    #     os.dup2(f.fileno(), 1)
    # # redirect error output to a file
    # with open("error.txt", "w") as f:
    #     os.dup2(f.fileno(), 2)

    input_paths = [ 
        "datasets/woman_jumping_on_horse.mp4",
        "datasets/woman_performing_an_intricate_dance_on_stage.mp4",
        "datasets/athletic_man.mp4"
    ]
    subject_prompts = [
        "the horse jumping", 
        "the woman dancing",
        "the man doing gymnastics",
    ]
    edit_prompts = [
        "a blue zebra", 
        "A woman performing an intricate dance on stage, illuminated by a single spotlight in the first frame.",
        "Athletic man doing gymnastics elements on horizontal bar in city park. Male sportsmen perform strength exercises outdoors."
    ]
    idx = 2
    for lst in [input_paths, subject_prompts, edit_prompts]:
        # remove all elements except the one at index idx
        lst[:] = [lst[idx]] if idx < len(lst) else []
    
    print(f"input_paths: {input_paths}")
    print(f"subject_prompts: {subject_prompts}")
    print(f"edit_prompts: {edit_prompts}")


    base_seed = 11
    experiment_name = "athletic"

    # os.system(f"""python generate.py --task t2v-1.3B --size 832*480 --ckpt_dir ./Wan2.1-T2V-1.3B --prompts "{'" "'.join(edit_prompts)}" --seeds {base_seed} --experiment_name "{experiment_name}"  """)

    # os.system("""python generate.py --task t2v-1.3B --size 832*480 --ckpt_dir ./Wan2.1-T2V-1.3B --prompts "A woman performing an intricate dance on stage, illuminated by a single spotlight in the first frame." --seeds 1024 """)


    os.system(f"""python generate.py --task t2v-1.3B --size 832*480 --ckpt_dir ./Wan2.1-T2V-1.3B --edit_mode "True" --input_paths "{'" "'.join(input_paths)}" --subject_prompts "{'" "'.join(subject_prompts)}" --edit_prompts "{'" "'.join(edit_prompts)}" --base_seed {base_seed} --experiment_name "{experiment_name}" """)