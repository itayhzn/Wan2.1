import os
import torch

if __name__ == "__main__":
    text_prompts = ["the horse jumping", "the person riding the horse"]

    args = samwise.get_samwise_args()
    model = samwise.build_samwise_model(args)

    input_path = '/storage/itaytuviah/SAMWISE/assets/example_video.mp4'  # Path to your video file

    save_path_prefix = os.path.join('demo_output')
    os.makedirs(save_path_prefix, exist_ok=True)

    samwise.inference(args, model, save_path_prefix, input_path, text_prompts)

    # redirect output to a file
    # with open("output.txt", "w") as f:
    #     os.dup2(f.fileno(), 1)
    # # redirect error output to a file
    # with open("error.txt", "w") as f:
    #     os.dup2(f.fileno(), 2)

    os.system("""python generate.py --task t2v-1.3B --size 832*480 --ckpt_dir ./Wan2.1-T2V-1.3B --prompts "A woman performing an intricate dance on stage, illuminated by a single spotlight in the first frame." --seeds 1024 """)

    os.system("""python generate.py --task t2v-1.3B --size 832*480 --ckpt_dir ./Wan2.1-T2V-1.3B --edit_mode "True" --input_paths "/storage/itaytuviah/SAMWISE/assets/example_video.mp4" --subject_prompts "the horse jumping" --edit_prompts "a blue zebra" --seeds 1024 --experiment_name "samwise_mask" """)