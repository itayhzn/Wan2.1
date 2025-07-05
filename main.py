import os
import torch


if __name__ == "__main__":
    # redirect output to a file
    # with open("output.txt", "w") as f:
    #     os.dup2(f.fileno(), 1)
    # # redirect error output to a file
    # with open("error.txt", "w") as f:
    #     os.dup2(f.fileno(), 2)

    # x = torch.randn(3,4).cuda()
    # y = torch.randn(3,4).cuda()
    # print("Tensor created successfully:", x.shape, y.shape)

    # xyt = x @ y.T
    # print("x @ y.T", (x @ y.T).shape)

    # xy = x @ y # This will raise an error because the shapes are not aligned
    # print("x @ y", (x @ y).shape)


    # print('Hello World!')

    os.system("""python generate.py --task t2v-1.3B --size 832*480 --ckpt_dir ./Wan2.1-T2V-1.3B --prompt "A woman performing an intricate dance on stage, illuminated by a single spotlight in the first frame."   --base_seed 1024 """)

    # os.system("""python generate.py --task t2v-1.3B --size 832*480 --ckpt_dir ./Wan2.1-T2V-1.3B --prompts "A woman performing an intricate dance on stage, illuminated by a single spotlight in the first frame."   --seeds 1024 --paired_generation "True" --addit_prompt "a cat" --experiment_name "default" """)