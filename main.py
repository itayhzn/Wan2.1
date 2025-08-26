import os
import torch


if __name__ == "__main__":
    # redirect output to a file
    # with open("output.txt", "w") as f:
    #     os.dup2(f.fileno(), 1)
    # # redirect error output to a file
    # with open("error.txt", "w") as f:
    #     os.dup2(f.fileno(), 2)

    prompts = [
        "Top-down white plane with a single ball moving steadily from left to right.",
        "Minimal diagram: one ball rolls across a white surface toward the right.",
        "On a flat white plane, a ball shifts horizontally from the left side to the right.",
        "White plane, one ball traveling diagonally down toward the right.",
        "Minimal schematic: a ball moves at an angle, top left to bottom right, across the white plane.",
        "Top-down white background, a single ball rolling diagonally downward and rightward.",
        "Two balls on a white plane, one from the left, one from the right, approaching each other.",
        "Minimal kinematics view: two balls roll toward one another from opposite horizontal directions.",
        "On a clean white plane, a ball moves rightward and another leftward, converging.",
        "Two balls on a white plane, moving side by side to the right without touching.",
        "Minimal diagram: both balls roll parallel rightward across the flat white background.",
        "Top-down view, two balls shift together toward the right in parallel paths.",
        "One ball moves upward, another downward, crossing paths on a white plane.",
        "Minimal top-down schematic: two balls roll vertically past each other, up and down.",
        "On a flat white plane, two balls travel opposite vertical directions and intersect.",
        "White plane with two balls approaching each other at an angle, converging toward impact.",
        "Minimal schematic: two balls move diagonally toward the same central point.",
        "On a clean plane, two balls roll from different angles and collide at the center.",
        "One ball moves rightward while another ascends diagonally upward from below.",
        "On a white background, two balls move: one horizontal right, one diagonal upward.",
        "Minimal kinematics diagram: two balls, paths crossing — one to the right, one upward diagonal.",
        "Two balls roll together left to right, side by side, across a white plane.",
        "Minimal schematic: both balls advance rightward in parallel, adjacent paths.",
        "Top-down white background with two balls aligned, moving together horizontally.",
        "One ball stands still on a white plane as another approaches it from the left.",
        "Minimal diagram: a stationary ball at center, a second ball rolls rightward toward it.",
        "On a flat plane, a ball moves from left toward a fixed ball.",
        "Two balls on opposite sides of a white plane, moving straight toward each other to collide.",
        "Minimal schematic: one ball goes left → right, another right → left, head-on at center.",
        "On a clean plane, two balls approach directly from opposite directions, colliding at midpoint.",
    ]

    seeds = [ '11' ]

    experiment_name = "physics-02"

    os.system(f"""python generate.py --task t2v-1.3B --size 832*480 --ckpt_dir ./Wan2.1-T2V-1.3B --prompts "{'" "'.join(prompts)}"  --seeds {' '.join(seeds)} --experiment_name "{experiment_name}" """)

