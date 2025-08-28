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
        # "Top-down white plane with a single ball moving steadily from left to right.",
        # "Minimal diagram: one ball rolls across a white surface toward the right.",
        # "On a flat white plane, a ball shifts horizontally from the left side to the right.",
        # "White plane, one ball traveling diagonally down toward the right.",
        # "Minimal schematic: a ball moves at an angle, top left to bottom right, across the white plane.",
        # "Top-down white background, a single ball rolling diagonally downward and rightward.",
        # "Two balls on a white plane, one from the left, one from the right, approaching each other.",
        # "Minimal kinematics view: two balls roll toward one another from opposite horizontal directions.",
        # "On a clean white plane, a ball moves rightward and another leftward, converging.",
        
        "Two balls on a white plane, moving side by side to the right without touching.",
        
        # "Minimal diagram: both balls roll parallel rightward across the flat white background.",
        # "Top-down view, two balls shift together toward the right in parallel paths.",
        # "One ball moves upward, another downward, crossing paths on a white plane.",
        # "Minimal top-down schematic: two balls roll vertically past each other, up and down.",
        # "On a flat white plane, two balls travel opposite vertical directions and intersect.",
        # "White plane with two balls approaching each other at an angle, converging toward impact.",
        # "Minimal schematic: two balls move diagonally toward the same central point.",
        # "On a clean plane, two balls roll from different angles and collide at the center.",
        # "One ball moves rightward while another ascends diagonally upward from below.",
        # "On a white background, two balls move: one horizontal right, one diagonal upward.",
        # "Minimal kinematics diagram: two balls, paths crossing — one to the right, one upward diagonal.",
        
        "Two balls roll together left to right, side by side, across a white plane.",
        
        # "Minimal schematic: both balls advance rightward in parallel, adjacent paths.",
        # "Top-down white background with two balls aligned, moving together horizontally.",
        
        "One ball stands still on a white plane as another approaches it from the left.",
        
        # "Minimal diagram: a stationary ball at center, a second ball rolls rightward toward it.",
        # "On a flat plane, a ball moves from left toward a fixed ball.",
        # "Two balls on opposite sides of a white plane, moving straight toward each other to collide.",
        # "Minimal schematic: one ball goes left to right, another right to left, head-on at center.",
        # "On a clean plane, two balls approach directly from opposite directions, colliding at midpoint.",

        # "An overview of a white plane, a red ball moving from left to right at constant velocity",
        # "An overview of a white plane, a green ball moving diagonally from top left to bottom right at constant velocity",
        # "An overview of a white plane, a red ball moving from left to right and a blue ball moving from right to left. they collide at the center and bounce back symmetrically",
        
        "An overview of a white plane, a yellow ball and a purple ball moving side by side from left to right at the same constant velocity, never colliding",
        
        # "An overview of a white plane, a red ball moving upward and a blue ball moving downward. they cross paths at the center without colliding, both at constant velocity",
        # "An overview of a white plane, a green ball moving diagonally downward from top left and a red ball moving diagonally upward from bottom left. they collide at the middle point",
        # "An overview of a white plane, a red ball moving rightward while a blue ball moves diagonally upward from below. their paths cross without collision",
        # "An overview of a white plane, a blue ball and a yellow ball aligned side by side, moving together from left to right at equal constant velocity",
        # "An overview of a white plane, a green ball remaining stationary at the center while a red ball moves from left to right, colliding with it at the midpoint",
        
        "An overview of a white plane, a red ball moving from left to right and a blue ball moving from right to left. they collide head-on at the center and stop",
        
        # "An overview of a white plane, a red ball moving from left to right and a blue ball moving from top to bottom. The balls do not collide and both move at a constant velocity.",
        # "An overview of a white plane, a red ball moving from bottom to top at constant velocity.",
        # "An overview of a white plane, a blue ball moving diagonally from bottom left to top right at constant velocity.",
        # "An overview of a white plane, a red ball moving rightward and a green ball moving leftward. they collide in the center and both stop.",
        # "An overview of a white plane, a yellow ball rolling from top to bottom while a purple ball remains stationary at the center. they collide when the yellow ball reaches the center.",
        # "An overview of a white plane, a red ball moving from left to right and a blue ball moving diagonally downward from top right. they cross paths without collision.",
        
        "An overview of a white plane, a green ball moving diagonally from top right to bottom left at constant velocity.",
        
        # "An overview of a white plane, a red ball moving upward and a yellow ball moving downward. they collide at the middle and bounce back.",
        # "An overview of a white plane, a purple ball moving horizontally rightward and a blue ball moving diagonally upward from bottom left. their paths cross without contact.",
        # "An overview of a white plane, a red ball moving from left to right at constant velocity, a green ball moving from top to bottom. they collide at the center.",
        # "An overview of a white plane, a blue ball rolling leftward and a yellow ball rolling parallel to it right above, both at constant velocity, never colliding.",
        # "An overview of a white plane, a red ball moving diagonally upward from bottom left, a blue ball moving diagonally downward from top left. they collide midway.",
        
        "An overview of a white plane, a green ball moving from right to left while a purple ball remains stationary at the center. they collide when the green ball arrives.",
        
        # "An overview of a white plane, a red ball moving leftward and a blue ball moving rightward. they miss each other by moving along parallel horizontal lines.",
        # "An overview of a white plane, a yellow ball moving diagonally upward from bottom right and a green ball moving diagonally downward from top right. they cross without colliding.",
        # "An overview of a white plane, a red ball moving from left to right and a blue ball moving from top to bottom. they collide exactly at the center and rebound away.",
        # "An overview of a white plane, a purple ball moving upward and a green ball moving downward. they pass by without touching.",
        # "An overview of a white plane, a red ball moving diagonally upward from bottom left and a yellow ball moving from left to right. they collide near the top right corner.",
        # "An overview of a white plane, a blue ball rolling upward at constant velocity while a green ball remains fixed at the top center. they collide when the blue ball arrives.",
        # "An overview of a white plane, a red ball moving rightward and a blue ball moving leftward. they collide elastically at the center and exchange velocities.",
        # "An overview of a white plane, a yellow ball moving diagonally from bottom right to top left while a red ball moves horizontally from left to right. they cross paths but do not collide.",
        # "An overview of a white plane, a green ball moving from bottom to top at constant velocity.",
        # "An overview of a white plane, a red ball moving diagonally from top left to bottom right, a blue ball moving horizontally left to right. they collide near the center.",
        # "An overview of a white plane, a purple ball rolling diagonally upward from bottom left and a yellow ball moving downward from the top. they collide at the middle point.",
        # "An overview of a white plane, a red ball moving rightward and a green ball moving rightward slightly ahead of it. both roll parallel without touching.",
        # "An overview of a white plane, a blue ball moving downward while a red ball remains fixed at the bottom. they collide when the blue ball reaches the bottom edge.",
        
        "An overview of a white plane, a yellow ball rolling diagonally downward from top right and a purple ball moving upward from the bottom. they pass each other without colliding.",
        
        # "An overview of a white plane, a red ball moving rightward and a blue ball moving leftward. they approach and come to rest just before colliding.",
        # "An overview of a white plane, a green ball moving diagonally upward from bottom right and a red ball moving diagonally downward from top left. they collide midway.",
        
        "An overview of a white plane, a blue ball moving horizontally from left to right and a yellow ball moving vertically from bottom to top. they collide at the center and stop.",
        
        # "An overview of a white plane, a red ball moving diagonally upward from bottom left and a blue ball moving diagonally downward from top right. they cross paths without colliding.",
    ]

    seeds = [ '11' ]

    experiment_name = "losses-01"

    lr = 0.03
    iterations = 1
    start_step = 0
    end_step = 15
    loss_names = ['mass_1', 'mass_2', 'mass_3', 'mass_4', 'momentum_1', 'momentum_2', 'kinetic_energy_1', 'kinetic_energy_2']

    for loss_name in loss_names:
        os.system(f"""python generate.py --task t2v-1.3B --size 832*480 --ckpt_dir ./Wan2.1-T2V-1.3B --prompts "{'" "'.join(prompts)}"  --seeds {' '.join(seeds)} --experiment_name "{experiment_name}" --optimization_lr {lr} --optimization_iterations {iterations} --optimization_start_step {start_step} --optimization_end_step {end_step} --loss_name {loss_name}""")

