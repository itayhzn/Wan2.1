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
        "A red ball rolling in a straight line across a smooth wooden floor.",
        "A green ball on the ground, suddenly struck by a mallet.",
        "A yellow ball colliding head-on with a wall.",
        "A blue ball released from rest at the top of a staircase.",
        "A red ball thrown diagonally upward into the air.",
        "Two identical metallic balls colliding on a smooth floor.",
        "A fast-moving green ball hitting a stationary red ball on a frictionless surface.",
        "Two equal-sized balls of different colors rolling toward each other and colliding.",
        "A red ball colliding with a blue ball of the same size.",
        "A line of five metallic balls hangs from strings; one end ball is pulled back and released.",
        
        "A smooth wooden floor, a red ball rolling at constant speed in a straight line with no obstacles, continuing until it exits the frame.",
        "A green ball is struck by a mallet; after the impact, it accelerates rapidly forward, with its acceleration proportional to the strength of the strike.",
        "A yellow ball collides head-on with a wall and bounces back at nearly the same speed but in the opposite direction.",
        "A blue ball is released from rest at the top of a staircase and falls vertically downwards, accelerating smoothly due to gravity.",
        "A red ball is thrown diagonally upward; it follows a curved parabolic trajectory, slows at the top, and falls back down to the ground.",
        "Two identical metallic balls collide elastically on a smooth floor, and after the collision, both balls remain whole and unbroken with unchanged size and mass.",
        "A fast-moving green ball strikes a stationary red ball on a frictionless surface; after collision, the green ball slows down while the red ball moves forward, total momentum preserved.",
        "Two equal-sized balls of different colors roll toward each other, collide, and stick together, moving as one combined ball afterward with momentum conserved.",
        "A red ball collides with a blue ball of the same size; both bounce apart with the same total kinetic energy before and after the collision.",
        "A line of five metallic balls hangs from strings; one end ball is pulled back and released, striking the row and causing the ball on the far end to swing out with equal speed.",
    ]

    seeds = [ '11' ]

    experiment_name = "physics-01"

    os.system(f"""python generate.py --task t2v-1.3B --size 832*480 --ckpt_dir ./Wan2.1-T2V-1.3B --prompts "{'" "'.join(prompts)}"  --seeds {' '.join(seeds)} --experiment_name "{experiment_name}" """)

