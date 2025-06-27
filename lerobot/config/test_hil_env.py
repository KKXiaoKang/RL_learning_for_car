import gym_hil
print(f"gym_hil package location: {gym_hil.__file__}")

# 查看包的内容
import os
gym_hil_dir = os.path.dirname(gym_hil.__file__)
print(f"gym_hil directory: {gym_hil_dir}")
print("Files in gym_hil:")
for file in os.listdir(gym_hil_dir):
    print(f"  {file}")