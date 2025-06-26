import os
import sys
import numpy as np
import time
from pathlib import Path
from datetime import datetime
from PIL import Image

# Import environment class
sys.path.append("./")
from script.eval_3dpolicy import Env

# Set random seed for reproducibility
np.random.seed(42)

# Set image save directory
def create_image_dir(task_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_dir = Path(f"./random_action_test/{task_name}_{timestamp}")
    image_dir.mkdir(parents=True, exist_ok=True)
    return image_dir

# Save observation images
def save_observation_images(observation, save_dir, step):
    obs_data = observation
    
    # Save all camera views
    for camera_name, camera_data in obs_data.items():
        if 'head_camera' in camera_data:
            img_array = camera_data['head_camera']['rgb']
            
            # Ensure data is uint8 type
            if img_array.dtype != np.uint8:
                if img_array.max() <= 1.0:
                    img_array = (img_array * 255).astype(np.uint8)
                else:
                    img_array = img_array.astype(np.uint8)
            
            # Save as PNG
            img = Image.fromarray(img_array)
            img_path = save_dir / f"{camera_name}_step_{step:04d}.png"
            img.save(img_path)
            print(f"Saved {camera_name} image to: {img_path}")

# Generate random action
def generate_random_action(action_dim=6):
    """Generate random action in range [-1, 1]"""
    return np.random.uniform(-0.5, 0.5, action_dim)

def main():
    # Test parameters
    task_name = "adjust_bottle"  # Task name
    task_config = "demo_clean"  # Task configuration
    head_camera = "D435"  # Camera type
    seed = 0  # Starting seed
    num_tasks = 3  # Number of tasks to test
    instruction_type = "unseen"  # Instruction type
    
    # Create save directory
    save_dir = create_image_dir(task_name)
    print(f"Test results will be saved to: {save_dir}")
    
    # Create environment
    print("\n=== Initializing Environment ===")
    env_manager = Env()
    
    try:
        # Create specific task environment and get valid seeds
        print(f"\n=== Finding {num_tasks} valid scenes for {task_name} task ===")
        result = env_manager.Create_env(
            task_name=task_name,
            head_camera_type=head_camera, 
            seed=seed,
            task_num=num_tasks,
            instruction_type=instruction_type,
            task_config=task_config
        )
        
        if not result:
            print("Failed to get valid seeds")
            return
        
        seed_list, id_list, episode_info_list_total = result
        print(f"Found {len(seed_list)} valid task seeds: {seed_list}")
        
        # Run random action test for each valid seed
        for i, (seed, task_id, episode_info_list) in enumerate(zip(seed_list, id_list, episode_info_list_total)):
            print(f"\n=== Executing task {i+1}/{len(seed_list)}, seed: {seed} ===")
            
            # Initialize task environment
            inst=env_manager.Init_task_env(seed, task_id, episode_info_list, len(seed_list))
            print(f"Task environment initialized successfully: {inst}")
            # Run random action sequence
            max_steps = 1000
            test_stats = {"success": False, "steps_taken": 0}
            
            for step in range(max_steps):
                # Get current observation
                observation = env_manager.get_observation()
                
                # Save observation images
                task_save_dir = save_dir / f"task_{i+1}_seed_{seed}"
                task_save_dir.mkdir(exist_ok=True)
                save_observation_images(observation, task_save_dir, step)
                
                # Generate random action sequence (here we generate one action at a time)
                # Note: Action dimension may need adjustment based on actual requirements
                action_dim = 14  # Adjust to appropriate dimension for your task (6 joints + 1 gripper)
                actions = np.array([generate_random_action(action_dim)])
                
                # Execute action
                # print(f"Step {step}: Execute random action {actions[0]}")
                status = env_manager.Take_action(actions)
                print(f"Status: {status}")
                
                # Check if completed
                if status != "run":
                    test_stats["success"] = (status == "success")
                    test_stats["steps_taken"] = step + 1
                    break
            
            # If still running, force close environment
            if status == "run":
                env_manager.Close_env()
                test_stats["steps_taken"] = max_steps
            
            # Save test statistics for this run
            result_file = task_save_dir / "result.txt"
            with open(result_file, "w") as f:
                f.write(f"Task: {task_name}\n")
                f.write(f"Seed: {seed}\n") 
                f.write(f"Success: {test_stats['success']}\n")
                f.write(f"Steps: {test_stats['steps_taken']}\n")
            
            print(f"Task {i+1} result: {'Success' if test_stats['success'] else 'Failed'}, Duration: {test_stats['steps_taken']} steps")
        
    except Exception as e:
        import traceback
        print(f"Error during testing: {e}")
        print(traceback.format_exc())
    
    print("\nTest completed!")

if __name__ == "__main__":
    main()