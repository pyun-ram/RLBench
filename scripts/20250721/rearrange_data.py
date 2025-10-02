import os
from pathlib import Path    
if __name__ == "__main__":
    root_dir = 'data/20250721_robot_data/train/reach_single_moving_target_on_the_table_cpst/all_variations/episodes/'
    episode_dirs = Path(root_dir).glob("episode*")
    episode_dirs = sorted(episode_dirs, key=lambda x: int(x.name.split("episode")[-1]))
    tmp_dir = Path("./tmp/rearrange_data")
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    for new_episode_id, episode_dir in enumerate(episode_dirs):
        new_episode_id = new_episode_id + 87
        new_episode_dir = tmp_dir/ f"episode{new_episode_id}"
        os.system(f"mv {episode_dir} {new_episode_dir}")
    os.system(f"mv {tmp_dir}/* {root_dir}/")
    # os.system(f"rm -rf {tmp_dir}")

