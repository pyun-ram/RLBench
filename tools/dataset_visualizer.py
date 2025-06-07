import fire
from pathlib import Path
import numpy as np
import imageio
import pickle
from PIL import Image, ImageDraw, ImageFont

def read_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data

def read_image(image_path):
    img = Image.open(image_path).convert('RGB')
    return np.array(img)

def save_gif(image_list, save_path, duration=0.1):
    imgs = [Image.fromarray(img) for img in image_list]
    imgs[0].save(save_path, save_all=True, append_images=imgs[1:], duration=int(duration*1000), loop=0)
    return

def write_text_on_image(image: np.ndarray, text: str) -> np.ndarray:
    pil_img = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.load_default()
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    x = pil_img.width - text_width - 20
    y = 20
    draw.text((x, y), text, fill=(255, 255, 255), font=font)
    return np.array(pil_img)

class Runner:
    def visualize_one_episode(
        self,
        root_dir: str,
        save_path: str,
    ):
        root_dir = Path(root_dir)
        front_rgb_dir = root_dir / "front_rgb"
        overhead_rgb_dir = root_dir / "overhead_rgb"
        left_shoulder_rgb_dir = root_dir / "left_shoulder_rgb"
        right_shoulder_rgb_dir = root_dir / "right_shoulder_rgb"
        wrist_rgb_dir = root_dir / "wrist_rgb"

        steps = [itm.stem for itm in front_rgb_dir.iterdir() if itm.suffix == '.png']
        steps = sorted(steps, key=lambda x: int(x))
        variation = read_pkl(root_dir / 'variation_number.pkl')

        image_list = []
        for step in steps:
            images = []
            for cam_dir in [front_rgb_dir, overhead_rgb_dir, left_shoulder_rgb_dir, right_shoulder_rgb_dir]:
                image_path = cam_dir / f"{step}.png"
                image_ = read_image(image_path)
                images.append(image_)
            image = np.concatenate(images, axis=1)

            image = write_text_on_image(image, f"var {variation} step {step}")
            image_list.append(image)

        save_gif(image_list, save_path)
        return
    
    def visualize_episodes(
        self,
        task: str,
        root_dir: str,
        save_dir: str,
    ):
        for episode_dir in sorted(Path(root_dir).iterdir()):
            print(episode_dir)
            save_path = Path(save_dir) / f"{Path(root_dir).name}_{episode_dir.name}.gif"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            self.visualize_one_episode(episode_dir, save_path)
        return

if __name__ == "__main__":
    fire.Fire(Runner)
