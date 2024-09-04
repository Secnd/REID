import os
import re
import shutil
from tqdm import tqdm
from typing import List
import config as cfg


def create_directories(output_dir: str) -> None:
    for subdir in ['train_rgb', 'gallery_rgb', 'train_ir', 'query_ir']:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)


def read_id_file(file_path: str) -> List[str]:
    with open(file_path, 'r') as file:
        ids = file.read().split(',')
        return [f'{int(id):04d}' for id in ids]


def get_image_files(base_dir: str, cameras: List[str], ids: List[str]) -> List[str]:
    image_files = []
    for id in sorted(ids):
        for cam in cameras:
            img_dir = os.path.join(base_dir, cam, id)
            if os.path.isdir(img_dir):
                image_files.extend(
                    [os.path.join(img_dir, img) for img in os.listdir(img_dir)]
                )
    return sorted(image_files)


def copy_images(image_files: List[str], target_dir: str, desc: str) -> None:
    pattern = re.compile(r'cam(\d+)[/\\](\d+)[/\\](\d+)')
    for img_path in tqdm(image_files, desc=desc):
        match = pattern.search(img_path)
        if match:
            camid, pid, imgid = match.groups()
            new_name = f'{pid}_c{camid}_{imgid}.jpg'
            shutil.copyfile(img_path, os.path.join(target_dir, new_name))
        else:
            print(f'Warning: Unexpected file path format: {img_path}')


def pre_process_sysu(
    base_dir: str = cfg.SUYU_DIR, output_dir: str = cfg.OUTPUT_DIR
) -> None:
    if all(
        os.path.isdir(os.path.join(output_dir, subdir))
        for subdir in ['train_rgb', 'gallery_rgb', 'train_ir', 'query_ir']
    ):
        print(
            f'Processed directories already exist in {output_dir}. Skipping preprocessing.'
        )
        return

    create_directories(output_dir)

    rgb_cameras = ['cam1', 'cam2', 'cam4', 'cam5']
    ir_cameras = ['cam3', 'cam6']

    id_train = read_id_file(os.path.join(base_dir, 'exp', 'train_id.txt'))
    id_val = read_id_file(os.path.join(base_dir, 'exp', 'val_id.txt'))
    id_test = read_id_file(os.path.join(base_dir, 'exp', 'test_id.txt'))

    id_train.extend(id_val)

    train_rgb = get_image_files(base_dir, rgb_cameras, id_train)
    train_ir = get_image_files(base_dir, ir_cameras, id_train)
    gallery_rgb = get_image_files(base_dir, rgb_cameras, id_test)
    query_ir = get_image_files(base_dir, ir_cameras, id_test)

    copy_images(train_rgb, os.path.join(output_dir, 'train_rgb'), 'train_rgb')
    copy_images(train_ir, os.path.join(output_dir, 'train_ir'), 'train_ir')
    copy_images(gallery_rgb, os.path.join(output_dir, 'gallery_rgb'), 'gallery_rgb')
    copy_images(query_ir, os.path.join(output_dir, 'query_ir'), 'query_ir')

    print(f'Preprocessing complete. Processed data saved in {output_dir}')


if __name__ == '__main__':
    pre_process_sysu()
