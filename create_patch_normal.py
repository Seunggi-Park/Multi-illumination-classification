import os
from PIL import Image
import json
import re
import random
# 이미지 파일들이 저장된 디렉토리 경로
image_directory = 'dataset_patches/sample_data'
# 자른 이미지를 저장할 디렉토리 경로
output_directory = 'dataset_patches/patches'

import cv2
import numpy as np
import random
import os

# 원본 이미지 파일이 저장된 폴더 경로
image_folder_path = "dataset/valid/Normal"
# 자른 이미지를 저장할 폴더 경로
save_folder_path = "dataset_patches/valid/Normal"

# 저장 폴더가 없다면 생성
os.makedirs(save_folder_path, exist_ok=True)

# 이미지 파일 이름 리스트 가져오기
image_files = sorted([f for f in os.listdir(image_folder_path) if f.endswith(".bmp")])

# 자를 패치 크기
patch_size = 200

# 첫 번째 이미지를 불러와서 크기를 확인
sample_image_path = os.path.join(image_folder_path, image_files[0])
sample_image = cv2.imread(sample_image_path)
height, width, _ = sample_image.shape

# 자를 위치를 랜덤으로 설정 (이미지 크기에서 200x200 패치가 들어갈 수 있는 범위로 제한)
x_start = random.randint(0, width - patch_size)
y_start = random.randint(0, height - patch_size)
# 10개의 이미지를 동일한 위치에서 자름 = 1
group_size = 10
num_groups = len(image_files) // group_size
globalid = 0
for x in range(20):
    for group_idx in range(num_groups):
        # 랜덤한 자를 위치를 설정 (이미지 크기에서 200x200 패치가 들어갈 수 있는 범위로 제한)
        sample_image_path = os.path.join(image_folder_path, image_files[group_idx * group_size])
        sample_image = cv2.imread(sample_image_path)
        height, width, _ = sample_image.shape

        x_start = random.randint(0, width - patch_size)
        y_start = random.randint(0, height - patch_size)

        # 해당 그룹의 10개의 이미지에 대해 동일한 좌표에서 자름
        for idx in range(group_size):
            image_file = image_files[group_idx * group_size + idx]
            image_path = os.path.join(image_folder_path, image_file)
            image = cv2.imread(image_path)

            # 동일한 좌표로 200x200 크기로 이미지 자르기
            cropped_image = image[y_start:y_start + patch_size, x_start:x_start + patch_size]

            # 숫자와 첫 번째 부분을 제거한 파일 이름 생성
            new_image_name = "_".join(image_file.split("_")[1:])  # 첫 번째 부분 제거
            cropped_image_name = f"{globalid + 1}_{new_image_name}"  # 같은 숫자를 공유

            cropped_image_path = os.path.join(save_folder_path, cropped_image_name)
            cv2.imwrite(cropped_image_path, cropped_image)

            #print(f"{image_file} 이미지를 {x_start}, {y_start}에서 잘라 {cropped_image_path}에 저장했습니다.")
        globalid += 1
print("모든 이미지를 동일한 위치에서 잘랐습니다.")