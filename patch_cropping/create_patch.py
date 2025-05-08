import os
from PIL import Image
import json
import re

# JSON 파일들이 저장된 디렉토리 경로
json_directory = 'C:/Users/brera/Desktop/dataset'
# # 이미지 파일들이 저장된 디렉토리 경로
# image_directory = 'dataset_patches/sample_data'
# # 자른 이미지를 저장할 디렉토리 경로
# output_directory = 'dataset_patches/patches'

# # 출력 디렉토리가 없다면 생성
# if not os.path.exists(output_directory):
#     os.makedirs(output_directory)

# 숫자를 추출하는 정규 표현식 패턴 (숫자가 하나 이상 포함된 부분을 추출)
def extract_number_from_filename(filename):
    return re.search(r'\d+', filename).group()

# 파일명에서 숫자를 제외한 나머지 부분 추출
def extract_name_without_number(filename):
    return re.sub(r'\d+', '', filename)

# 자르는 영역의 크기


def cut_patch(json_directory,image_directory,output_directory):
    CROP_SIZE = 200
# 전역적으로 증가하는 패치 번호 (이미지 파일을 넘어가도 계속 증가)
    global_patch_num = 1
    iter = 10
# 지정된 디렉토리의 모든 JSON 파일을 순회
    for i in range(iter):
        for filename in os.listdir(json_directory):
            if filename.endswith('.json'):
                json_file_path = os.path.join(json_directory, filename)

                # JSON 파일 열기
                with open(json_file_path, 'r') as f:
                    data = json.load(f)

                # JSON 파일에서 이미지 파일 경로가 포함되어 있을 수 있으므로 추출
                json_image_name = data.get('imagePath', None)

                if json_image_name:
                    # JSON 파일에서 이미지 파일의 숫자 추출
                    json_number = extract_number_from_filename(json_image_name)
                else:
                    # JSON 파일 이름에서 숫자 추출
                    json_number = extract_number_from_filename(filename)

                # 해당 숫자를 포함하는 모든 이미지 파일 찾기
                matching_image_files = []
                for image_file in os.listdir(image_directory):
                    if image_file.endswith(('.png', '.jpg', '.jpeg', '.bmp')):  # 이미지 파일 확장자
                        image_number = extract_number_from_filename(image_file)
                        if image_number == json_number:
                            matching_image_files.append(image_file)
                for matching_image_file in matching_image_files:
                    # 이미지 파일 경로
                    image_file_path = os.path.join(image_directory, matching_image_file)
                    image = Image.open(image_file_path)
                    image_width, image_height = image.size
                if matching_image_files:
                    # 각 이미지 파일에서 동일한 좌표에 대해 패치 번호를 기억할 딕셔너리
                    location_to_patch_num = {}
                    shapes = data['shapes']

                    for shape in shapes:
                        if shape['shape_type'] == 'rectangle':
                            points = shape['points']
                            # 좌표 추출
                            import random

                            # 좌표 설정
                            x1, y1 = points[0]
                            x2, y2 = points[1]

                            # 좌표를 정수로 변환
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                            # 좌표의 크기를 CROP_SIZE에 맞춰서 조정 (넘을 경우 자르고, 부족할 경우 확장)
                            width = x2 - x1
                            height = y2 - y1

                            # 랜덤한 정도로 자르거나 확장하는 요소 추가
                            if width > 0:
                                random_factor_x = random.randint(-width // 4, width // 4)
                            else:
                                # width가 0 이하인 경우 처리 방법을 결정 (예: 0으로 설정)
                                random_factor_x = 0
                            if height > 0:
                                random_factor_y =  random.randint(-height // 4, height // 4)
                            else:
                                # width가 0 이하인 경우 처리 방법을 결정 (예: 0으로 설정)
                                random_factor_y = 0
                            #random_factor_y = random.randint(-height // 4, height // 4)

                            # x축 자르기 및 확장 조정
                            if width > CROP_SIZE:
                                x2 = x1 + CROP_SIZE
                            elif width < CROP_SIZE:
                                # 좌우로 확장
                                pad_x = (CROP_SIZE - width) // 2
                                x1 = max(0, x1 - pad_x + random_factor_x)
                                x2 = min(image_width, x1 + CROP_SIZE)

                            # y축 자르기 및 확장 조정
                            if height > CROP_SIZE:
                                y2 = y1 + CROP_SIZE
                            elif height < CROP_SIZE:
                                # 상하로 확장
                                pad_y = (CROP_SIZE - height) // 2
                                y1 = max(0, y1 - pad_y + random_factor_y)
                                y2 = min(image_height, y1 + CROP_SIZE)

                            # 자를 영역
                            box = (x1, y1, x2, y2)

                            # 동일한 자르기 위치에 대해 같은 번호를 사용하기 위해 좌표 키 생성
                            location_key = (x1, y1)
                            # 해당 좌표가 이전에 자른 위치인지 확인
                            if location_key not in location_to_patch_num:
                                # 새로운 좌표일 경우, 새로운 패치 번호를 부여
                                location_to_patch_num[location_key] = global_patch_num
                                global_patch_num += 1  # 새로운 좌표일 때만 패치 번호 증가

                        for matching_image_file in matching_image_files:
                            # 이미지 파일 경로
                            image_file_path = os.path.join(image_directory, matching_image_file)

                            try:
                                # 이미지 열기
                                image = Image.open(image_file_path)
                                image_width, image_height = image.size

                                # 이미지 크기가 자를 크기보다 작으면 예외 처리
                                if image_width < CROP_SIZE or image_height < CROP_SIZE:
                                    print(f"Image {matching_image_file} is smaller than the crop size {CROP_SIZE}x{CROP_SIZE}. Skipping.")
                                    continue

                                # 원본 파일명에서 숫자를 제외한 부분 추출
                                original_filename_part = extract_name_without_number(matching_image_file)

                                # JSON에서 'shapes' 정보 추출


                                        # 자른 이미지 저장 (해당 좌표의 패치 번호 사용 + 원본 파일명 유지)

                                current_patch_num = location_to_patch_num[location_key]
                                cropped_image = image.crop(box)  # 이미지 자르기 수행
                                cropped_image_save_path = os.path.join(output_directory, f"{current_patch_num}{original_filename_part}")
                                cropped_image.save(cropped_image_save_path)
                                print(f"Cropped image saved to {cropped_image_save_path}")

                            except Exception as e:
                                print(f"Error processing {matching_image_file}: {str(e)}")


import os

# 원본 데이터셋 경로와 패치 데이터셋 경로 설정
dataset_dir = 'dataset'
dataset_patches_dir = 'dataset_patches'


def get_folder_paths(dataset_dir):
    folder_paths = []  # 세부 폴더 경로를 저장할 리스트

    for split in ['train', 'test', 'valid']:
        split_path = os.path.join(dataset_dir, split)

        if os.path.exists(split_path):  # split 폴더가 존재하는지 확인
            class_folders = os.listdir(split_path)  # split 폴더 내의 클래스 폴더 가져오기

            for class_folder in class_folders:
                class_path = os.path.join(split_path, class_folder)

                # 클래스 폴더 경로를 리스트에 추가
                folder_paths.append(class_path)
    return  folder_paths

# 원본 데이터셋의 폴더 구조를 읽어와서 패치 데이터셋 폴더에 동일하게 생성
def create_patch_folders(dataset_dir, dataset_patches_dir):
    for split in ['train', 'test', 'valid']:
        # split 폴더 경로 설정 (예: train, test, valid)
        split_path = os.path.join(dataset_dir, split)
        split_patches_path = os.path.join(dataset_patches_dir, split)

        # split 폴더가 존재하지 않으면 새로 생성
        if not os.path.exists(split_patches_path):
            os.makedirs(split_patches_path)

        # 각 split 폴더 내의 class 폴더들을 가져옴
        class_folders = os.listdir(split_path)
        for class_folder in class_folders:
            class_path = os.path.join(split_patches_path, class_folder)

            # class 폴더가 존재하지 않으면 새로 생성
            if not os.path.exists(class_path):
                os.makedirs(class_path)


# 함수 실행하여 폴더 구조 생성
create_patch_folders(dataset_dir, dataset_patches_dir)

# 함수 실행하여 폴더 경로 가져오기
folder_paths1 = get_folder_paths(dataset_dir)
folder_paths2 = get_folder_paths(dataset_patches_dir)

for k in range(len(folder_paths1)):
    print(folder_paths1[k], folder_paths2[k])
    cut_patch(json_directory,folder_paths1[k],folder_paths2[k])