import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from collections import defaultdict
import random

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = self._load_data()

    def _load_data(self):
        data = []
        for class_folder in os.listdir(self.root_dir):
            class_path = os.path.join(self.root_dir, class_folder)
            if os.path.isdir(class_path):
                # 이미지 파일들을 그룹핑할 딕셔너리
                image_groups = {}
                for file_name in os.listdir(class_path):
                    file_path = os.path.join(class_path, file_name)
                    if os.path.isfile(file_path):
                        # 파일 이름에서 번호 부분 추출
                        file_id = file_name.split('_')[0]
                        if file_id not in image_groups:
                            image_groups[file_id] = []
                        image_groups[file_id].append(file_path)

                # 번호별로 10개의 이미지를 가진 그룹만 데이터로 추가
                for file_id, image_paths in image_groups.items():
                    if len(image_paths) == 10:
                        image_paths = sorted(image_paths, key=self._sort_key)
                        label = class_folder
                        data.append((image_paths, label))
        return data

    def _sort_key(self, path):
        # Assuming the filenames start with a number
        filename = os.path.basename(path)
        num = filename.split('_')[0]
        return int(num)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_paths, label = self.data[idx]
        images = [Image.open(img_path) for img_path in image_paths]

        if self.transform:
            images = [self.transform(img) for img in images]

        return images, label

class CustomDatasetOver(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = self._load_data()
        self._balance_data()

    def _load_data(self):
        data = []
        for class_folder in os.listdir(self.root_dir):
            class_path = os.path.join(self.root_dir, class_folder)
            if os.path.isdir(class_path):
                # 이미지 파일들을 그룹핑할 딕셔너리
                image_groups = {}
                for file_name in os.listdir(class_path):
                    file_path = os.path.join(class_path, file_name)
                    if os.path.isfile(file_path):
                        # 파일 이름에서 번호 부분 추출
                        file_id = file_name.split('_')[0]
                        if file_id not in image_groups:
                            image_groups[file_id] = []
                        image_groups[file_id].append(file_path)

                # 번호별로 10개의 이미지를 가진 그룹만 데이터로 추가
                for file_id, image_paths in image_groups.items():
                    if len(image_paths) == 10:
                        image_paths = sorted(image_paths, key=self._sort_key)
                        label = class_folder
                        data.append((image_paths, label))
        return data

    def _balance_data(self):
        # 각 클래스별 데이터 수를 카운트
        label_counts = defaultdict(list)
        for images, label in self.data:
            label_counts[label].append((images, label))

        # 최대 데이터 수를 가진 레이블의 수를 기준으로 데이터셋을 오버샘플링
        max_count = max(len(v) for v in label_counts.values())

        balanced_data = []
        for label, items in label_counts.items():
            # 부족한 수만큼 아이템을 랜덤하게 복제해서 추가
            while len(items) < max_count:
                items.append(random.choice(items))
            balanced_data.extend(items)

        self.data = balanced_data

    def _sort_key(self, path):
        # Assuming the filenames start with a number
        filename = os.path.basename(path)
        num = filename.split('_')[0]
        return int(num)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_paths, label = self.data[idx]
        images = [Image.open(img_path) for img_path in image_paths]

        if self.transform:
            images = [self.transform(img) for img in images]

        return images, label
class CustomDatasetUnder(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = self._load_data()
        self._balance_data()

    def _load_data(self):
        data = []
        for class_folder in os.listdir(self.root_dir):
            class_path = os.path.join(self.root_dir, class_folder)
            if os.path.isdir(class_path):
                # 이미지 파일들을 그룹핑할 딕셔너리
                image_groups = {}
                for file_name in os.listdir(class_path):
                    file_path = os.path.join(class_path, file_name)
                    if os.path.isfile(file_path):
                        # 파일 이름에서 번호 부분 추출
                        file_id = file_name.split('_')[0]
                        if file_id not in image_groups:
                            image_groups[file_id] = []
                        image_groups[file_id].append(file_path)

                # 번호별로 10개의 이미지를 가진 그룹만 데이터로 추가
                for file_id, image_paths in image_groups.items():
                    if len(image_paths) == 10:
                        image_paths = sorted(image_paths, key=self._sort_key)
                        label = class_folder
                        data.append((image_paths, label))
        return data

    def _balance_data(self):
        # 각 클래스별 데이터 수를 카운트
        label_counts = defaultdict(list)
        for images, label in self.data:
            label_counts[label].append((images, label))

        # 최소 데이터 수를 가진 레이블의 수를 기준으로 데이터셋을 언더샘플링
        min_count = min(len(v) for v in label_counts.values())

        balanced_data = []
        for label, items in label_counts.items():
            # 일정하게 앞에서부터 min_count 개수만큼 선택
            items = sorted(items)[:min_count]
            balanced_data.extend(items)

        self.data = balanced_data

    def _sort_key(self, path):
        # Assuming the filenames start with a number
        filename = os.path.basename(path)
        num = filename.split('_')[0]
        return int(num)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_paths, label = self.data[idx]
        images = [Image.open(img_path) for img_path in image_paths]

        if self.transform:
            images = [self.transform(img) for img in images]

        return images, label
class CustomDataset2(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = self._load_data()
    def _load_data(self):
        data = []
        for class_folder in os.listdir(self.root_dir):
            class_path = os.path.join(self.root_dir, class_folder)
            if os.path.isdir(class_path):
                # 이미지 파일들을 그룹핑할 딕셔너리
                image_groups = {}
                for file_name in os.listdir(class_path):
                    file_path = os.path.join(class_path, file_name)
                    if os.path.isfile(file_path):
                        # 파일 이름에서 번호 부분 추출
                        file_id = file_name.split('_')[0]
                        if file_id not in image_groups:
                            image_groups[file_id] = []
                        image_groups[file_id].append(file_path)

                # 번호별로 10개의 이미지를 가진 그룹만 데이터로 추가
                for file_id, image_paths in image_groups.items():
                    if len(image_paths) == 10:
                        image_paths = sorted(image_paths, key=self._sort_key)
                        label = class_folder
                        if label in ['BrightLine', 'Deformation',  'Dent', 'Scratch']:  # label이 "Normal"인 경우만 추가
                            data.append((image_paths, label))
        return data

    def _sort_key(self, path):
        # Assuming the filenames start with a number
        filename = os.path.basename(path)
        num = filename.split('_')[0]
        return int(num)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_paths, label = self.data[idx]
        images = [Image.open(img_path) for img_path in image_paths]

        if self.transform:
            images = [self.transform(img) for img in images]

        return images, label
# Define the transformations for the images
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
transform2 = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor()
])
transform3 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Create the dataset and dataloader
# dataset = CustomDataset(root_dir='dataset/train', transform=transform)
# dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
# dataset = CustomDataset(root_dir='dataset/train', transform=transform)
#
# # Example of loading one batch of data and displaying the images
# images, label = dataset[0]  # 첫 번째 데이터 항목 로드
# print(f'Label: {label}')
# fig, axes = plt.subplots(1, 10, figsize=(20, 2))
#
# for i, img in enumerate(images):
#     img = img.permute(1, 2, 0).clamp(0, 1).numpy()  # 채널 순서를 변경하고 값 범위를 조정
#     axes[i].imshow(img)
#     axes[i].axis('off')
#
# plt.suptitle(f'Class: {label}')
# plt.show()