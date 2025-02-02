# -*- coding: utf-8 -*-
"""clip-face_0822.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1G5SE-Mv1xi9vCQC6DCzyK1pDP2EISsjH

# 모델 불러오기
"""

import torch
from torch import nn
from transformers import AutoFeatureExtractor, AutoTokenizer, AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CLIP과 KoBERT 모델 불러오기
vit_model = AutoModel.from_pretrained("google/vit-base-patch16-224").to(device)
vit_processor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
kobert_model = AutoModel.from_pretrained("monologg/kobert").to(device)
kobert_tokenizer = AutoTokenizer.from_pretrained("monologg/kobert")


for param in kobert_model.parameters():
    param.requires_grad = False

# KoBERT의 마지막 레이어만 학습되도록 설정
# for name, param in kobert_model.named_parameters():
#     if "encoder.layer.11" in name or "pooler" in name:  # 마지막 레이어와 pooler는 학습 가능하게
#         param.requires_grad = True
#     else:
#         param.requires_grad = False

for param in kobert_model.parameters():
    param.requires_grad = False

# text model encoder output

text = "안녕하세요, GPT-4입니다."
text_inputs = kobert_tokenizer(text, return_tensors="pt").to(device)
text_outputs = kobert_model(**text_inputs)

text_last_hidden_state = text_outputs.last_hidden_state
text_pooled_output = text_outputs.pooler_output

print(text_last_hidden_state.shape)
print(text_pooled_output.shape)

from PIL import Image
import requests
from transformers import AutoProcessor, CLIPVisionModel

# image model encoder output
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

image_inputs = vit_processor(images=image, return_tensors="pt").pixel_values.to(device)

image_outputs = vit_model(image_inputs)
image_last_hidden_state = image_outputs.last_hidden_state
image_pooled_output = image_outputs.pooler_output  # pooled CLS states

print(image_last_hidden_state.shape)
print(image_pooled_output.shape)

image_inputs.shape

"""# Dataset & DataLoder"""

from google.colab import files
files.upload()  # kaggle.json 파일 업로드

# 이거 각자 kaggle account에서 Create API token해서 다운로드해서 여기 업로드한 후 진행

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Affectnet
!kaggle datasets download -d mstjebashazida/affectnet

# 압축 해제
import zipfile

with zipfile.ZipFile('affectnet.zip', 'r') as zip_ref:
    zip_ref.extractall('affectnet')

import os
import shutil

# 폴더 경로 설정
affectnet_path = '/content/affectnet/archive (3)'

# 1. Contempt 폴더 삭제
contempt_path = os.path.join(affectnet_path, 'Test', 'Contempt')
if os.path.exists(contempt_path):
    shutil.rmtree(contempt_path)

# 2. Anger 폴더 이름 변경
anger_path = os.path.join(affectnet_path, 'Test', 'Anger')
new_anger_path = os.path.join(affectnet_path, 'Test', 'anger')
os.rename(anger_path, new_anger_path)

# 3. 폴더 생성 및 이미지 이동
def move_images(src_folder, dst_folder):
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    for category in ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']:
        src_category = os.path.join(src_folder, category)
        dst_category = os.path.join(dst_folder, category)
        if os.path.exists(src_category):
            if not os.path.exists(dst_category):
                os.makedirs(dst_category)
            for file_name in os.listdir(src_category):
                src_file = os.path.join(src_category, file_name)
                dst_file = os.path.join(dst_category, file_name)
                shutil.move(src_file, dst_file)

# Test 및 Train 폴더 처리
content_data_path = '/content/data'
os.makedirs(content_data_path, exist_ok=True)

for folder in ['Test', 'Train']:
    folder_path = os.path.join(affectnet_path, folder)
    move_images(folder_path, content_data_path)

# 데이터 폴더 내 이미지 개수 출력
def count_files_in_folder(folder_path):
    return sum([len(files) for r, d, files in os.walk(folder_path)])

print("Number of images in content/data:")
for category in ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']:
    folder_path = os.path.join(content_data_path, category)
    print(f"{category}: {count_files_in_folder(folder_path)}")

# FER-2013
!kaggle datasets download -d astraszab/facial-expression-dataset-image-folders-fer2013

# 압축 해제
with zipfile.ZipFile('facial-expression-dataset-image-folders-fer2013.zip', 'r') as zip_ref:
    zip_ref.extractall('fer2013')

# FER-2013 폴더 경로 설정
fer2013_path = '/content/fer2013/data'
emotion_map = {
    '0': 'anger',
    '1': 'disgust',
    '2': 'fear',
    '3': 'happy',
    '4': 'sad',
    '5': 'surprise',
    '6': 'neutral'
}

def rename_folders(base_path):
    for subset in ['test', 'train', 'val']:
        subset_path = os.path.join(base_path, subset)
        for num, emotion in emotion_map.items():
            num_folder = os.path.join(subset_path, num)
            emotion_folder = os.path.join(subset_path, emotion)
            if os.path.exists(num_folder):
                os.rename(num_folder, emotion_folder)

def move_fer2013_images(base_path, target_path):
    for subset in ['test', 'train', 'val']:
        subset_path = os.path.join(base_path, subset)
        move_images(subset_path, target_path)

# 폴더 이름 변경
rename_folders(fer2013_path)

# FER-2013 이미지를 content/data로 이동
move_fer2013_images(fer2013_path, content_data_path)

# 데이터 폴더 내 이미지 개수 출력 최종 !!!
print("Number of images in content/data after FER-2013 merge:")
for category in ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']:
    folder_path = os.path.join(content_data_path, category)
    print(f"{category}: {count_files_in_folder(folder_path)}")

# 폴더와 파일 경로 설정
affectnet_folder = '/content/affectnet'
fer2013_folder = '/content/fer2013'
affectnet_zip = '/content/affectnet.zip'
fer2013_zip = '/content/facial-expression-dataset-image-folders-fer2013.zip'

# 폴더 삭제 함수
def delete_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"Deleted folder: {folder_path}")

# 파일 삭제 함수
def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Deleted file: {file_path}")

# 폴더 삭제
delete_folder(affectnet_folder)
delete_folder(fer2013_folder)

# 파일 삭제
delete_file(affectnet_zip)
delete_file(fer2013_zip)

print("Cleanup complete.")

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

def process_image(image):
    processed_image = vit_processor(images=image, return_tensors="pt").pixel_values.to(device)
    return processed_image.squeeze(0)

import os
import shutil
from sklearn.model_selection import train_test_split

# 원본 데이터 경로
data_dir = '/content/data'
# train, val, test 폴더를 생성할 기본 경로
output_dir = '/content/split_data'

train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# 클래스 폴더 리스트
class_folders = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# train, val, test 폴더 생성
for folder in ['train', 'val', 'test']:
    for class_folder in class_folders:
        os.makedirs(os.path.join(output_dir, folder, class_folder), exist_ok=True)


for class_folder in class_folders:
    # 원본 데이터의 클래스별 이미지 파일 목록 가져오기
    class_path = os.path.join(data_dir, class_folder)
    images = os.listdir(class_path)

    # train, test split
    train_images, temp_images = train_test_split(images, test_size=(val_ratio + test_ratio), random_state=42)
    # validation, test split
    val_images, test_images = train_test_split(temp_images, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42)

    # 이미지 파일을 각 폴더로 복사
    for image in train_images:
        src = os.path.join(class_path, image)
        dst = os.path.join(output_dir, 'train', class_folder, image)
        shutil.copy(src, dst)

    for image in val_images:
        src = os.path.join(class_path, image)
        dst = os.path.join(output_dir, 'val', class_folder, image)
        shutil.copy(src, dst)

    for image in test_images:
        src = os.path.join(class_path, image)
        dst = os.path.join(output_dir, 'test', class_folder, image)
        shutil.copy(src, dst)

print("Dataset split completed!")

# 기존 폴더 지저분하니 삭제
delete_folder('/content/data')

# colab에서 폴더 만들 때 자동으로 생성되는 불필요한 파일 제거
!rm -R /content/split_data/train.ipynb_checkpoints
!rm -R /content/split_data/val.ipynb_checkpoints
!rm -R /content/split_data/test.ipynb_checkpoints

size = (224, 224)

train_data_augmentation = transforms.Compose([
    transforms.Lambda(process_image),
    transforms.RandomResizedCrop(size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=2),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

val_data_augmentation = transforms.Compose([
    transforms.Lambda(process_image),
    transforms.CenterCrop(size),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Rescaling(scale=1.0 / 127.5, offset=-1) 적용  ...?
])

train_dir = "/content/split_data/train"
val_dir = "/content/split_data/val"
test_dir = "/content/split_data/test"

train_dataset = ImageFolder(
    root=train_dir,
    transform=train_data_augmentation
)
val_dataset = ImageFolder(
    root=val_dir,
    transform=val_data_augmentation
)
test_dataset = ImageFolder(
    root=test_dir,
    transform=transforms.Lambda(process_image)
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

from torch.utils.data import Subset
import numpy as np

# 함수 정의: 데이터셋에서 지정된 수량의 데이터를 무작위로 선택
def get_fixed_subset(dataset, num_samples):
    dataset_size = len(dataset)
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)  # 인덱스를 랜덤으로 섞음
    subset_indices = indices[:num_samples]
    return Subset(dataset, subset_indices)

# 각 데이터셋에서 원하는 수량의 이미지를 무작위로 선택
train_subset = get_fixed_subset(train_dataset, num_samples=140)
val_subset = get_fixed_subset(val_dataset, num_samples=40)
test_subset = get_fixed_subset(test_dataset, num_samples=20)

# DataLoader 생성
train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_subset, batch_size=16, shuffle=False)

for batch in train_loader:
    images, labels = batch
    print(images.shape)
    print(labels.shape)
    break

train_dataset.class_to_idx

label_to_class = {0:'화난', 1:'역겨운', 2:'두려운', 3:'행복한', 4:'중립적인', 5:'슬픈', 6:'놀란'}

# 2. [batch_size,] 형태의 라벨 (예시로 batch_size = 4)
labels = torch.tensor([0, 2, 1, 0])

# 3. 라벨을 클래스 이름으로 변환하고, 문장 생성
def generate_sentence(label):
    class_name = label_to_class[int(label)]
    sentence = f"이것은 {class_name} 표정입니다."
    return sentence

# 4. 배치 내의 모든 라벨에 대해 문장 생성
sentences = [generate_sentence(label) for label in labels]

# 5. 생성된 문장 출력
for sentence in sentences:
    print(sentence)

"""# Model & Loss 정의"""

def _get_vector_norm(tensor: torch.Tensor) -> torch.Tensor:
        square_tensor = torch.pow(tensor, 2)
        sum_tensor = torch.sum(square_tensor, dim=-1, keepdim=True)
        # normed_tensor = torch.pow(sum_tensor, 0.5)
        normed_tensor = torch.pow(sum_tensor, 0.5) + 1e-10  # 작은 값을 더해 0으로 나누는 문제를 방지
        return normed_tensor

import torch
import torch.nn as nn
import torch.optim as optim

class VisionTextModel(nn.Module):
    def __init__(self, vision_model, text_model, visual_projection_dim=512, text_projection_dim=512, num_classes=7, logit_scale=1.0):
        super(VisionTextModel, self).__init__()
        # vision_model.init_weights() # 가중치 초기화 제거
        self.vision_model = vision_model    # vit model은 초기화 해서 사용
        self.text_model = text_model

        self.visual_projection = nn.Linear(vision_model.config.hidden_size, visual_projection_dim)
        self.text_projection = nn.Linear(text_model.config.hidden_size, text_projection_dim)

        # # 최종 레이어: 512차원 임베딩을 7개의 클래스에 대한 출력으로 변환
        self.visual_classifier = nn.Linear(visual_projection_dim, num_classes)
        self.text_classifier = nn.Linear(text_projection_dim, num_classes)

        self.logit_scale = nn.Parameter(torch.tensor(logit_scale))

    def forward(self, images, texts, return_loss=False):
        # Get embeddings from both models
        vision_outputs = self.vision_model(images)
        text_outputs = self.text_model(**texts)

        # Extract the embeddings
        image_embeds = vision_outputs.pooler_output
        text_embeds = text_outputs.pooler_output

        # Project embeddings to a common space
        image_embeds = self.visual_projection(image_embeds)
        text_embeds = self.text_projection(text_embeds)

        # Normalize the embeddings
        image_embeds = image_embeds / _get_vector_norm(image_embeds)
        text_embeds = text_embeds / _get_vector_norm(text_embeds)

        # Calculate cosine similarity as logits
        logit_scale = self.logit_scale.exp().clamp(max=100)  # logit_scale의 상한을 둠
        logits_per_text = torch.matmul(text_embeds, image_embeds.t().to(text_embeds.device)) * logit_scale.to(text_embeds.device)
        logits_per_image = logits_per_text.t()

        return logits_per_text

class VisionTextModel(nn.Module):
    def __init__(self, vision_model, text_model, visual_projection_dim=512, text_projection_dim=512, num_classes=7, logit_scale=1.0):
        super(VisionTextModel, self).__init__()
        self.vision_model = vision_model
        self.text_model = text_model

        # MLP 구조 추가: 더 깊게 만듦
        self.visual_projection = nn.Sequential(
            nn.Linear(vision_model.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, visual_projection_dim)
        )

        self.text_projection = nn.Sequential(
            nn.Linear(text_model.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, text_projection_dim)
        )

        # 최종 레이어: 512차원 임베딩을 7개의 클래스에 대한 출력으로 변환
        self.visual_classifier = nn.Linear(visual_projection_dim, num_classes)
        self.text_classifier = nn.Linear(text_projection_dim, num_classes)

        self.logit_scale = nn.Parameter(torch.tensor(logit_scale))

    def forward(self, images, texts, return_loss=False):
        vision_outputs = self.vision_model(images)
        text_outputs = self.text_model(**texts)

        image_embeds = vision_outputs.pooler_output
        text_embeds = text_outputs.pooler_output

        # Project embeddings to a common space
        image_embeds = self.visual_projection(image_embeds)
        text_embeds = self.text_projection(text_embeds)

        # Normalize the embeddings
        image_embeds = image_embeds / _get_vector_norm(image_embeds)
        text_embeds = text_embeds / _get_vector_norm(text_embeds)

        # Calculate cosine similarity as logits
        logit_scale = self.logit_scale.exp().clamp(max=100)
        logits_per_text = torch.matmul(text_embeds, image_embeds.t().to(text_embeds.device)) * logit_scale.to(text_embeds.device)
        logits_per_image = logits_per_text.t()

        return logits_per_text

import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()

    def contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        text_loss = self.contrastive_loss(logits)
        image_loss = self.contrastive_loss(logits.t())

        # print(f'Text Loss: {text_loss.item()}')  # 텍스트 쌍의 손실 확인
        # print(f'Image Loss: {image_loss.item()}')  # 이미지 쌍의 손실 확인

        average_loss = (text_loss + image_loss) / 2.0

        return average_loss

model = VisionTextModel(vit_model, kobert_model, visual_projection_dim=512, text_projection_dim=512).to(device)

model_output = model(image_inputs, text_inputs)
model_output.shape    # N x N similarity matrix

"""# Training

"""

from torch.optim.lr_scheduler import StepLR

criterion = ContrastiveLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-5)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

import numpy as np

epochs = 100

train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)  # 라벨도 GPU로 이동
        texts = [generate_sentence(label) for label in labels]
        text_tokens = kobert_tokenizer(texts, padding=True, return_tensors="pt").to(device)

        optimizer.zero_grad()
        outputs = model(images, text_tokens)
        loss = criterion(outputs)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        _, predicted_labels = torch.max(outputs, dim=1)

        correct += (predicted_labels == torch.arange(len(outputs)).to(device)).sum().item()

        total += labels.size(0)
    scheduler.step()

    train_loss = running_loss / total
    train_losses.append(train_loss)
    train_accuracy = correct / total * 100

    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            texts = [generate_sentence(label) for label in labels]
            text_tokens = kobert_tokenizer(texts, padding=True, return_tensors="pt").to(device)

            outputs = model(images, text_tokens)
            loss = criterion(outputs)
            val_loss += loss.item() * labels.size(0)


            _, predicted_labels = torch.max(outputs, dim=1)
            val_correct += (predicted_labels == torch.arange(len(outputs)).to(device)).sum().item()
            val_total += labels.size(0)

    val_loss /= val_total
    val_losses.append(val_loss)
    val_accuracy = val_correct / val_total * 100

    print(f'Epoch {epoch+1}/{epochs}, '
          f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, '
          f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

model.eval()
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        texts = [generate_sentence(label) for label in labels]
        text_tokens = kobert_tokenizer(texts, padding=True, return_tensors="pt").to(device)

        print("Actual Labels:", labels)
        print("Generated Sentences:", texts)

        outputs = model(images, text_tokens)
        logits_per_text = outputs
        logits_per_image = logits_per_text.t()

        _, predicted_labels_text = torch.max(logits_per_text, dim=1)
        _, predicted_labels_image = torch.max(logits_per_image, dim=1)

        print("Predicted Labels (Text):", predicted_labels_text.cpu().numpy())
        print("Predicted Labels (Image):", predicted_labels_image.cpu().numpy())

        break  # 한 배치만 확인

