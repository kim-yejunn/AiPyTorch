import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision.models import resnet18
from torchvision.models.resnet import ResNet18_Weights

from torchvision import transforms, models
from PIL import Image
import os

# GPU 사용 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 이미지 변환 설정
transform = transforms.Compose([
    transforms.Resize((224, 224)),                  # 이미지 크기 조정
    transforms.ToTensor(),                          # 이미지를 PyTorch 텐서로 변환
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 이미지를 정규화
])

# 이미지 데이터셋 클래스
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, img) for img in os.listdir(root_dir) if os.path.splitext(img)[1].lower() in ['.png', '.jpg', '.jpeg']]
        self.images = [Image.open(img_path) for img_path in self.image_paths]
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# 신경망 모델 정의 (사전 훈련된 ResNet 사용)
class ImageSimilarityModel(nn.Module):
    def __init__(self, num_classes):
        super(ImageSimilarityModel, self).__init__()
        self.resnet = models.resnet18(pretrained=True)  # 사전 훈련된 ResNet 모델 불러오기
        self.fc = nn.Linear(1000, num_classes)           # 출력 레이어 정의 (이미지 유사성 분류)

    def forward(self, x):
        x = self.resnet(x)          # ResNet 모델의 특징 추출기로 이미지 특징 추출
        x = self.fc(x)              # 추출된 특징을 기반으로 클래스 분류
        return x

# 이미지 폴더 경로
image_folder = "downloaded_images"

# 데이터셋 생성
dataset = CustomDataset(image_folder, transform=transform)

# 데이터로더 생성
data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

# 모델 초기화
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # 데이터셋의 클래스 수에 맞게 모델 초기화
model.to(device)                            # 모델을 GPU로 옮기기

# 손실 함수 및 최적화 알고리즘 정의
criterion = nn.CrossEntropyLoss()           # 분류 문제이므로 CrossEntropyLoss 사용
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # 확률적 경사하강법 사용

# 모델 학습
num_epochs = 3
for epoch in range(num_epochs):
    running_loss = 0.0
    for images in data_loader:
        images = images.to(device)         # 이미지를 GPU로 옮기기
        
        # 순전파
        outputs = model(images)
        
        # 손실 계산
        loss = criterion(outputs, torch.tensor([0]))  # 임의의 레이블 사용
        
        # 역전파 및 가중치 업데이트
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss}")
    

# 모델 저장
torch.save(model.state_dict(), "model.pth")

# 예측을 위한 모델 평가 모드 설정
model.eval()

# 유사한 이미지 찾기
def find_similar_image(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)  # 이미지 변환 및 GPU로 옮기기
    with torch.no_grad():
        output = model(image)
    _, predicted = torch.max(output, 1)  # 가장 높은 확률을 갖는 클래스 선택
    return dataset.image_paths[predicted.item()]  # 선택된 클래스에 해당하는 이미지 경로 반환

# 첫 번째 이미지를 테스트 이미지로 사용
test_image_path = dataset.image_paths[7]

# 유사한 이미지 찾기
similar_image = find_similar_image(test_image_path)
print(f"Most similar image: {similar_image}")
