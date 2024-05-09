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
    transforms.Resize((224, 224)),                  
    transforms.ToTensor(),                          
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])

# 이미지 데이터셋 클래스
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, img) for img in os.listdir(root_dir) if os.path.splitext(img)[1].lower() in ['.png', '.jpg', '.jpeg']]
        self.images = [Image.open(img_path).convert('RGB') for img_path in self.image_paths]
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        return image

# 신경망 모델 정의 (사전 훈련된 ResNet 사용)
class ImageSimilarityModel(nn.Module):
    def __init__(self, num_classes):
        super(ImageSimilarityModel, self).__init__()
        self.resnet = models.resnet18(pretrained=True)  
        self.fc = nn.Linear(1000, num_classes)           

    def forward(self, x):
        x = self.resnet(x)          
        x = self.fc(x)              
        return x

# 이미지 폴더 경로
image_folder = "downloaded_images"

# 데이터셋 생성
dataset = CustomDataset(image_folder, transform=transform)

# 데이터로더 생성
data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

# 모델 초기화
model = ImageSimilarityModel(num_classes=len(dataset))  # 클래스 수에 맞게 모델 초기화
model.to(device)                           

# 손실 함수 및 최적화 알고리즘 정의
criterion = nn.CrossEntropyLoss()           
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  

# 모델 학습
num_epochs = 3
for epoch in range(num_epochs):
    running_loss = 0.0
    for images in data_loader:
        images = images.to(device)         
        outputs = model(images)
        labels = torch.randint(0, len(dataset), (1,)).to(device)  # 랜덤 레이블 사용
        
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
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
    image = transform(image).unsqueeze(0).to(device)  
    with torch.no_grad():
        output = model(image)
    _, predicted = torch.max(output, 1)  
    return dataset.image_paths[predicted.item()]  

# 테스트 이미지로 사용할 이미지 선택
test_image_path = dataset.image_paths[20]

# 유사한 이미지 찾기
similar_image = find_similar_image(test_image_path)
print(f"Most similar image: {similar_image}")
