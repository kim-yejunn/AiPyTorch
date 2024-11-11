import os
import numpy as np
import torch
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision import transforms
from PIL import Image
from numpy import dot
from numpy.linalg import norm
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# 1. EfficientNet-B0 모델 설정
weights = EfficientNet_B0_Weights.DEFAULT
model = efficientnet_b0(weights=weights)
model = create_feature_extractor(model, return_nodes={'avgpool': 'avgpool'})
model.eval()

# 2. 이미지 전처리 함수
preprocess = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

# 3. 데이터셋 로드 및 전처리
dataset_path = 'images_database'
all_image_paths = []

# os.walk를 사용하여 모든 이미지 파일 경로를 리스트에 추가
for root, _, files in os.walk(dataset_path):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  # 이미지 파일 확장자 필터링
            all_image_paths.append(os.path.join(root, file))

# 4. 이미지 특징 추출 함수
def extract_features(image_paths):
    features = []
    paths = []
    with torch.no_grad():
        for image_path in image_paths:
            image = Image.open(image_path).convert('RGB')
            image = preprocess(image).unsqueeze(0)
            output = model(image)
            features.append(output['avgpool'].flatten().numpy())
            paths.append(image_path)
    return np.array(features), paths

# 5. 데이터셋 이미지 특징 벡터 추출 및 저장
dataset_features, image_paths = extract_features(all_image_paths)

# 6. 코사인 유사도 계산 함수
def cos_sim(A, B):
    return dot(A, B) / (norm(A) * norm(B))

# 7. 입력 이미지 처리 및 특징 벡터 추출 함수
def predict(image_path):
    image = Image.open(image_path).convert('RGB')
    image = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        image_feature = output['avgpool'].flatten().numpy()
    return image_feature

# 8. 가장 유사한 이미지 찾기 함수
def find_most_similar_image(input_image_path):
    input_feature = predict(input_image_path)
    similarities = [cos_sim(input_feature, feature) for feature in dataset_features]
    most_similar_idx = np.argmax(similarities)
    most_similar_image_path = image_paths[most_similar_idx]
    return most_similar_image_path

@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files['image']
    if file:
        file_path = 'uploaded_image.jpg'
        file.save(file_path)
        most_similar_image = find_most_similar_image(file_path)
        return jsonify({
            'input_image': file_path,
            'most_similar_image': most_similar_image
        })
    return 'No file uploaded', 400

if __name__ == '__main__':
    app.run(debug=True)
