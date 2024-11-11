import re
import os
import requests
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import torch
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision import transforms
from PIL import Image
from numpy import dot
from numpy.linalg import norm
import firebase_admin
from firebase_admin import credentials, storage
from io import BytesIO

# 모든 origin에 대해 CORS를 허용합니다.
app = Flask(__name__)
CORS(app, supports_credentials=True)
 
cred = credentials.Certificate('/Users/dust/Downloads/WEB Firebase(06.10)/webhotplace-1cce1-firebase-adminsdk-bfvwa-07f658a26f.json')
firebase_admin.initialize_app(cred, {
    'storageBucket': 'webhotplace-1cce1.appspot.com'
})

bucket = storage.bucket()

# EfficientNet-B0 모델 설정
model = efficientnet_b0(pretrained=True)  # Use pretrained weights from ImageNet
model = create_feature_extractor(model, return_nodes={'avgpool': 'avgpool'})
model.eval()

# 이미지 전처리 함수
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),  # 변경
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

# 데이터셋 로드 및 전처리
dataset_path = '/Users/dust/Downloads/WEB Firebase(06.10)/public/images_database'
all_image_paths = []

# os.walk를 사용하여 모든 이미지 파일 경로를 리스트에 추가
for root, _, files in os.walk(dataset_path):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  # 이미지 파일 확장자 필터링
            all_image_paths.append(os.path.join(root, file))
            
# 데이터셋이 올바르게 로드되었는지 확인
print(f"Total images in dataset: {len(all_image_paths)}")

# 이미지 특징 추출 함수
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

# 데이터셋 이미지 특징 벡터 추출 및 저장
dataset_features, image_paths = extract_features(all_image_paths)

# 코사인 유사도 계산 함수
def cos_sim(A, B):
    return dot(A, B) / (norm(A) * norm(B))

# 입력 이미지 처리 및 특징 벡터 추출 함수
def predict(image_path):
    image = Image.open(image_path).convert('RGB')
    image = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        image_feature = output['avgpool'].flatten().numpy()
    return image_feature

# 가장 유사한 이미지 찾기 함수
def find_most_similar_image(input_image_path):
    input_feature = predict(input_image_path)
    similarities = [cos_sim(input_feature, feature) for feature in dataset_features]
    if len(similarities) == 0:
        raise ValueError("No images found in the dataset or unable to extract features.")
    most_similar_idx = np.argmax(similarities)
    most_similar_image_path = image_paths[most_similar_idx]
    return most_similar_image_path

# find-similar 엔드포인트 정의
@app.route('/find-similar', methods=['POST', 'OPTIONS'])
def find_similar_image():
    if request.method == 'OPTIONS':
        response = jsonify({'message': 'Preflight request received'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response

    try:
        data = request.get_json()
        image_url = data.get('imageUrl')
        if not image_url:
            raise ValueError("No image URL provided")

        # 클라이언트에서 전송한 이미지 URL로부터 이미지를 다운로드하고 특징 추출
        response = requests.get(image_url)
        image_data = BytesIO(response.content)
        input_image = Image.open(image_data).convert('RGB')
        input_image_path = 'temp_input_image.jpg'  # 임시 파일로 저장 (실제 프로덕션 환경에서는 이 부분을 변경해야 함)
        input_image.save(input_image_path)

        # 가장 유사한 이미지 찾기
        most_similar_image_path = find_most_similar_image(input_image_path)

        # 이미지 파일명 추출 (예시: 'images_database/abc.jpg'에서 'abc.jpg' 부분)
        similar_image_name = os.path.basename(most_similar_image_path)

        # 결과를 JSON 형식으로 반환
        return jsonify({'similarImage': similar_image_name})  # 'similarImage' 키로 수정

    except Exception as e:
        print(f"Error in find_similar_image endpoint: {e}")
        return jsonify({'error': str(e)}), 500
    
# success.html 렌더링
@app.route('/success', methods=['GET'])
def success():
    similar_image_name = request.args.get('similar_image_name')
    print(f"Similar Image Name: {similar_image_name}")
    if similar_image_name:
        return render_template('success.html', similar_image_name=similar_image_name)
    else:
        return "Error: No similar image name provided", 400

if __name__ == '__main__':
    app.run('0.0.0.0', port=5002, debug=True)