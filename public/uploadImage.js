import { initializeApp } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-app.js";
import { getStorage, ref, uploadBytesResumable, getDownloadURL } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-storage.js";

const firebaseConfig = {
    apiKey: "AIzaSyAWoNvLsqbVzfDMpoubR1e7t6Ny4Xxk0JY",
    authDomain: "webhotplace-1cce1.firebaseapp.com",
    projectId: "webhotplace-1cce1",
    storageBucket: "webhotplace-1cce1.appspot.com",
    messagingSenderId: "331407528116",
    appId: "1:331407528116:web:b4f6a50ec1587b9c4707e1",
    measurementId: "G-N5Y9J96L8M"
};

// Firebase 초기화
const app = initializeApp(firebaseConfig);
const storage = getStorage(app);

// 이미지 입력 및 업로드 처리
$('#imageInput').on('change', function() {
    var file = this.files[0];

    var storageRef = ref(storage, file.name);
    var uploadTask = uploadBytesResumable(storageRef, file);
    
    // 업로드 상태를 모니터링
    uploadTask.on('state_changed',
        function(snapshot) {
            // 업로드 진행 상태를 표시할 수 있습니다.
        }, function(error) {
            // 업로드 중 에러 처리
            console.error(error);
        }, function() {
            
            // 업로드 성공 후 다운로드 URL 검색
            getDownloadURL(uploadTask.snapshot.ref).then(function(downloadURL) {
                console.log('File available at', downloadURL);           
                // Python 서버로 요청을 보내 유사 이미지 찾기
                fetch('http://127.0.0.1:5002/find-similar', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ imageUrl: downloadURL }), // 서버에 이미지 URL 전송
                })
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Network response was not ok');
                    }
                    return response.json(); // 응답을 JSON으로 변환
                })
                .then(data => {
                    const similarImage = data.similarImage; // 유사한 이미지의 URL 추출
                    // 다운로드 URL과 유사한 이미지 URL을 success.html 페이지로 전달
                    window.location.href = `success.html?imageUrl=${encodeURIComponent(downloadURL)}&similarImage=${encodeURIComponent(similarImage)}`;
                })
                .catch(error => {
                    console.error('Error:', error); // 오류 처리
                });
            });
        });
});

// URL에서 쿼리 파라미터를 파싱하는 함수
function getQueryParam(param) {
    var queryString = window.location.search.substring(1);
    var queryParams = queryString.split('&');
    for (var i = 0; i < queryParams.length; i++) {
        var pair = queryParams[i].split('=');
        if (decodeURIComponent(pair[0]) == param) {
            return decodeURIComponent(pair[1]);
        }
    }
    return null;
}

// 이미지 URL을 쿼리 파라미터에서 추출하고 img 태그의 src 속성에 설정
var imageUrl = getQueryParam('image');
if (imageUrl) {
    document.getElementById('uploadedImage').src = imageUrl;
    document.getElementById('image').src = imageUrl;
}