<script defer src="/__/firebase/10.12.0/firebase-app-compat.js"></script>;
<script defer src="/__/firebase/10.12.0/firebase-storage-compat.js"></script>;
<script defer src="/__/firebase/10.12.0/firebase-auth-compat.js"></script>;
<script defer src="/__/firebase/10.12.0/firebase-database-compat.js"></script>;
<script defer src="/__/firebase/10.12.0/firebase-firestore-compat.js"></script>;
<script defer src="/__/firebase/10.12.0/firebase-functions-compat.js"></script>;
<script defer src="/__/firebase/10.12.0/firebase-messaging-compat.js"></script>;
<script defer src="/__/firebase/10.12.0/firebase-analytics-compat.js"></script>;
<script defer src="/__/firebase/10.12.0/firebase-remote-config-compat.js"></script>;
<script defer src="/__/firebase/10.12.0/firebase-performance-compat.js"></script>;
<script defer src="/__/firebase/init.js?useEmulator=true"></script>;


document.addEventListener('DOMContentLoaded', function() {
  const loadEl = document.querySelector('#load');


  try {
    let app = firebase.app();
    let features = [
      'auth', 
      'database', 
      'firestore',
      'functions',
      'messaging', 
      'storage', 
      'analytics', 
      'remoteConfig',
      'performance',
    ].filter(feature => typeof app[feature] === 'function');
    loadEl.textContent = `Firebase SDK loaded with ${features.join(', ')}`;
  } catch (e) {
    console.error(e);
    loadEl.textContent = 'Error loading the Firebase SDK, check the console.';
  }
});



// import { initializeApp } from "firebase/app";
// import { getAnalytics } from "firebase/analytics";
// import { getStorage } from "firebase/storage";
import { getAuth } from "firebase/auth";
import { getFirestore } from "firebase/firestore";

// Import the functions you need from the SDKs you need
import { initializeApp } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-app.js";
import { getStorage, ref, uploadBytesResumable, getDownloadURL } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-storage.js";
import { getAnalytics } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-analytics.js";



// API Key
const firebaseConfig = {
apiKey: "AIzaSyAWoNvLsqbVzfDMpoubR1e7t6Ny4Xxk0JY",
authDomain: "webhotplace-1cce1.firebaseapp.com",
projectId: "webhotplace-1cce1",
storageBucket: "webhotplace-1cce1.appspot.com",
messagingSenderId: "331407528116",
appId: "1:331407528116:web:b4f6a50ec1587b9c4707e1",
measurementId: "G-N5Y9J96L8M"
};



// Initialize Firebase
const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);
const storage = getStorage(app);