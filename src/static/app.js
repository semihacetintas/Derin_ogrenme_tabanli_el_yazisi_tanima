/* ==========================================================================
   HTR Front-End Controller Script
   ========================================================================== */

let activeTab = 'upload';
let stream = null;

const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const webcamVideo = document.getElementById('webcamVideo');
const webcamOverlay = document.getElementById('webcamOverlay');
const startCamBtn = document.getElementById('startCamBtn');
const captureBtn = document.getElementById('captureBtn');
const previewPanel = document.getElementById('previewPanel');
const previewImg = document.getElementById('previewImg');
const loadingOverlay = document.getElementById('loadingOverlay');
const resultsPanel = document.getElementById('resultsPanel');
const correctedText = document.getElementById('correctedText');
const rawText = document.getElementById('rawText');
const breakdownTableBody = document.getElementById('breakdownTableBody');
const toast = document.getElementById('toast');
const toastMsg = document.getElementById('toastMsg');
const themeToggleBtn = document.getElementById('themeToggleBtn');

// 1. Tema Değiştirme Denetleyicisi
themeToggleBtn.addEventListener('click', () => {
    document.body.classList.toggle('light-theme');
    document.body.classList.toggle('dark-theme');
    
    const isLight = document.body.classList.contains('light-theme');
    localStorage.setItem('theme', isLight ? 'light' : 'dark');
});

// Kayıtlı temayı yükle
window.addEventListener('DOMContentLoaded', () => {
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'light') {
        document.body.classList.remove('dark-theme');
        document.body.classList.add('light-theme');
    }
});

// 2. Sekme Yönetimi (Tabs)
function switchTab(tabId) {
    if (activeTab === tabId) return;
    
    // Aktif sekme butonunu değiştir
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    event.currentTarget.classList.add('active');
    
    // Sekme içeriklerini gizle/göster
    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active-content'));
    document.getElementById(`${tabId}Tab`).classList.add('active-content');
    
    activeTab = tabId;
    
    // Eğer webcam sekmesinden çıkılıyorsa kamerayı kapat
    if (tabId !== 'webcam' && stream) {
        stopWebcam();
    }
}

// 3. Dosya Sürükle ve Bırak (Drag & Drop)
dropzone.addEventListener('click', () => fileInput.click());

dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
});

dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('dragover');
});

dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        showToast("Lütfen geçerli bir resim dosyası yükleyin!", "error");
        return;
    }

    // Önizleme paneline resmi yerleştir
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg.src = e.target.result;
        previewPanel.style.display = 'block';
        resultsPanel.style.display = 'none';
        
        // Sunucuya gönder
        uploadImage(file);
    };
    reader.readAsDataURL(file);
}

// 4. API İstekleri (Upload & Capture)
function uploadImage(file) {
    showLoading(true);
    
    const formData = new FormData();
    formData.append('file', file);
    
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => handleServerResponse(data))
    .catch(error => {
        showLoading(false);
        showToast("Görsel yüklenirken bir ağ hatası oluştu!", "error");
        console.error(error);
    });
}

// 5. Kamera (Webcam) Yönetimi
startCamBtn.addEventListener('click', startWebcam);

function startWebcam() {
    startCamBtn.disabled = true;
    webcamOverlay.style.display = 'flex';
    
    navigator.mediaDevices.getUserMedia({ 
        video: { 
            facingMode: 'environment', // Mobil cihazlar için arka kamera önceliği
            width: { ideal: 1280 },
            height: { ideal: 720 }
        } 
    })
    .then(mediaStream => {
        stream = mediaStream;
        webcamVideo.srcObject = mediaStream;
        webcamOverlay.style.display = 'none';
        captureBtn.disabled = false;
        showToast("Kamera başarıyla açıldı.");
    })
    .catch(err => {
        startCamBtn.disabled = false;
        webcamOverlay.style.display = 'none';
        showToast("Kameraya erişilemedi! İzin verdiğinizden emin olun.", "error");
        console.error(err);
    });
}

function stopWebcam() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
        webcamVideo.srcObject = null;
    }
    startCamBtn.disabled = false;
    captureBtn.disabled = true;
    webcamOverlay.style.display = 'flex';
}

captureBtn.addEventListener('click', () => {
    if (!stream) return;
    
    // Video karesini canvas'a çiz
    const canvas = document.createElement('canvas');
    canvas.width = webcamVideo.videoWidth;
    canvas.height = webcamVideo.videoHeight;
    
    const ctx = canvas.getContext('2d');
    ctx.drawImage(webcamVideo, 0, 0, canvas.width, canvas.height);
    
    const dataUrl = canvas.toDataURL('image/png');
    
    // Önizleme paneline yerleştir
    previewImg.src = dataUrl;
    previewPanel.style.display = 'block';
    resultsPanel.style.display = 'none';
    
    // Kamerayı durdur
    stopWebcam();
    
    // Sunucuya gönder
    showLoading(true);
    fetch('/capture', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: dataUrl })
    })
    .then(response => response.json())
    .then(data => handleServerResponse(data))
    .catch(error => {
        showLoading(false);
        showToast("Görüntü gönderilirken ağ hatası oluştu!", "error");
        console.error(error);
    });
});

// 6. Yanıt İşleme & Dinamik DOM Oluşturma
function handleServerResponse(data) {
    showLoading(false);
    
    if (!data.success) {
        showToast(data.message, "error");
        return;
    }
    
    // Sonuçları ekrana bas
    correctedText.textContent = data.corrected_text || "(Boş)";
    rawText.textContent = data.raw_text || "(Boş)";
    
    // Tabloyu temizle ve doldur
    breakdownTableBody.innerHTML = '';
    
    data.words.forEach((word, index) => {
        const tr = document.createElement('tr');
        
        const isMatch = word.raw === word.corrected;
        const statusText = isMatch ? 'Eşleşti' : 'Düzeltildi';
        const statusClass = isMatch ? 'status-match' : 'status-corrected';
        
        tr.innerHTML = `
            <td><strong>${index + 1}</strong></td>
            <td><code style="font-size: 1rem; color: #F87171;">${word.raw || '-'}</code></td>
            <td><strong style="color: #10B981; font-size: 1rem;">${word.corrected || '-'}</strong></td>
            <td><span class="status-badge ${statusClass}">${statusText}</span></td>
        `;
        breakdownTableBody.appendChild(tr);
    });
    
    resultsPanel.style.display = 'block';
    showToast("El yazısı başarıyla çözümlendi!");
    
    // Sonuçlar paneline pürüzsüz kaydırma yap
    resultsPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// 7. Yardımcı Fonksiyonlar (Reset, Toast, Copy)
function showLoading(isLoading) {
    loadingOverlay.style.display = isLoading ? 'flex' : 'none';
}

function resetApp() {
    previewPanel.style.display = 'none';
    resultsPanel.style.display = 'none';
    previewImg.src = '';
    fileInput.value = '';
    
    if (activeTab === 'webcam') {
        startWebcam();
    }
}

function copyText(elementId) {
    const text = document.getElementById(elementId).textContent;
    if (!text || text === "(Boş)") return;
    
    navigator.clipboard.writeText(text)
    .then(() => {
        showToast("Panoya kopyalandı! 📋");
    })
    .catch(err => {
        showToast("Kopyalanamadı!", "error");
    });
}

function showToast(message, type = "success") {
    toastMsg.textContent = message;
    toast.className = 'toast'; // Reset
    
    if (type === "error") {
        toast.style.background = "#EF4444";
        toast.style.boxShadow = "0 10px 30px rgba(239, 68, 68, 0.3)";
    } else {
        toast.style.background = "#10B981";
        toast.style.boxShadow = "0 10px 30px rgba(16, 185, 129, 0.3)";
    }
    
    toast.classList.add('show');
    
    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}
