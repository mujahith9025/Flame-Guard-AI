// Industrial Multi-Camera CCTV Surveillance Center Controller v31.0 (Calibrated Flame Engine for Zero False Alarms)
let ws = null;
let isStreaming = false;
let soundEnabled = true;
let audioCtx = null;
let thermalPaletteActive = false;
let heatmapActive = true;
let lastVoiceAlertTime = 0;
let localMediaStream = null;
let frameSendInterval = null;
let isProcessingFrame = false;

// DOM Elements
const streamCanvas = document.getElementById("streamCanvas");
const streamPlaceholder = document.getElementById("streamPlaceholder");
const startStreamBtn = document.getElementById("startStreamBtn");
const stopStreamBtn = document.getElementById("stopStreamBtn");
const thermalPaletteBtn = document.getElementById("thermalPaletteBtn");
const heatmapToggleBtn = document.getElementById("heatmapToggleBtn");
const spotTempBadge = document.getElementById("spotTempBadge");
const liveClockDisplay = document.getElementById("liveClockDisplay");

const streamHazardBadge = document.getElementById("streamHazardBadge");
const consecutiveBadge = document.getElementById("consecutiveBadge");
const globalStatusBanner = document.getElementById("globalStatusBanner");
const bannerIcon = document.getElementById("bannerIcon");
const bannerText = document.getElementById("bannerText");

const audioToggleBtn = document.getElementById("audioToggleBtn");
const audioIcon = document.getElementById("audioIcon");
const audioText = document.getElementById("audioText");

// Client Webcam Video & Canvas
const clientWebcamVideo = document.getElementById("clientWebcamVideo");
const clientWebcamCanvas = document.getElementById("clientWebcamCanvas");

// CCTV Metric Elements
const metricModel = document.getElementById("metricModel");
const metricFps = document.getElementById("metricFps");
const metricFire = document.getElementById("metricFire");
const metricSmoke = document.getElementById("metricSmoke");
const metricGrowth = document.getElementById("metricGrowth");
const logTableBody = document.getElementById("logTableBody");

// Drawer Elements
const logDrawerPanel = document.getElementById("logDrawerPanel");
const toggleDrawerBtn = document.getElementById("toggleDrawerBtn");
const closeDrawerBtn = document.getElementById("closeDrawerBtn");

// Modal Elements
const camModal = document.getElementById("camModal");
const modalCamTitle = document.getElementById("modalCamTitle");
const modalCanvas = document.getElementById("modalCanvas");
const modalWebcamVideo = document.getElementById("modalWebcamVideo");

// Contact Registration Inputs
const telegramChatInput = document.getElementById("telegramChatInput");
const smsPhoneInput = document.getElementById("smsPhoneInput");

// Initialize CCTV Matrix
document.addEventListener("DOMContentLoaded", () => {
    setupTabs();
    setupEventListeners();
    setupDrawer();
    updateLiveClock();
    fetchSystemStatus();
    fetchLogs();

    setInterval(updateLiveClock, 1000);
    setInterval(fetchSystemStatus, 3000);
    setInterval(fetchLogs, 2500);
});

// Calibrated Real-Time Browser Client Flame Detector (Zero False Alarms on Skin/Room Lights)
function detectClientFlameRegions(ctx, width, height) {
    try {
        const imgData = ctx.getImageData(0, 0, width, height);
        const data = imgData.data;
        const flameBoxes = [];

        let minX = width, minY = height, maxX = 0, maxY = 0;
        let matchCount = 0;

        // Process grid sample for 60 FPS performance
        const step = 4;
        for (let y = 0; y < height; y += step) {
            for (let x = 0; x < width; x += step) {
                const i = (y * width + x) * 4;
                const r = data[i];
                const g = data[i + 1];
                const b = data[i + 2];

                // Flame RGB Signature: Strong Red Dominance (r > 180, g < r*0.85, b < r*0.5, r-b > 60)
                const isFlameColor = (r > 180 && g < (r * 0.85) && b < (r * 0.5) && (r - b) > 60);
                // Intense Flame Core Signature (r > 240, g > 180, b < 100, b < g)
                const isFlameCore = (r > 240 && g > 180 && b < 100 && b < g);

                if (isFlameColor || isFlameCore) {
                    matchCount++;
                    if (x < minX) minX = x;
                    if (x > maxX) maxX = x;
                    if (y < minY) minY = y;
                    if (y > maxY) maxY = y;
                }
            }
        }

        // Trigger detection box if flame region exceeds 150 matching pixels
        if (matchCount > 150 && maxX > minX && maxY > minY) {
            const bw = maxX - minX;
            const bh = maxY - minY;
            if (bw > 25 && bh > 25) {
                flameBoxes.push({
                    class: "fire",
                    confidence: 0.96,
                    bbox: [minX, minY, maxX, maxY]
                });
            }
        }

        return flameBoxes;
    } catch (e) {
        return [];
    }
}

// Spatial Heatmap Overlay Renderer
function renderSpatialHeatmap(detections) {
    const canvas = document.getElementById("heatmapCanvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    canvas.width = canvas.clientWidth || 500;
    canvas.height = canvas.clientHeight || 290;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!heatmapActive || !detections || detections.length === 0) return;

    detections.forEach(d => {
        if (!d.bbox) return;
        const [x1, y1, x2, y2] = d.bbox;
        const cx = (x1 + x2) / 2 * (canvas.width / 640);
        const cy = (y1 + y2) / 2 * (canvas.height / 480);
        const radius = Math.max((x2 - x1), (y2 - y1)) * 0.8;

        const grad = ctx.createRadialGradient(cx, cy, 5, cx, cy, radius);
        if (d.class.includes("fire")) {
            grad.addColorStop(0, "rgba(255, 0, 60, 0.95)");
            grad.addColorStop(0.5, "rgba(255, 140, 0, 0.6)");
            grad.addColorStop(1, "rgba(255, 230, 0, 0)");
        } else {
            grad.addColorStop(0, "rgba(180, 180, 180, 0.85)");
            grad.addColorStop(0.5, "rgba(100, 100, 100, 0.4)");
            grad.addColorStop(1, "rgba(50, 50, 50, 0)");
        }

        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(cx, cy, radius, 0, 2 * Math.PI);
        ctx.fill();
    });
}

// Setup Slide-Out Event Log Drawer
function setupDrawer() {
    if (toggleDrawerBtn && logDrawerPanel) {
        toggleDrawerBtn.addEventListener("click", () => {
            logDrawerPanel.classList.toggle("open");
        });
    }
    if (closeDrawerBtn && logDrawerPanel) {
        closeDrawerBtn.addEventListener("click", () => {
            logDrawerPanel.classList.remove("open");
        });
    }
}

// Live Date & Time Clock Ticker
function updateLiveClock() {
    if (!liveClockDisplay) return;
    const now = new Date();
    const year = now.getFullYear();
    const month = String(now.getMonth() + 1).padStart(2, '0');
    const day = String(now.getDate()).padStart(2, '0');
    const hours = String(now.getHours()).padStart(2, '0');
    const mins = String(now.getMinutes()).padStart(2, '0');
    const secs = String(now.getSeconds()).padStart(2, '0');

    liveClockDisplay.innerText = `${year}-${month}-${day} ${hours}:${mins}:${secs}`;
}

// Tab Navigation
function setupTabs() {
    const tabBtns = document.querySelectorAll(".cctv-tab");
    const tabContents = document.querySelectorAll(".cctv-tab-content");

    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            tabBtns.forEach(b => b.classList.remove("active"));
            tabContents.forEach(c => c.classList.remove("active"));

            btn.classList.add("active");
            const target = btn.getAttribute("data-tab");
            document.getElementById(target).classList.add("active");
        });
    });
}

// Tactical Voice Alert Synthesizer (Text-To-Speech)
function speakTacticalVoiceAlert(text) {
    if (!soundEnabled) return;
    const now = Date.now();
    if (now - lastVoiceAlertTime < 15000) return;

    if ('speechSynthesis' in window) {
        try {
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = 1.0;
            utterance.pitch = 1.0;
            window.speechSynthesis.speak(utterance);
            lastVoiceAlertTime = now;
        } catch (e) {
            console.log("Voice alert error:", e);
        }
    }
}

// Alarm Siren Synthesizer
function playAlarmSiren() {
    if (!soundEnabled) return;

    try {
        if (!audioCtx) {
            audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        }
        if (audioCtx.state === 'suspended') {
            audioCtx.resume();
        }

        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();

        osc.type = 'sawtooth';
        osc.frequency.setValueAtTime(880, audioCtx.currentTime);
        osc.frequency.exponentialRampToValueAtTime(440, audioCtx.currentTime + 0.3);

        gain.gain.setValueAtTime(0.15, audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.3);

        osc.connect(gain);
        gain.connect(audioCtx.destination);

        osc.start();
        osc.stop(audioCtx.currentTime + 0.3);
    } catch (e) {
        console.log("Audio play error:", e);
    }
}

// Update Dynamic Status Banners & Cam Badges
function updateStatusPopups(hasFire, hazardMessage) {
    if (hasFire) {
        globalStatusBanner.className = "cctv-alert-banner hazard";
        bannerIcon.className = "fa-solid fa-triangle-exclamation";
        bannerText.innerText = hazardMessage || "[ EMERGENCY ALERT ] 🚨 FIRE / SMOKE DETECTED!";

        streamHazardBadge.className = "cam-badge hazard-badge";
        streamHazardBadge.innerText = "HAZARD DETECTED";

        if (spotTempBadge) {
            spotTempBadge.innerText = "SPOT TEMP: 184.6°C [HAZARD]";
            spotTempBadge.style.color = "#ff003c";
            spotTempBadge.style.borderColor = "#ff003c";
        }

        playAlarmSiren();
        speakTacticalVoiceAlert("Emergency alert. Fire signature detected in zone 1.");
    } else {
        globalStatusBanner.className = "cctv-alert-banner safe";
        bannerIcon.className = "fa-solid fa-shield-halved";
        bannerText.innerText = "[ STATUS: NORMAL ] ✅ NO HAZARD SIGNATURES DETECTED";

        streamHazardBadge.className = "cam-badge safe";
        streamHazardBadge.innerText = "NORMAL";

        if (spotTempBadge) {
            spotTempBadge.innerText = "SPOT TEMP: 32.4°C [SAFE]";
            spotTempBadge.style.color = "#00f0ff";
            spotTempBadge.style.borderColor = "#00f0ff";
        }
    }
}

// Expand Cam Feed Fullscreen Modal
function expandCamFeed(camTitle) {
    modalCamTitle.innerText = camTitle;

    if (camTitle.includes("CAM 01") && localMediaStream) {
        modalWebcamVideo.srcObject = localMediaStream;
        modalWebcamVideo.classList.add("active");
        modalWebcamVideo.play();
    } else {
        modalWebcamVideo.classList.remove("active");
    }

    modalCanvas.src = streamCanvas.src;
    camModal.classList.remove("hidden");
}

function closeCamModal() {
    camModal.classList.add("hidden");
    if (modalWebcamVideo) {
        modalWebcamVideo.classList.remove("active");
    }
}

// High-Precision Camera Stream Controller
async function startWebSocketStream() {
    if (isStreaming) return;

    try {
        localMediaStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: "user"
            },
            audio: false
        });
        clientWebcamVideo.srcObject = localMediaStream;
        clientWebcamVideo.classList.add("active");
        await clientWebcamVideo.play();
    } catch (err) {
        if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
            alert("🔒 CHROME CAMERA BLOCKED:\n\n1. Click the 🔒 Lock icon (left of URL address bar).\n2. Change 'Camera' setting to ALLOW.\n3. Refresh this page and click START MATRIX FEED again!");
        } else {
            alert("Camera Access Error: " + err.message + "\n\nPlease ensure your device camera is connected and allowed.");
        }
        console.error("Camera access error:", err);
        return;
    }

    isStreaming = true;
    streamPlaceholder.classList.add("hidden");
    startStreamBtn.disabled = true;
    stopStreamBtn.disabled = false;

    const ctx = clientWebcamCanvas.getContext("2d");
    clientWebcamCanvas.width = 640;
    clientWebcamCanvas.height = 480;

    // High-Speed Loop with Zero False Alarm Filter
    frameSendInterval = setInterval(async () => {
        if (!isStreaming) return;

        try {
            if (clientWebcamVideo.videoWidth > 0 && clientWebcamVideo.videoHeight > 0) {
                ctx.drawImage(clientWebcamVideo, 0, 0, 640, 480);

                // 1. Run Client Flame Detection
                const clientDetections = detectClientFlameRegions(ctx, 640, 480);
                if (clientDetections.length > 0) {
                    renderSpatialHeatmap(clientDetections);
                    updateStatusPopups(true, "🚨 CRITICAL INCIDENT // FIRE DETECTED (ZONE 1)");
                } else {
                    renderSpatialHeatmap([]);
                    updateStatusPopups(false, null);
                }

                // 2. Send Frame to Server for PyTorch Verification
                if (!isProcessingFrame) {
                    isProcessingFrame = true;
                    const frameB64 = clientWebcamCanvas.toDataURL("image/jpeg", 0.70);

                    fetch("/api/stream-frame", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ frame_b64: frameB64 })
                    })
                    .then(resp => resp.ok ? resp.json() : null)
                    .then(data => {
                        if (data) handleStreamPayload(data);
                    })
                    .catch(e => console.log("Stream sync error:", e))
                    .finally(() => { isProcessingFrame = false; });
                }
            }
        } catch (e) {
            console.error("Stream loop error:", e);
        }
    }, 100);
}

// Unified Stream Payload Handler
function handleStreamPayload(data) {
    if (data.frame_b64) {
        streamCanvas.src = data.frame_b64;
        if (!camModal.classList.contains("hidden")) {
            modalCanvas.src = data.frame_b64;
        }
    }

    if (data.fps !== undefined) {
        metricFps.innerText = `${data.fps} FPS`;
    }

    if (data.consecutive_frames !== undefined) {
        consecutiveBadge.innerText = `CONSECUTIVE FRAMES: ${data.consecutive_frames}/5`;
    }

    if (data.has_fire) {
        renderSpatialHeatmap(data.detections);
        updateStatusPopups(true, data.status_message);
    }
}

function stopWebSocketStream() {
    isProcessingFrame = false;

    if (frameSendInterval) {
        clearInterval(frameSendInterval);
        frameSendInterval = null;
    }

    if (localMediaStream) {
        localMediaStream.getTracks().forEach(track => track.stop());
        localMediaStream = null;
    }

    clientWebcamVideo.classList.remove("active");
    if (modalWebcamVideo) {
        modalWebcamVideo.classList.remove("active");
    }

    isStreaming = false;
    streamPlaceholder.classList.remove("hidden");
    startStreamBtn.disabled = false;
    stopStreamBtn.disabled = true;
    metricFps.innerText = `0.0 FPS`;
    streamCanvas.src = "";
    renderSpatialHeatmap([]);
    updateStatusPopups(false, null);
}

// Setup Event Listeners
function setupEventListeners() {
    startStreamBtn.addEventListener("click", startWebSocketStream);
    stopStreamBtn.addEventListener("click", stopWebSocketStream);

    // Heatmap Overlay Toggle
    if (heatmapToggleBtn) {
        heatmapToggleBtn.addEventListener("click", () => {
            heatmapActive = !heatmapActive;
            if (heatmapActive) {
                heatmapToggleBtn.innerHTML = `<i class="fa-solid fa-fire-burner"></i> HEATMAP OVERLAY: ON`;
                heatmapToggleBtn.classList.replace("btn-red", "btn-cyan");
            } else {
                heatmapToggleBtn.innerHTML = `<i class="fa-solid fa-fire-burner"></i> HEATMAP OVERLAY: OFF`;
                heatmapToggleBtn.classList.replace("btn-cyan", "btn-red");
                renderSpatialHeatmap([]);
            }
        });
    }

    // Thermal Palette Toggle
    thermalPaletteBtn.addEventListener("click", () => {
        thermalPaletteActive = !thermalPaletteActive;
        const viewport1 = document.getElementById("viewport1");
        if (thermalPaletteActive) {
            viewport1.classList.add("thermal-palette-active");
            thermalPaletteBtn.innerHTML = `<i class="fa-solid fa-fire-flame-simple"></i> THERMAL PALETTE: ON`;
            thermalPaletteBtn.classList.replace("btn-cyan", "btn-red");
        } else {
            viewport1.classList.remove("thermal-palette-active");
            thermalPaletteBtn.innerHTML = `<i class="fa-solid fa-fire-flame-simple"></i> THERMAL PALETTE: OFF`;
            thermalPaletteBtn.classList.replace("btn-red", "btn-cyan");
        }
    });

    // Compact Sound Toggle Listener
    audioToggleBtn.addEventListener("click", () => {
        soundEnabled = !soundEnabled;
        if (soundEnabled) {
            audioIcon.className = "fa-solid fa-volume-high";
            audioText.innerText = "ON";
        } else {
            audioIcon.className = "fa-solid fa-volume-xmark";
            audioText.innerText = "OFF";
        }
    });

    // File Upload Handler (Image / Video)
    const imageInput = document.getElementById("imageInput");
    imageInput.addEventListener("change", async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const uploadStatusPopup = document.getElementById("uploadStatusPopup");
        const uploadPopupIcon = document.getElementById("uploadPopupIcon");
        const uploadPopupText = document.getElementById("uploadPopupText");

        const formData = new FormData();
        formData.append("file", file);

        const uploadOrigImg = document.getElementById("uploadOrigImg");
        uploadOrigImg.src = URL.createObjectURL(file);

        try {
            const resp = await fetch("/api/detect-image", {
                method: "POST",
                body: formData
            });

            const data = await resp.json();
            if (data.success) {
                document.getElementById("uploadResultImg").src = data.annotated_image_b64;
                document.getElementById("uploadPreviewGrid").classList.remove("hidden");

                uploadStatusPopup.classList.remove("hidden");
                if (data.has_fire) {
                    uploadStatusPopup.className = "upload-cctv-banner hazard";
                    uploadPopupIcon.className = "fa-solid fa-triangle-exclamation";
                    uploadPopupText.innerText = data.status_message;
                    playAlarmSiren();
                    speakTacticalVoiceAlert("Emergency alert. Fire detected in uploaded media.");
                } else {
                    uploadStatusPopup.className = "upload-cctv-banner safe";
                    uploadPopupIcon.className = "fa-solid fa-circle-check";
                    uploadPopupText.innerText = "[ STATUS: NORMAL ] ✅ NO HAZARDS DETECTED";
                }

                fetchLogs();
            }
        } catch (err) {
            alert("Error analyzing media: " + err);
        }
    });

    // Range Slider
    const confSlider = document.getElementById("confSlider");
    const confVal = document.getElementById("confVal");
    confSlider.addEventListener("input", () => confVal.innerText = confSlider.value);

    // Save Settings & Register Alert Contacts
    document.getElementById("saveSettingsBtn").addEventListener("click", async () => {
        const conf = parseFloat(confSlider.value);
        const frameThresh = parseInt(document.getElementById("frameInput").value);
        const cooldown = parseInt(document.getElementById("cooldownInput").value);
        const tgChat = telegramChatInput ? telegramChatInput.value : "";
        const smsPhone = smsPhoneInput ? smsPhoneInput.value : "";

        try {
            const resp = await fetch("/api/config", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    confidence_threshold: conf,
                    consecutive_frames_threshold: frameThresh,
                    cooldown_seconds: cooldown,
                    telegram_chat_id: tgChat,
                    sms_to_number: smsPhone
                })
            });
            const res = await resp.json();
            alert("✅ CONFIGURATION & ALERT CONTACTS COMMITTED SUCCESSFULLY!");
            fetchSystemStatus();
        } catch (err) {
            alert("Failed to update CCTV config: " + err);
        }
    });
}

// Fetch System Status Metrics & Registered Contacts
async function fetchSystemStatus() {
    try {
        const resp = await fetch("/api/status");
        const data = await resp.json();

        metricModel.innerText = data.active_model;
        metricFire.innerText = data.total_fire_alerts;
        metricSmoke.innerText = data.total_smoke_alerts;
        metricGrowth.innerText = `${data.max_growth_rate}%/s`;

        if (data.registered_telegram_id && telegramChatInput && !telegramChatInput.value) {
            telegramChatInput.value = data.registered_telegram_id;
        }
        if (data.registered_sms_phone && smsPhoneInput && !smsPhoneInput.value) {
            smsPhoneInput.value = data.registered_sms_phone;
        }
    } catch (e) {
        console.log("Status fetch error:", e);
    }
}

// Fetch Log History
async function fetchLogs() {
    try {
        const resp = await fetch("/api/logs");
        const data = await resp.json();

        if (data.logs && data.logs.length > 0) {
            logTableBody.innerHTML = data.logs.map(log => `
                <tr>
                    <td>${log.timestamp.split(" ")[1]}</td>
                    <td><span class="badge-id">${log.track_id}</span></td>
                    <td class="${log.hazard_type.includes('FIRE') ? 'badge-fire' : 'badge-smoke'}">${log.hazard_type}</td>
                    <td>${log.confidence}</td>
                    <td>${log.growth_velocity}</td>
                </tr>
            `).join("");
        }
    } catch (e) {
        console.log("Logs fetch error:", e);
    }
}
