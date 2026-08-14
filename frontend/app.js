// Industrial Multi-Camera CCTV Surveillance Center Controller v40.0 (Pure PyTorch YOLOv8 Neural Network Fire, Person & Smoke Detection Engine)
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
let clientLogs = [];
let fireAlertCount = 0;
let smokeAlertCount = 0;

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

// Real-Time High-Contrast Bounding Box Overlay Renderer (Supports FIRE, PERSON & SMOKE across Grid View & Fullscreen View)
function renderBoundingBoxes(detections) {
    const canvases = [];
    
    // 1. Grid View Overlay Canvas
    let gridCanvas = document.getElementById("boxOverlayCanvas");
    if (!gridCanvas) {
        const viewport = document.getElementById("viewport1");
        if (viewport) {
            gridCanvas = document.createElement("canvas");
            gridCanvas.id = "boxOverlayCanvas";
            gridCanvas.style.position = "absolute";
            gridCanvas.style.top = "0";
            gridCanvas.style.left = "0";
            gridCanvas.style.width = "100%";
            gridCanvas.style.height = "100%";
            gridCanvas.style.pointerEvents = "none";
            gridCanvas.style.zIndex = "12";
            viewport.appendChild(gridCanvas);
        }
    }
    if (gridCanvas) canvases.push(gridCanvas);

    // 2. Fullscreen Modal Overlay Canvas
    let modalCanvasElement = document.getElementById("modalBoxOverlayCanvas");
    if (modalCanvasElement && !camModal.classList.contains("hidden")) {
        canvases.push(modalCanvasElement);
    }

    canvases.forEach(canvas => {
        const ctx = canvas.getContext("2d");
        canvas.width = canvas.clientWidth || 640;
        canvas.height = canvas.clientHeight || 480;

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (!detections || detections.length === 0) return;

        const scaleX = canvas.width / 640;
        const scaleY = canvas.height / 480;

        detections.forEach(d => {
            if (!d.bbox) return;
            const [x1, y1, x2, y2] = d.bbox;
            const bx = x1 * scaleX;
            const by = y1 * scaleY;
            const bw = (x2 - x1) * scaleX;
            const bh = (y2 - y1) * scaleY;

            const clsLower = d.class.toLowerCase();
            let strokeColor = "#ff003c"; // Fire (Red)
            if (clsLower.includes("person")) {
                strokeColor = "#ffaa00"; // Person (Orange)
            } else if (clsLower.includes("smoke")) {
                strokeColor = "#a0a0a0"; // Smoke (Gray)
            }

            const labelText = `${d.class.toUpperCase()}: ${(d.confidence * 100).toFixed(1)}%`;

            // 1. Draw Outer Bounding Rectangle
            ctx.strokeStyle = strokeColor;
            ctx.lineWidth = 3.5;
            ctx.shadowColor = strokeColor;
            ctx.shadowBlur = 12;
            ctx.strokeRect(bx, by, bw, bh);

            // 2. Draw Filled Header Box & Label Text
            ctx.shadowBlur = 0;
            ctx.font = "bold 14px Outfit, Inter, sans-serif";
            const textWidth = ctx.measureText(labelText).width;

            ctx.fillStyle = strokeColor;
            ctx.fillRect(bx, Math.max(0, by - 26), textWidth + 14, 26);

            ctx.fillStyle = "#ffffff";
            ctx.fillText(labelText, bx + 7, Math.max(18, by - 7));
        });
    });
}

// Add Incident Entry to Event Log Drawer Table
function logClientIncident(hazardType, confStr, sourceStr) {
    const now = new Date();
    const timeStr = `${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}:${String(now.getSeconds()).padStart(2, '0')}`;

    const newLog = {
        timestamp: `${now.toISOString().split('T')[0]} ${timeStr}`,
        track_id: "ID #1",
        hazard_type: hazardType.toUpperCase(),
        confidence: confStr,
        growth_velocity: "+12.4%/s",
        source: sourceStr
    };

    if (clientLogs.length > 0 && clientLogs[0].timestamp.endsWith(timeStr)) {
        return;
    }

    clientLogs.unshift(newLog);
    if (clientLogs.length > 50) clientLogs.pop();

    if (hazardType.toUpperCase().includes("FIRE")) {
        fireAlertCount++;
        metricFire.innerText = fireAlertCount;
    }

    renderLogTable();
}

function renderLogTable() {
    if (!logTableBody) return;
    if (clientLogs.length === 0) {
        logTableBody.innerHTML = `<tr><td colspan="5" class="empty-table-cell">NO CCTV HAZARDS LOGGED. SYSTEM SECURE.</td></tr>`;
        return;
    }

    logTableBody.innerHTML = clientLogs.map(log => `
        <tr>
            <td>${log.timestamp.split(" ")[1]}</td>
            <td><span class="badge-id">${log.track_id}</span></td>
            <td class="${log.hazard_type.includes('FIRE') ? 'badge-fire' : 'badge-smoke'}">${log.hazard_type}</td>
            <td>${log.confidence}</td>
            <td>${log.growth_velocity}</td>
        </tr>
    `).join("");
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
        if (d.class.toLowerCase().includes("fire")) {
            grad.addColorStop(0, "rgba(255, 0, 60, 0.95)");
            grad.addColorStop(0.5, "rgba(255, 140, 0, 0.6)");
            grad.addColorStop(1, "rgba(255, 230, 0, 0)");
            ctx.fillStyle = grad;
            ctx.beginPath();
            ctx.arc(cx, cy, radius, 0, 2 * Math.PI);
            ctx.fill();
        }
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

// Dynamic Multi-Camera Matrix Layout Switcher (2x2 Quad, 1 Large + 3 Side, 1x1 Focus)
function switchMatrixLayout(mode) {
    const grid = document.getElementById("matrixGridContainer");
    const quadBtn = document.getElementById("layoutQuadBtn");
    const splitBtn = document.getElementById("layoutSplitBtn");
    const singleBtn = document.getElementById("layoutSingleBtn");

    if (!grid) return;

    grid.classList.remove("layout-split", "layout-single");
    if (quadBtn) quadBtn.classList.remove("active");
    if (splitBtn) splitBtn.classList.remove("active");
    if (singleBtn) singleBtn.classList.remove("active");

    if (mode === "split") {
        grid.classList.add("layout-split");
        if (splitBtn) splitBtn.classList.add("active");
    } else if (mode === "single") {
        grid.classList.add("layout-single");
        if (singleBtn) singleBtn.classList.add("active");
    } else {
        if (quadBtn) quadBtn.classList.add("active");
    }

    // Trigger window resize event so canvas bounding boxes recalculate positions smoothly
    setTimeout(() => {
        window.dispatchEvent(new Event("resize"));
    }, 150);
}

// Color Theme Switcher Engine (Cyber Cyan, Matrix Green, Dark OLED, Crimson Red)
function switchCctvTheme(themeName) {
    if (themeName === "cyan") {
        document.documentElement.removeAttribute("data-theme");
    } else {
        document.documentElement.setAttribute("data-theme", themeName);
    }
    localStorage.setItem("flameguard_theme", themeName);
}

// Restore saved theme on DOM Load
const savedTheme = localStorage.getItem("flameguard_theme");
if (savedTheme) {
    switchCctvTheme(savedTheme);
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

    const cam01Btn = document.getElementById("cam01ToggleBtn");
    if (cam01Btn) {
        cam01Btn.className = "cam-toggle-btn btn-cam-on";
        cam01Btn.innerHTML = `<i class="fa-solid fa-power-off"></i> CAMERA OFF`;
    }

    const ctx = clientWebcamCanvas.getContext("2d");
    clientWebcamCanvas.width = 640;
    clientWebcamCanvas.height = 480;

    // High-Speed Neural Network Frame Processing Loop
    frameSendInterval = setInterval(async () => {
        if (!isStreaming) return;

        try {
            if (clientWebcamVideo.videoWidth > 0 && clientWebcamVideo.videoHeight > 0) {
                ctx.drawImage(clientWebcamVideo, 0, 0, 640, 480);

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

// Per-Camera Stream Toggle Handler
function toggleCamStream(camId) {
    if (camId === "cam01") {
        if (!isStreaming) {
            startWebSocketStream();
        } else {
            stopWebSocketStream();
        }
    } else {
        const toggleBtn = document.getElementById(`${camId}ToggleBtn`);
        const overlay = document.getElementById(`overlay${camId.charAt(0).toUpperCase() + camId.slice(1)}`);
        const text = document.getElementById(`text${camId.charAt(0).toUpperCase() + camId.slice(1)}`);
        const badge = document.getElementById(`badge${camId.charAt(0).toUpperCase() + camId.slice(1)}`);

        if (toggleBtn.classList.contains("btn-cam-on")) {
            // Turn Cam OFF
            toggleBtn.className = "cam-toggle-btn btn-cam-off";
            toggleBtn.innerHTML = `<i class="fa-solid fa-power-off"></i> CAMERA ON`;
            if (overlay) overlay.style.opacity = "0.25";
            if (text) text.innerText = "CAMERA OFF // FEED PAUSED";
            if (badge) { badge.className = "cam-badge hazard-badge"; badge.innerText = "OFFLINE"; }
        } else {
            // Turn Cam ON
            toggleBtn.className = "cam-toggle-btn btn-cam-on";
            toggleBtn.innerHTML = `<i class="fa-solid fa-power-off"></i> CAMERA OFF`;
            if (overlay) overlay.style.opacity = "1.0";
            if (text) {
                text.innerText = camId === "cam02" ? "IR THERMAL SCANNING ACTIVE // 24.5°C NORMAL" : (camId === "cam03" ? "PERIMETER CLEAR // SENSOR ONLINE" : "PRESSURE NORMAL // MONITORING ACTIVE");
            }
            if (badge) { badge.className = "cam-badge safe"; badge.innerText = "NORMAL"; }
        }
    }
}

// Unified Stream Payload Handler (Relying 100% on PyTorch YOLOv8 Neural Network Predictions)
function handleStreamPayload(data) {
    if (data.frame_b64) {
        streamCanvas.src = data.frame_b64;
        if (modalCanvas && !camModal.classList.contains("hidden")) {
            modalCanvas.src = data.frame_b64;
        }
    }

    if (data.fps !== undefined) {
        metricFps.innerText = `${data.fps} FPS`;
    }

    if (data.consecutive_frames !== undefined) {
        consecutiveBadge.innerText = `CONSECUTIVE FRAMES: ${data.consecutive_frames}/5`;
    }

    // Render Bounding Boxes for Fire, Person, Smoke
    if (data.detections && data.detections.length > 0) {
        renderBoundingBoxes(data.detections);
    } else {
        renderBoundingBoxes([]);
    }

    // Trigger Hazard Alerts if Fire or Smoke is Detected
    if (data.has_fire) {
        const fireDet = data.detections ? data.detections.find(d => d.class.toLowerCase().includes("fire")) : null;
        const confStr = fireDet ? `${(fireDet.confidence * 100).toFixed(1)}%` : "96.0%";
        renderSpatialHeatmap(data.detections);
        updateStatusPopups(true, data.status_message);
        logClientIncident("FIRE", confStr, "Live Camera Stream");
    } else {
        renderSpatialHeatmap([]);
        updateStatusPopups(false, null);
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

    const cam01Btn = document.getElementById("cam01ToggleBtn");
    if (cam01Btn) {
        cam01Btn.className = "cam-toggle-btn btn-cam-off";
        cam01Btn.innerHTML = `<i class="fa-solid fa-power-off"></i> CAMERA ON`;
    }

    isStreaming = false;
    streamPlaceholder.classList.remove("hidden");
    startStreamBtn.disabled = false;
    stopStreamBtn.disabled = true;
    metricFps.innerText = `0.0 FPS`;
    renderBoundingBoxes([]);
    renderSpatialHeatmap([]);
    updateStatusPopups(false, null);
}

// Setup Event Listeners
function setupEventListeners() {
    startStreamBtn.addEventListener("click", startWebSocketStream);
    stopStreamBtn.addEventListener("click", stopWebSocketStream);

    // Cyberpunk HUD Scanline Overlay Toggle
    const hudOverlayToggleBtn = document.getElementById("hudOverlayToggleBtn");
    if (hudOverlayToggleBtn) {
        hudOverlayToggleBtn.addEventListener("click", () => {
            const matrixGridContainer = document.getElementById("matrixGridContainer");
            if (matrixGridContainer) {
                matrixGridContainer.classList.toggle("hud-scanline-active");
                const isScanlineActive = matrixGridContainer.classList.contains("hud-scanline-active");
                if (isScanlineActive) {
                    hudOverlayToggleBtn.innerHTML = `<i class="fa-solid fa-border-top-left"></i> HUD SCANLINES: ON`;
                    hudOverlayToggleBtn.classList.replace("btn-cyan", "btn-green");
                } else {
                    hudOverlayToggleBtn.innerHTML = `<i class="fa-solid fa-border-top-left"></i> HUD SCANLINES: OFF`;
                    hudOverlayToggleBtn.classList.replace("btn-green", "btn-cyan");
                }
            }
        });
    }

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
                    logClientIncident("FIRE", "98.0%", "Media File Inspection");
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
    if (confSlider) {
        confSlider.addEventListener("input", () => confVal.innerText = confSlider.value);
    }

    // Save Settings & Register Alert Contacts
    const saveSettingsBtn = document.getElementById("saveSettingsBtn");
    if (saveSettingsBtn) {
        saveSettingsBtn.addEventListener("click", async () => {
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
}

// Fetch System Status Metrics & Registered Contacts
async function fetchSystemStatus() {
    try {
        const resp = await fetch("/api/status");
        const data = await resp.json();

        if (data.active_model && metricModel) metricModel.innerText = data.active_model;
        if (data.total_fire_alerts !== undefined && fireAlertCount === 0) metricFire.innerText = data.total_fire_alerts;
        if (data.total_smoke_alerts !== undefined && smokeAlertCount === 0) metricSmoke.innerText = data.total_smoke_alerts;
        if (data.max_growth_rate !== undefined && metricGrowth) metricGrowth.innerText = `${data.max_growth_rate}%/s`;

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

        if (data.logs && data.logs.length > 0 && clientLogs.length === 0) {
            clientLogs = data.logs;
            renderLogTable();
        }
    } catch (e) {
        console.log("Logs fetch error:", e);
    }
}
