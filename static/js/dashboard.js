/**
 * AI Exam Invigilator - Frontend Dashboard Controller
 * Author: Venkatrao Yasarapu
 */

let lastAlertCount = 0;
const audioAlert = document.getElementById("alertSound");

// Digital Clock
function updateClock() {
    const now = new Date();
    const timeStr = now.toLocaleTimeString();
    const clockEl = document.getElementById("clock");
    if (clockEl) clockEl.innerText = timeStr;
}
setInterval(updateClock, 1000);
updateClock();

// Fetch Telemetry Status
async function updateStatus() {
    try {
        const res = await fetch("/api/status");
        if (!res.ok) return;
        const data = await res.json();

        const pulse = document.getElementById("pulseIndicator");
        const statusText = document.getElementById("systemStatusText");
        const totalViolations = document.getElementById("totalViolationsCount");

        if (totalViolations) {
            totalViolations.innerText = data.total_alerts;
        }

        if (data.monitoring) {
            if (pulse) pulse.className = "pulse-dot active";
            if (statusText) statusText.innerText = "Proctor Active";
        } else {
            if (pulse) pulse.className = "pulse-dot";
            if (statusText) statusText.innerText = "Monitoring Paused";
        }
    } catch (e) {
        console.warn("Status fetch error:", e);
    }
}
setInterval(updateStatus, 1500);

// Fetch Security Event Logs
async function fetchLogs() {
    try {
        const res = await fetch("/api/logs");
        if (!res.ok) return;
        const data = await res.json();

        const logBox = document.getElementById("logScrollBox");
        const liveCountBadge = document.getElementById("liveAlertsCount");
        const poseStatus = document.getElementById("poseStatus");
        const objectStatus = document.getElementById("objectStatus");

        if (liveCountBadge) {
            liveCountBadge.innerText = `${data.logs.length} Events`;
        }

        if (data.logs.length === 0) {
            logBox.innerHTML = `
                <div class="log-placeholder" id="emptyPlaceholder">
                    <i class="fa-regular fa-circle-check"></i>
                    <p>No suspicious activities recorded. System monitoring actively.</p>
                </div>
            `;
            if (poseStatus) poseStatus.innerText = "Center / Focused";
            if (objectStatus) objectStatus.innerText = "Clear";
            return;
        }

        // Check if new alert arrived to play chime
        if (data.total_alerts > lastAlertCount) {
            if (audioAlert) {
                audioAlert.play().catch(() => {});
            }
            lastAlertCount = data.total_alerts;
        }

        // Render logs
        let html = "";
        data.logs.forEach(log => {
            html += `
                <div class="log-item">
                    <div class="log-time">${log.time}</div>
                    <div class="log-text">${log.text}</div>
                </div>
            `;
            // Update quick indicators based on recent events
            if (log.text.includes("looking")) {
                if (poseStatus) poseStatus.innerText = "Suspicious Gaze";
            }
            if (log.text.includes("PROHIBITED")) {
                if (objectStatus) objectStatus.innerText = "Device Flagged!";
            }
        });

        logBox.innerHTML = html;
        logBox.scrollTop = logBox.scrollHeight;
    } catch (err) {
        console.warn("Logs fetch error:", err);
    }
}
setInterval(fetchLogs, 1000);

// Actions
async function startMonitoring() {
    await fetch("/api/start", { method: "POST" });
    const stream = document.getElementById("videoStream");
    if (stream) {
        stream.src = "/video_feed?" + new Date().getTime();
    }
}

async function stopMonitoring() {
    await fetch("/api/stop", { method: "POST" });
}

async function clearLogs() {
    await fetch("/api/clear_logs", { method: "POST" });
    lastAlertCount = 0;
    fetchLogs();
}
