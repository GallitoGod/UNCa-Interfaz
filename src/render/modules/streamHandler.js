import { streamUrl } from "./constants.js";

export function initVideoStream(videoElement) {
    const outputCanvas = document.getElementById("outputCanvas");
    const outputCtx    = outputCanvas.getContext("2d");

    const captureCanvas = document.createElement("canvas");
    const captureCtx    = captureCanvas.getContext("2d");

    let ws;
    let animationFrameId   = null;
    let waitingForResponse = false;
    let intentionallyClosed = false;
    let retryDelay = 1000;

    function connect() {
        ws = new WebSocket(streamUrl);

        ws.onopen = () => {
            console.log("WebSocket conectado");
            retryDelay = 1000;
            startFrameLoop();
        };

        ws.onclose = () => {
            cancelAnimationFrame(animationFrameId);
            animationFrameId = null;
            if (!intentionallyClosed) {
                console.log(`WebSocket cerrado. Reconectando en ${retryDelay}ms...`);
                setTimeout(connect, retryDelay);
                retryDelay = Math.min(retryDelay * 2, 10000);
            } else {
                console.log("WebSocket cerrado.");
            }
        };

        ws.onerror = (err) => {
            console.error("WebSocket error:", err);
        };

        ws.onmessage = (event) => {
            const img = new Image();
            img.onload = () => {
                if (outputCanvas.width  !== img.width ||
                    outputCanvas.height !== img.height) {
                    outputCanvas.width  = img.width;
                    outputCanvas.height = img.height;
                }
                outputCtx.drawImage(img, 0, 0);
            };
            img.src = "data:image/jpeg;base64," + event.data;
            waitingForResponse = false;
        };
    }

    function startFrameLoop() {
        function tick() {
            if (
                !waitingForResponse &&
                ws.readyState === WebSocket.OPEN &&
                videoElement.readyState >= HTMLMediaElement.HAVE_ENOUGH_DATA
            ) {
                const vw = videoElement.videoWidth;
                const vh = videoElement.videoHeight;
                if (vw > 0 && vh > 0) {
                    captureCanvas.width  = vw;
                    captureCanvas.height = vh;
                    captureCtx.drawImage(videoElement, 0, 0);
                    const frameData = captureCanvas.toDataURL("image/jpeg", 0.8);
                    ws.send(frameData);
                    waitingForResponse = true;
                }
            }
            animationFrameId = requestAnimationFrame(tick);
        }
        tick();
    }

    connect();

    return {
        close() {
            intentionallyClosed = true;
            cancelAnimationFrame(animationFrameId);
            if (ws) ws.close();
        }
    };
}
