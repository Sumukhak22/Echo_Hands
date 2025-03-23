import React, { useState, useEffect, useRef } from 'react';
import './Learn.css';

function IDk() {
  const [isRecognizing, setIsRecognizing] = useState(false);
  const [backendStatus, setBackendStatus] = useState('Checking...');
  const [detectedSign, setDetectedSign] = useState('No sign detected');
  const [confidence, setConfidence] = useState(0);
  const [debugInfo, setDebugInfo] = useState('');
  const [processedImage, setProcessedImage] = useState(null);
  
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  
  const BACKEND_URL = 'http://localhost:5000' ;
  
  // Check backend health
  const checkBackendStatus = async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/api/health`);
      const data = await response.json();
      setBackendStatus(
        data.status === 'ok' 
          ? (data.model_loaded ? 'Connected (Model Loaded)' : 'Connected (No Model)')
          : 'Error'
      );
    } catch (err) {
      console.error("Backend connection error:", err);
      setBackendStatus('Disconnected');
    }
  };

  // Initialize webcam
  useEffect(() => {
    checkBackendStatus();
    
    async function setupCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480 },
          audio: false
        });
        
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          streamRef.current = stream;
        }
      } catch (err) {
        console.error("Error accessing webcam:", err);
        alert("Error accessing webcam. Please ensure you have a webcam connected and have given permission to access it.");
      }
    }
    
    setupCamera();
    
    // Cleanup function
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, []);
  
  // Toggle recognition
  const toggleRecognition = () => {
    setIsRecognizing(!isRecognizing);
    if (!isRecognizing) {
      setDetectedSign('No sign detected');
      setConfidence(0);
    }
  };
  
  // Send frame to backend for processing
  const processFrame = async () => {
    if (!isRecognizing || !videoRef.current || !canvasRef.current) return;
    
    const canvas = canvasRef.current;
    const video = videoRef.current;
    const context = canvas.getContext('2d');
    
    // Match canvas dimensions to video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    // Draw current video frame to canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Get image data as base64
    const imageData = canvas.toDataURL('image/jpeg');
    
    try {
      const response = await fetch(`${BACKEND_URL}/api/recognize`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: imageData }),
      });
      
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      
      const result = await response.json();
      
      // Update UI with results
      if (result.sign) setDetectedSign(result.sign);
      if (result.confidence !== undefined) setConfidence(result.confidence);
      if (result.debug) setDebugInfo(result.debug);
      if (result.processed_image) setProcessedImage(result.processed_image);
      
    } catch (err) {
      console.error("Error sending frame to backend:", err);
      setDebugInfo(`Error: ${err.message}`);
    }
  };
  
  // Process frames at regular intervals when recognition is active
  useEffect(() => {
    let intervalId;
    if (isRecognizing) {
      intervalId = setInterval(processFrame, 100); // ~10 frames per second
    }
    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [isRecognizing]);
  
  return (
    <div className="new-app-container">
      {/* Top section with two boxes */}
      <div className="top-section">
        {/* Left box: Video for Learning (static placeholder) */}
        <div className="box learning-box">
          <h2>Video for Learning</h2>
          <div className="learning-video-placeholder">
            <div className="video-timestamp">0:00</div>
          </div>
        </div>
        
        {/* Right box: Video for Recognition (webcam feed) */}
        <div className="box recognition-box">
          <h2>Video for Recognition</h2>
          <div className="video-wrapper">
            <video 
              ref={videoRef} 
              autoPlay 
              playsInline
              className={isRecognizing ? 'hidden' : ''}
            />
            {processedImage && isRecognizing && (
              <img 
                src={processedImage} 
                alt="Processed video feed" 
                className="processed-feed" 
              />
            )}
            <canvas ref={canvasRef} className="hidden-canvas" />
          </div>
          <div className="btn-group">
            <button 
              className={`toggle-btn ${isRecognizing ? 'active' : ''}`}
              onClick={toggleRecognition}
            >
              {isRecognizing ? 'Stop Webcam' : 'Start Webcam'}
            </button>
          </div>
        </div>
      </div>
      
      {/* Welcome section */}
      <div className="welcome-section">
        Welcome
      </div>
      
      {/* Bottom buttons */}
      <div className="bottom-buttons">
        <button className="action-btn">Replay</button>
        <button className="action-btn">Continue</button>
      </div>
      
      {/* Results panel (kept logic the same) */}
      <div className="results-panel">
        <div className="backend-status">
          Backend Status: <span>{backendStatus}</span>
        </div>
        <div className="detected-sign">
          <strong>Detected Sign:</strong> {detectedSign}
        </div>
        <div className="confidence">
          <strong>Confidence:</strong> {confidence.toFixed(1)}%
        </div>
        <div className="debug-info">{debugInfo}</div>
      </div>
    </div>
  );
}

export default IDk;
