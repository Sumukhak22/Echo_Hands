import React, { useState, useRef, useEffect } from "react";
import "./Learn.css";

const Video = () => {
  const [signPassed, setSignPassed] = useState(false);
  const videoRef = useRef(null);
  const [isWebcamActive, setIsWebcamActive] = useState(false);

  const learningVideoLabel = "Welcome";

  // Start webcam
  const startWebcam = () => {
    navigator.mediaDevices
      .getUserMedia({ video: true })
      .then((stream) => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          setIsWebcamActive(true);
        }
      })
      .catch((err) => {
        console.error("Error accessing webcam:", err);
        alert("Webcam access denied or not supported.");
      });
  };

  // Stop webcam
  const stopWebcam = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = videoRef.current.srcObject.getTracks();
      tracks.forEach((track) => track.stop());
      setIsWebcamActive(false);
    }
  };

  // Handle replay
  const handleReplay = () => {
    setSignPassed(false);
    startWebcam();
  };

  // Handle continue
  const handleContinue = () => {
    alert("Continuing to the next sign!");
  };

  // Cleanup when component unmounts
  useEffect(() => {
    return () => {
      stopWebcam();
    };
  }, []);

  return (
    <div className="container">
      <div className="video-section">
        <div className="video-box">
          <p>Video for Learning</p>
          <video src="learning_video.mp4" controls></video>
        </div>

        <div className="video-box">
          <p>Video for Recognition</p>
          <video ref={videoRef} autoPlay muted></video>
          <div className="button-group">
            <button onClick={startWebcam} disabled={isWebcamActive}>
              Start Webcam
            </button>
            <button onClick={stopWebcam} disabled={!isWebcamActive}>
              Stop Webcam
            </button>
          </div>
        </div>
      </div>

      <div className="label-section">
        <p>{learningVideoLabel}</p>
      </div>

      <div className="button-section">
        <button onClick={handleReplay} disabled={signPassed}>
          Replay
        </button>
        <button onClick={handleContinue} disabled={!signPassed}>
          Continue
        </button>
      </div>
    </div>
  );
};

export default Video;
