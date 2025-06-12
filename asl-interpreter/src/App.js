import React, { useRef, useEffect, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as handPoseDetection from '@tensorflow-models/hand-pose-detection';
import ASLInterpreter from './components/ASLInterpreter';

function App() {
  const videoRef = useRef(null);
  const [detector, setDetector] = useState(null);
  const [detectedLetter, setDetectedLetter] = useState('');

  useEffect(() => {
    async function setupCamera() {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
    }
    setupCamera();

    async function loadModel() {
      const model = handPoseDetection.SupportedModels.MediaPipeHands;
      const detectorConfig = {
        runtime: 'mediapipe',
        solutionPath: 'https://cdn.jsdelivr.net/npm/@mediapipe/hands'
      };
      const det = await handPoseDetection.createDetector(model, detectorConfig);
      setDetector(det);
    }
    loadModel();
  }, []);

  useEffect(() => {
    let interval;
    if (detector && videoRef.current) {
      interval = setInterval(async () => {
        const hands = await detector.estimateHands(videoRef.current);
        if (hands.length) {
          const landmarks = hands[0].keypoints.map(p => [p.x, p.y, p.z]);
          const predicted = classify(landmarks);
          if (predicted) {
            setDetectedLetter(predicted);
          }
        }
      }, 200);
    }
    return () => clearInterval(interval);
  }, [detector]);

  function classify(landmarks) {
    // TODO: implement ML classification of landmarks to letters
    // For now return a dummy value to test:
    return ''; // return like 'A', 'B', etc.
  }

  return (
    <div>
      <h1>ASL Interpreter</h1>
      <video ref={videoRef} width="640" height="480" />
      <p>Latest Letter: {detectedLetter}</p>

      {/* Sentence detection and TTS logic handled here */}
      <ASLInterpreter letter={detectedLetter} />
    </div>
  );
}

export default App;

