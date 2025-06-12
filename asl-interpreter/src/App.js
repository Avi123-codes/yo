import React, { useRef, useEffect, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as handPoseDetection from '@tensorflow-models/hand-pose-detection';
import ASLInterpreter from './components/ASLInterpreter';

function App() {
  const videoRef = useRef(null);
  const [detector, setDetector] = useState(null);
  const [aslModel, setAslModel] = useState(null);
  const [detectedLetter, setDetectedLetter] = useState('');

  // Setup webcam and MediaPipe detector
  useEffect(() => {
    async function setupCamera() {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
    }

    async function loadDetector() {
      const model = handPoseDetection.SupportedModels.MediaPipeHands;
      const detectorConfig = {
        runtime: 'mediapipe',
        solutionPath: 'https://cdn.jsdelivr.net/npm/@mediapipe/hands'
      };
      const det = await handPoseDetection.createDetector(model, detectorConfig);
      setDetector(det);
    }

    async function loadASLModel() {
      const model = await tf.loadLayersModel('/models/asl_model.json');
      setAslModel(model);
    }

    setupCamera();
    loadDetector();
    loadASLModel();
  }, []);

  // Detect hands and classify to ASL letter
  useEffect(() => {
    let interval;
    if (detector && aslModel && videoRef.current) {
      interval = setInterval(async () => {
        const hands = await detector.estimateHands(videoRef.current);
        if (hands.length) {
          const landmarks = hands[0].keypoints.map(p => [p.x, p.y, p.z]);
          const letter = classify(landmarks);
          if (letter) {
            setDetectedLetter(letter);
          }
        }
      }, 500);
    }
    return () => clearInterval(interval);
  }, [detector, aslModel]);

  // Classify hand landmarks into ASL letters
  function classify(landmarks) {
    if (!aslModel) return '';

    // Convert to tensor and predict
    const input = tf.tensor([landmarks.flat()]);
    const prediction = aslModel.predict(input);
    const predictedIndex = prediction.argMax(-1).dataSync()[0];
    const letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';

    return letters[predictedIndex];
  }

  return (
    <div>
      <h1>ASL Interpreter</h1>
      <video ref={videoRef} width="640" height="480" style={{ border: '1px solid black' }} />
      <p>Detected Letter: {detectedLetter}</p>
      <ASLInterpreter letter={detectedLetter} />
    </div>
  );
}

export default App;


