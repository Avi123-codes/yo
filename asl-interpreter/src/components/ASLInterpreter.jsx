import React, { useState, useEffect } from 'react';

const ASLInterpreter = ({ letter }) => {
  const [letterBuffer, setLetterBuffer] = useState('');
  const [wordBuffer, setWordBuffer] = useState('');
  const [sentence, setSentence] = useState('');
  const [lastLetterTime, setLastLetterTime] = useState(Date.now());

  useEffect(() => {
    const interval = setInterval(() => {
      const now = Date.now();

      if (now - lastLetterTime > 1000 && letterBuffer) {
        setWordBuffer(prev => prev + letterBuffer);
        setLetterBuffer('');
      }

      if (now - lastLetterTime > 2000 && wordBuffer) {
        setSentence(prev => prev + ' ' + wordBuffer);
        setWordBuffer('');
      }
    }, 500);

    return () => clearInterval(interval);
  }, [letterBuffer, wordBuffer, lastLetterTime]);

  useEffect(() => {
    if (letter && letter !== '') {
      setLetterBuffer(prev => prev + letter);
      setLastLetterTime(Date.now());
    }
  }, [letter]);

  const speakText = (text) => {
    const utterance = new SpeechSynthesisUtterance(text);
    window.speechSynthesis.speak(utterance);
  };

  return (
    <div>
      <h2>Live Sentence:</h2>
      <p>{sentence}</p>
      <button onClick={() => speakText(sentence)}>Speak</button>
    </div>
  );
};

export default ASLInterpreter;

