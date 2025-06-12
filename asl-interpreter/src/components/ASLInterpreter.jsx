import React, { useState, useEffect } from 'react';

const ASLInterpreter = () => {
  const [letterBuffer, setLetterBuffer] = useState('');
  const [wordBuffer, setWordBuffer] = useState('');
  const [sentence, setSentence] = useState('');
  const [lastLetterTime, setLastLetterTime] = useState(Date.now());

  useEffect(() => {
    const interval = setInterval(() => {
      if (Date.now() - lastLetterTime > 1000 && letterBuffer) {
        setWordBuffer(prev => prev + letterBuffer);
        setLetterBuffer('');
      }
      if (Date.now() - lastLetterTime > 2000 && wordBuffer) {
        setSentence(prev => prev + ' ' + wordBuffer);
        setWordBuffer('');
      }
    }, 500);
    return () => clearInterval(interval);
  }, [letterBuffer, wordBuffer, lastLetterTime]);

  const handleNewLetter = (letter) => {
    setLetterBuffer(prev => prev + letter);
    setLastLetterTime(Date.now());
  };

  const speakText = (text) => {
    const utterance = new SpeechSynthesisUtterance(text);
    speechSynthesis.speak(utterance);
  };

  // TEMP TEST: simulate a new letter being detected every 3 seconds
  useEffect(() => {
    const fakeLetters = ['H', 'E', 'L', 'L', 'O'];
    let index = 0;
    const interval = setInterval(() => {
      if (index < fakeLetters.length) {
        handleNewLetter(fakeLetters[index]);
        index++;
      }
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div>
      <h2>Live Sentence: {sentence}</h2>
      <button onClick={() => speakText(sentence)}>Speak</button>
    </div>
  );
};

export default ASLInterpreter;
