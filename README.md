
# ASL Interpreter Example

This repository contains a minimal example of a real-time American Sign Language (ASL) interpreter.

## Front-end (`asl-interpreter`)

A React application that accesses the webcam, runs the MediaPipe hand pose detection model via TensorFlow.js and outputs detected letters. The application was created manually and lists its dependencies in `package.json`.

To run the front-end:

```bash
cd asl-interpreter
npm install
npm start
```

## Back-end (`server`)

An optional Node.js/Express server that stores recognised translations in MongoDB. The endpoint `/api/translations` accepts `{ text: "A" }` and stores the record with a timestamp.

```bash
cd server
npm install
npm start
```

Set the `MONGO_URI` environment variable to configure the database connection. The server listens on port `3001` by default.

## Custom ASL Classifier

The file `asl-interpreter/src/App.js` includes a placeholder `classify` function. Implement this function with your own model to convert hand landmarks to ASL letters. The recognised letter is displayed on the page and spoken using the browser's Speech Synthesis API.
