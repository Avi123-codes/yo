import express from 'express';
import mongoose from 'mongoose';
import cors from 'cors';

const app = express();
app.use(cors());
app.use(express.json());

const mongoUri = process.env.MONGO_URI || 'mongodb://localhost/asl';
mongoose.connect(mongoUri, { useNewUrlParser: true, useUnifiedTopology: true });

const translationSchema = new mongoose.Schema({
  text: String,
  createdAt: { type: Date, default: Date.now }
});
const Translation = mongoose.model('Translation', translationSchema);

app.post('/api/translations', async (req, res) => {
  try {
    const translation = new Translation({ text: req.body.text });
    await translation.save();
    res.json(translation);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

const port = process.env.PORT || 3001;
app.listen(port, () => console.log(`Server running on port ${port}`));
