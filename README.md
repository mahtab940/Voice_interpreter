# Claude AI Voice Tutor — Assignment 11

## Overview

A real-time voice-controlled AI tutoring application developed using Python.

The application allows users to speak naturally, ask questions, and receive spoken AI-generated responses.

## Technologies

- Python
- Anthropic Claude API
- Faster-Whisper (Speech-to-Text)
- Piper TTS (Text-to-Speech)
- Silero VAD (Voice Activity Detection)
- SoundDevice
- PyTorch

## Features

- Automatic microphone recording
- Speech recognition
- AI-generated answers
- Text-to-speech responses
- Automatic end-of-speech detection
- Experimental voice interruption
- Performance logging
- Error handling

## System Architecture

Microphone → Silero VAD → Faster-Whisper → Claude AI → Piper TTS → Speakers

## Installation

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Install FFmpeg:

```bash
sudo apt install ffmpeg
```

Configure a `.env` file containing your Anthropic API key and the Claude model available to your account.

Download the Piper English voice model `en_US-lessac-medium`.

## Usage

Start the application:

```bash
python voice.py
```

Speak naturally into the microphone. The application automatically detects speech and processes questions.

For reliable interruption tests, headphones are recommended.

## Performance Evaluation

The application records speech recognition, Claude response, Piper generation, and playback times.

Evaluation data is stored in the `results/` directory.

## Limitations

- Requires internet connectivity for Claude API requests.
- Voice interruption remains experimental.
- Speaker echo can cause false interruption triggers.
- Performance depends on microphone quality and system hardware.

## Assignment

Assignment 11 — Real-Time Voice-Controlled AI Application.
