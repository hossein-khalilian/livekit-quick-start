import io
import logging
import wave

import requests
import torch
import torchaudio
from dotenv import load_dotenv
from livekit.agents import stt, utils
from livekit.agents.utils import AudioBuffer
from pydub import AudioSegment

load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("voice-agent")


class NemoStt(stt.STT):
    def __init__(self):
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False)
        )

    async def _recognize_impl(
        self, buffer: AudioBuffer, *, language: str | None = None, **kwargs
    ) -> stt.SpeechEvent:
        buffer = utils.merge_frames(buffer)

        # Convert raw audio bytes to torch tensor
        waveform = torch.frombuffer(buffer.data, dtype=torch.int16)
        waveform = (
            waveform.reshape(1, -1)
            if buffer.num_channels == 1
            else waveform.reshape(buffer.num_channels, -1)
        )
        waveform = waveform.float() / 32768.0  # normalize to [-1, 1]
        sample_rate = buffer.sample_rate

        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=16000
            )
            waveform = resampler(waveform)
            sample_rate = 16000

        # Save as wav to memory
        resampled_io = io.BytesIO()
        torchaudio.save(resampled_io, waveform, sample_rate, format="wav")
        resampled_io.seek(0)

        # Send to local transcription service
        url = "http://localhost:8000/transcribe"
        files = {"file": ("audio.wav", resampled_io.read(), "audio/wav")}
        response = requests.post(url, files=files)
        response.raise_for_status()

        resultText = response.json().get("text", "")
        logger.info(f"Transcription: {resultText}")

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                stt.SpeechData(text=resultText or "", language=language or "")
            ],
        )
