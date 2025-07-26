import io
import logging
import wave

import requests
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

        # Convert raw audio to WAV in memory
        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as wav_file:
            wav_file.setnchannels(buffer.num_channels)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(buffer.sample_rate)
            wav_file.writeframes(buffer.data)

        wav_io.seek(0)

        # Load with pydub and resample if needed
        audio = AudioSegment.from_file(wav_io, format="wav")
        if audio.frame_rate != 16000:
            audio = audio.set_frame_rate(16000)
        if audio.channels != 1:
            audio = audio.set_channels(1)

        # Export resampled audio as wav to a new buffer
        resampled_io = io.BytesIO()
        audio.export(resampled_io, format="wav")
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
