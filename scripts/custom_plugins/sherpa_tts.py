import asyncio
import json
from typing import Optional

import aiofiles
import httpx
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.utils import is_given


class TTS(tts.TTS):
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:5000",
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=False,
            ),
            sample_rate=16000,
            num_channels=1,
        )

        self.model = model
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.AsyncClient()

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ):
        return ChunkedStream(
            tts=self,
            input_text=text,
            model=self.model,
            url=self.base_url,
            conn_options=conn_options,
            client=self.client,
        )


class ChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        tts: TTS,
        input_text: str,
        model: str,
        url: str,
        conn_options: APIConnectOptions,
        client: httpx.AsyncClient,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._model = model
        self._url = url
        self._client = client

    async def _run(self):
        endpoint = f"{self._url}/speech/generate-speech"
        request_id = utils.shortuuid()
        decoder = utils.codecs.AudioStreamDecoder(
            sample_rate=self._tts.sample_rate,
            num_channels=self._tts.num_channels,
        )

        payload = {
            "text": self.input_text,
            "model": self._model,
        }
        headers = {"Content-Type": "application/json"}

        try:
            response = await self._client.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=httpx.Timeout(30.0, connect=self._conn_options.timeout),
            )
            response.raise_for_status()
            json_response = response.json()
            file_path = json_response.get("file_url")

            if not file_path:
                raise APIStatusError("No file_url in response", status_code=500)

            # Load audio from local file path
            async with aiofiles.open(file_path, "rb") as f:
                audio_bytes = await f.read()

            emitter = tts.SynthesizedAudioEmitter(
                event_ch=self._event_ch,
                request_id=request_id,
            )

            decoder.push(audio_bytes)
            decoder.end_input()

            async for frame in decoder:
                emitter.push(frame)

            emitter.flush()

        except httpx.TimeoutException:
            raise APITimeoutError() from None
        except httpx.HTTPStatusError as e:
            raise APIStatusError(
                str(e),
                status_code=e.response.status_code,
                request_id=request_id,
                body=e.response.text,
            ) from None
        except Exception as e:
            raise APIConnectionError() from e
        finally:
            await decoder.aclose()
