import logging

from dotenv import load_dotenv
from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.plugins import openai, silero

from custom_plugins.nemo_stt import NemoStt

load_dotenv()

logging.basicConfig(level=logging.INFO)


class LoggingAgent(Agent):
    async def on_message(self, message: str):
        logging.info(f"Transcription: {message}")


async def entrypoint(ctx: JobContext):
    await ctx.connect()
    logging.info("Connected to room.")

    agent = LoggingAgent(instructions="You are a transcription agent.")

    session = AgentSession(
        vad=silero.VAD.load(
            max_buffered_speech=15,
            min_silence_duration=0.3,
            prefix_padding_duration=1,
        ),
        stt=NemoStt(),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=openai.TTS(),
    )

    try:
        await session.start(agent=agent, room=ctx.room)
        logging.info("Session started.")
    except Exception as e:
        logging.error(f"Failed to start session: {e}")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
