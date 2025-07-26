import logging

from dotenv import load_dotenv
from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.plugins import assemblyai, deepgram, openai, silero

from my_plugins.local_stt import NemoStt

# from my_plugins.nemo_plugin import NeMoSTT

# from plugins.nemo_stt import NeMoSTT

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
        vad=silero.VAD.load(),
        # stt=deepgram.STT(model="nova-3", language="multi"),
        # stt=openai.STT(model="gpt-4o-transcribe"),
        # stt=assemblyai.STT(),
        # stt=NeMoSTT(capabilities=None),
        stt=NemoStt(),
    )

    try:
        await session.start(agent=agent, room=ctx.room)
        logging.info("Session started.")
    except Exception as e:
        logging.error(f"Failed to start session: {e}")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
