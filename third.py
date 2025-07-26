from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
)
from livekit.plugins import (
    deepgram,
    elevenlabs,
    openai as livekit_openai,
    silero,
)
from langfuse.openai import openai as langfuse_openai
from langfuse.openai import AsyncOpenAI


from dotenv import load_dotenv

load_dotenv()


async def entrypoint(ctx: JobContext):
    # client = langfuse_openai.AsyncClient()
    # client.chat.completions.create(model="gpt-4o-mini", messages=[], session_id="15")
    async_client = AsyncOpenAI()

    await ctx.connect()

    agent = Agent(
        instructions="You are a friendly voice assistant built by LiveKit.",
    )
    session = AgentSession(
        vad=silero.VAD.load(),
        stt=deepgram.STT(model="nova-3", language="multi"),
        llm=livekit_openai.LLM(
            model="gpt-4o-mini", client=async_client, metadata={"session_id": "15"}
        ),
        tts=elevenlabs.TTS(),
    )

    await session.start(agent=agent, room=ctx.room)
    await session.generate_reply(instructions="greet the user and ask about their day")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
