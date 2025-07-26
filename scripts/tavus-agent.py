import os
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import AgentSession, RoomOutputOptions
from livekit.plugins import tavus
from livekit.plugins import deepgram, elevenlabs, openai, silero
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
)


load_dotenv()


async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()

    agent = Agent(
        instructions="You are a friendly voice assistant built by LiveKit.",
    )
    session = AgentSession(
        vad=silero.VAD.load(),
        stt=deepgram.STT(model="nova-3", language="multi"),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=elevenlabs.TTS(),
    )

    avatar = tavus.AvatarSession(
        replica_id=os.getenv("TAVUS_REPLICA_ID"),
        persona_id=os.getenv("TAVUS_PERSONA_ID"),
    )

    # Start the avatar and wait for it to join
    await avatar.start(session, room=ctx.room)

    # Start your agent session with the user

    await session.start(agent=agent, room=ctx.room)
    await session.generate_reply(instructions="greet the user and ask about their day")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
