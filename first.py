from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
)
from livekit.plugins import deepgram, elevenlabs, silero, openai

# Crucially, use Langfuse's OpenAI integration for automatic LLM tracing
# from langfuse.openai import openai
from dotenv import load_dotenv
from langfuse import get_client, observe

load_dotenv()

# Initialize the client once when the script starts
langfuse = get_client()


@observe()
async def entrypoint(ctx: JobContext):
    """
    This is the main entrypoint for the LiveKit agent worker.
    Each time this function is called, a new Langfuse trace will be created.
    """
    # NOTE: The try...finally block with langfuse.shutdown() has been removed.
    # The Langfuse SDK will handle shutdown automatically when the worker process exits.

    langfuse.update_current_trace(
        name=f"agent-run-{ctx.room.name}",
        # Use the participant's identity as the user_id for the trace
        # user_id=ctx.participant.identity,
        # Use the unique session ID instead of the room name
        session_id=ctx.room.name,
        metadata={
            "livekit_room_name": ctx.room.name,
        },
    )

    await ctx.connect()

    with langfuse.start_as_current_span(name="agent-session") as span:
        vad = silero.VAD.load()
        stt = deepgram.STT(model="nova-3", language="multi")
        # The openai object from langfuse.openai is used here
        llm = openai.LLM(model="gpt-4o-mini")
        tts = elevenlabs.TTS()

        span.update(
            input={
                "instructions": "You are a friendly voice assistant built by LiveKit."
            },
            metadata={
                "vad_plugin": vad.__class__.__name__,
                "stt_plugin": stt.__class__.__name__,
                # "stt_model": stt.model,  # Add STT model info
                "llm_plugin": llm.__class__.__name__,
                "llm_model": llm,
                "tts_plugin": tts.__class__.__name__,
            },
        )

        agent = Agent(
            instructions="You are a friendly voice assistant built by LiveKit.",
        )

        session = AgentSession(
            vad=vad,
            stt=stt,
            llm=llm,
            tts=tts,
        )

        await session.start(agent=agent, room=ctx.room)

        await session.generate_reply(
            instructions="greet the user and ask about their day"
        )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
