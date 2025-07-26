from dotenv import load_dotenv
from langfuse import get_client
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
)
from livekit.plugins import deepgram, elevenlabs, openai, silero

load_dotenv()

# Langfuse client initialization
langfuse = get_client()


async def entrypoint(ctx: JobContext):
    # Langfuse: create root span
    span = langfuse.start_span(name="livekit-session")

    try:
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

        # Langfuse: nested span for starting session
        nested_span = span.start_span(name="start-session")
        await session.start(agent=agent, room=ctx.room)
        nested_span.update(output="Session started")
        nested_span.end()

        # Langfuse: nested span for generating reply
        reply_span = span.start_span(name="generate-reply")
        await session.generate_reply(
            instructions="greet the user and ask about their day"
        )
        reply_span.update(output="Greeting generated")
        reply_span.end()

        span.update(output="Session completed successfully")

    except Exception as e:
        span.update(output=f"Error occurred: {str(e)}", level="ERROR")
        raise

    finally:
        span.end()
        langfuse.flush()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
