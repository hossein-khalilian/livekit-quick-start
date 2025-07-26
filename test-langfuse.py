from langfuse import get_client
from dotenv import load_dotenv

load_dotenv()

langfuse = get_client()

# Create a span without a context manager
span = langfuse.start_span(name="user-request")

# Your processing logic here
span.update(output="Request processed")

# Child spans must be created using the parent span object
nested_span = span.start_span(name="nested-span")
nested_span.update(output="Nested span output")

# Important: Manually end the span
nested_span.end()

# Important: Manually end the parent span
span.end()


# Flush events in short-lived applications
langfuse.flush()
