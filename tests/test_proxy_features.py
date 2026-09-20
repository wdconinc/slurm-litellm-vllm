import os
import pytest
from openai import OpenAI

@pytest.fixture(scope="module")
def client():
    # Ensure variables exist, else the test should fail gracefully
    api_base = os.getenv("OPENAI_API_BASE")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_base or not api_key:
        pytest.fail("Environment variables OPENAI_API_BASE and OPENAI_API_KEY must be set. Source run/endpoint.env first.")
    
    return OpenAI()

@pytest.fixture(scope="module")
def model_name():
    return os.getenv("OPENAI_MODEL", "my-local-model")

def test_basic_chat_completion(client, model_name):
    """Test a simple user prompt to ensure the proxy is alive and generating text."""
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": "Say exactly the word 'Banana' and nothing else."}],
        temperature=0.01,
        max_tokens=10
    )
    content = response.choices[0].message.content.lower()
    assert "banana" in content, f"Expected 'banana', got: {content}"

def test_system_prompt_adherence(client, model_name):
    """Test that the model respects system prompt instructions."""
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a French translator. Always respond by translating the user's input to French."},
            {"role": "user", "content": "Hello!"}
        ],
        temperature=0.1,
        max_tokens=20
    )
    content = response.choices[0].message.content.lower()
    assert "bonjour" in content or "salut" in content, f"Model did not translate to French. Got: {content}"

def test_tool_calling(client, model_name):
    """Test if the model supports OpenAI-compatible tool/function calling."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            }
        }
    ]

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": "What's the weather like in Boston today?"}],
        tools=tools,
        tool_choice="auto",
        temperature=0.1
    )

    message = response.choices[0].message
    # Check if the model decided to call a tool
    assert message.tool_calls is not None, "Model did not generate a tool call. It may not support function calling or ignoring the tool schema."
    assert len(message.tool_calls) > 0, "Model generated empty tool calls list."
    
    tool_call = message.tool_calls[0]
    assert tool_call.function.name == "get_current_weather", f"Expected get_current_weather, got {tool_call.function.name}"
    assert "Boston" in tool_call.function.arguments or "boston" in tool_call.function.arguments.lower(), f"Model didn't extract 'Boston' correctly. Got arguments: {tool_call.function.arguments}"

def test_multi_turn_conversation(client, model_name):
    """Test that the model retains context over multiple messages."""
    messages = [
        {"role": "user", "content": "My favorite color is emerald green."},
        {"role": "assistant", "content": "That's a beautiful color!"},
        {"role": "user", "content": "What is my favorite color?"}
    ]
    
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.1,
        max_tokens=20
    )
    
    content = response.choices[0].message.content.lower()
    assert "emerald" in content or "green" in content, f"Model failed to recall context. Got: {content}"
