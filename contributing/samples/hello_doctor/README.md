# hello_doctor

A health assessment agent sample that demonstrates safe, educational health conversations with structured intake questions and risk assessment tools.

## Features

- **Structured health intake**: Asks six key questions (age, smoking, alcohol, medical conditions/medications, allergies, lifestyle) before providing any advice
- **Session state tracking**: Uses `log_health_answer` tool to build a longitudinal picture of user responses
- **Risk assessment**: Uses `summarize_risk_profile` tool to provide non-diagnostic risk summaries
- **Strong safety disclaimers**: Always emphasizes that it is not a medical professional and directs users to licensed healthcare providers
- **Few-shot examples**: Includes examples for handling mild symptoms, concerning symptoms, and supplement questions

## Safety

**Important**: This agent is for **educational purposes only**. It:
- Does NOT diagnose, treat, or prescribe
- Does NOT replace professional medical advice
- Always directs users to licensed healthcare professionals for any real health concerns
- Emphasizes emergency care for serious symptoms

## Files

- `agent.py`: Defines the `root_agent` with safety instructions, tools, and few-shot examples
- `main.py`: CLI demo script showing how to run the agent programmatically

## Running

### Via CLI script

```bash
# Make sure you have GOOGLE_GENAI_API_KEY set in .env
python contributing/samples/hello_doctor/main.py
```

### Via ADK Web UI

```bash
adk web contributing/samples
```

Then select `hello_doctor` from the app dropdown in the web UI at `http://127.0.0.1:8000`.

## Configuration

Create a `.env` file in the project root with:

```env
GOOGLE_GENAI_API_KEY=your_api_key_here
```

Or configure Vertex AI credentials if using the Vertex AI backend.

