# hello_doctor sample

This sample demonstrates a simple **health-oriented educational agent**
built with ADK. It is designed to provide high-level information about
symptoms and wellness while repeatedly reminding users that it is **not
a medical professional** and cannot diagnose, treat, or prescribe.

> This sample is for educational and testing purposes only and must not
> be used as a substitute for professional medical advice, diagnosis, or
> treatment.

## Files

- `agent.py`: Defines `hello_doctor_agent` and two tools:
  - `log_health_answer(question, answer, tool_context)`: Logs structured
    question/answer pairs into the session state so the agent can build a
    longitudinal view of the conversation.
  - `summarize_risk_profile(tool_context)`: Produces a brief, **non-
    diagnostic** textual summary based on the logged answers, which the
    agent can incorporate into its final response.
- `main.py`: Minimal CLI demo that runs a one-shot health assessment
  prompt using the agent and prints the response to the console.

## Running the sample

From the repository root:

```bash
source .venv/bin/activate
python contributing/samples/hello_doctor/main.py
```

This sends a single, long health-assessment request to the agent and
prints the model's reply. To experiment interactively, use the ADK web
UI instead.

## Using with ADK Web

Start the web server from the project root:

```bash
source .venv/bin/activate
adk web contributing/samples
```

Then open `http://127.0.0.1:8000` in a browser and choose the
`hello_doctor` app from the dropdown. You can then chat with the agent
and, if desired, instruct it to log answers and summarize its risk
profile using the tools described above.

## Configuration and API keys

This sample relies on a Gemini model (`gemini-2.0-flash`) and expects
credentials to be provided via environment variables, typically loaded
from a local `.env` file (which should **not** be committed to source
control). A common configuration is:

```env
GOOGLE_GENAI_API_KEY=your_api_key_here
```

Contributors and users should supply their own API keys or Vertex AI
configuration when running the sample locally.


