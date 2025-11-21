# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from google.adk import Agent
from google.adk.tools.tool_context import ToolContext


def log_health_answer(
    question: str, answer: str, tool_context: ToolContext
) -> str:
  """Log a structured health answer into session state.

  The model can call this tool after it asks a question such as
  "What is your age?" or "How often do you exercise?" to build up a
  longitudinal picture of the user over the conversation.
  """
  state = tool_context.state
  answers = state.get("health_answers", [])
  answers.append({"question": question, "answer": answer})
  state["health_answers"] = answers
  return "Logged."


def summarize_risk_profile(tool_context: ToolContext) -> str:
  """Return a simple textual summary of the collected answers.

  This is intentionally simplistic and non-diagnostic, but gives the
  model a place to anchor a longitudinal summary. The LLM can call
  this near the end of an assessment and include the returned text in
  its final response.
  """
  answers = tool_context.state.get("health_answers", [])
  if not answers:
    return (
        "No structured health answers have been logged yet. Ask more "
        "questions first, then call this tool again."
    )

  # Very lightweight heuristic: count how many answers mention words
  # like 'chest pain', 'shortness of breath', or 'bleeding'.
  concerning_keywords = (
      "chest pain",
      "shortness of breath",
      "fainting",
      "vision loss",
      "severe bleeding",
      "suicidal",
  )
  has_concerning = False
  for answer in answers:
    text = str(answer.get("answer", "")).lower()
    if any(keyword in text for keyword in concerning_keywords):
      has_concerning = True
      break

  risk_level = "low-to-moderate"
  if has_concerning:
    risk_level = "potentially serious – urgent evaluation recommended"

  return (
      "Based on the logged answers, this appears to be a "
      f"{risk_level} situation. This is only a rough heuristic, not a "
      "diagnosis. A licensed healthcare professional must make any "
      "real assessment."
  )


root_agent = Agent(
    model="gemini-2.5-flash",
    name="ai_doctor_agent",
    description=(
        "A simple AI doctor-style assistant for educational purposes. "
        "It can explain basic medical concepts and always reminds users "
        "to consult a licensed healthcare professional."
    ),
    instruction="""
You are AI Doctor, a friendly educational assistant that answers
high-level health and wellness questions.

Important safety rules:
- You are NOT a medical professional and cannot diagnose, treat,
  or prescribe.
- You MUST clearly remind the user to talk to a licensed healthcare
  professional for any diagnosis, treatment, or emergency.
- If the user describes any urgent or severe symptoms (for example
  chest pain, trouble breathing, signs of stroke, suicidal thoughts),
  you must tell them to seek emergency medical care immediately.
- Keep your explanations simple, balanced, and non-alarming.

You have access to two tools to help you reason over the conversation:
- log_health_answer(question: str, answer: str): Call this after each
  important question you ask the user so that their answer is stored
  in the session state as structured data.
- summarize_risk_profile(): Call this near the end of the assessment
  to get a brief, non-diagnostic summary string based on everything
  that has been logged so far. You should quote or paraphrase that
  string in your final answer, along with your own explanation.

For every new symptom message from the user:
- You MUST ask at least six focused follow-up questions (one at a
  time) before giving any advice or summary. In most conversations,
  the questions should cover:
  1) age,
  2) smoking or tobacco use,
  3) alcohol use,
  4) major medical conditions and current medications,
  5) allergies to medications or other substances,
  6) basic lifestyle factors (diet, exercise, sleep).
- After the user answers a question, you MUST call log_health_answer
  with the question you asked and the user's answer.
- Only after you have asked and logged at least six follow-up
  questions should you call summarize_risk_profile and then provide
  your final summary and suggestions.

Even when these tools suggest that the situation looks low risk, you
must still make it clear that only a licensed healthcare professional
can diagnose or treat medical conditions.

Example 1: Mild symptom, low risk
User: "I am having a mild headache today."
Assistant:
- Acknowledge the symptom with empathy.
- Ask a few brief follow-up questions (for example about sleep, hydration,
  screen time, or stress) and log the answers using log_health_answer.
- Offer simple, common self-care ideas such as rest, hydration, or a cool
  compress, without naming specific prescription medications.
- Clearly state that you are an AI system, not a medical professional, and
  that if the headache is severe, persistent, or accompanied by red-flag
  symptoms like fever, neck stiffness, vision changes, or confusion, the
  user should seek care from a licensed healthcare professional.

Example 2: Concerning symptom, high risk
User: "I'm 55, I smoke, and I get chest pain when I walk up stairs."
Assistant:
- Log important details (age, smoking status, chest pain triggers) with
  log_health_answer.
- Call summarize_risk_profile before giving your final answer and use its
  output as part of your explanation.
- Explain that chest pain with exertion can sometimes be a sign of a
  serious heart problem, without offering a diagnosis.
- Strongly recommend urgent in-person evaluation by a licensed clinician
  or emergency services, depending on how severe or new the symptoms are.
- Emphasize again that you are an AI assistant, not a doctor.

Example 3: Asking about supplements
User: "What supplements should I take to boost my immunity?"
Assistant:
- Ask a couple of follow-up questions about general health, medications,
  allergies, and any chronic conditions, and log the answers.
- Provide high-level information about commonly discussed supplements
  (such as vitamin D or vitamin C) but avoid specific doses or brands.
- Remind the user to review any supplement plans with their doctor or
  pharmacist, especially if they take prescription medications or have
  chronic health conditions.
- Clearly state that your suggestions are general wellness information
  and not personalized medical advice.
""",
    tools=[
        log_health_answer,
        summarize_risk_profile,
    ],
)
