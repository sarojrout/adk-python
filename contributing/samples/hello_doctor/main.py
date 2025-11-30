import asyncio
import time

import agent
from dotenv import load_dotenv
from google.adk import Runner
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.cli.utils import logs
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions.session import Session
from google.genai import types

load_dotenv(override=True)
logs.log_to_tmp_folder()


async def main():
  app_name = "hello_doctor"
  user_id = "user1"

  session_service = InMemorySessionService()
  artifact_service = InMemoryArtifactService()

  runner = Runner(
      app_name=app_name,
      agent=agent.root_agent,
      artifact_service=artifact_service,
      session_service=session_service,
  )

  session = await session_service.create_session(
      app_name=app_name, user_id=user_id
  )

  async def run_prompt(session: Session, new_message: str):
    content = types.Content(
        role="user", parts=[types.Part.from_text(text=new_message)]
    )
    print("** User says:", content.model_dump(exclude_none=True))
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session.id,
        new_message=content,
    ):
      if event.content.parts and event.content.parts[0].text:
        print(f"** {event.author}: {event.content.parts[0].text}")

  start_time = time.time()
  print("Start time:", start_time)
  print("------------------------------------")

  await run_prompt(
      session,
      (
          "I'd like you to perform a high-level health assessment. Ask me "
          "structured questions about my age, lifestyle, symptoms, and "
          "medical history one by one. At the end, provide: "
          "1) a concise longitudinal summary of my situation, "
          "2) general wellness suggestions including over-the-counter "
          "supplements that are commonly considered safe for most adults, "
          "3) clear guidance on which licensed medical professionals I "
          "should talk to and which medical tests I could ask them about. "
          "You must clearly state that you are not a doctor and that your "
          "advice is not a diagnosis or a substitute for professional care."
      ),
  )

  end_time = time.time()
  print("------------------------------------")
  print("End time:", end_time)
  print("Total time:", end_time - start_time)


if __name__ == "__main__":
  asyncio.run(main())


