import argparse
import uvicorn
import uuid
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import AgentCapabilities, AgentCard, AgentSkill, Message, Part, TextPart
from a2a.utils import get_message_text, new_task, new_agent_text_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue

class MockForecasterAgent:
    async def run(self, message: Message, updater: TaskUpdater) -> None:
        prompt = get_message_text(message)
        
        # Robustly count missing values in the CSV data
        # We look for lines where the last column (endog) is empty or 'nan'
        lines = prompt.strip().split("\n")
        num_required = 0
        for line in lines:
            if "," in line:
                parts = line.split(",")
                # The prompt structure puts the CSV in the middle. 
                # We check the last part of each comma-separated line.
                last_val = parts[-1].strip().lower()
                # When endog is NaN, it usually appears as an empty string or 'nan' in CSV
                if last_val == "" or last_val == "nan":
                    num_required += 1
        
        # If we didn't find any (maybe regex failed or format changed), 
        # try the old simple count as a safety fallback
        if num_required == 0:
            num_required = prompt.count(",nan") or prompt.count(",NaN") or 60
            
        print(f"Mock forecaster determined problem size: {num_required}")
            
        # Generate dummy floats
        forecast_values = [10.5 + (i * 0.01) for i in range(num_required)]
        
        # Return as a bracketed list to match the prompt's request
        response_text = f"Here is the forecast: {forecast_values}"

        await updater.add_artifact(
            parts=[Part(root=TextPart(text=response_text))],
            name="Forecast",
        )

class MockExecutor(AgentExecutor):
    def __init__(self):
        self.agent = MockForecasterAgent()
        self.agents = {}

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        msg = context.message
        task = context.current_task
        if not task:
            task = new_task(msg)
            await event_queue.enqueue_event(task)
        
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.start_work()
        await self.agent.run(msg, updater)
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9010)
    args = parser.parse_args()

    agent_card = AgentCard(
        name="Mock Forecaster",
        description="A simple mock forecaster for testing.",
        url=f"http://127.0.0.1:{args.port}/",
        version='1.0.0',
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(streaming=False),
        skills=[AgentSkill(id="mock", name="mock", description="mock", tags=[])]
    )

    app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=DefaultRequestHandler(agent_executor=MockExecutor(), task_store=InMemoryTaskStore())
    )
    uvicorn.run(app.build(), host="127.0.0.1", port=args.port)

if __name__ == "__main__":
    main()
