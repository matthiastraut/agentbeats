import logging
from typing import Any
import numpy as np
import pandas as pd
import re
import statsmodels.api as sm
from pydantic import BaseModel, HttpUrl, ValidationError

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart, DataPart
from a2a.utils import get_message_text, new_agent_text_message
from messenger import Messenger

logger = logging.getLogger(__name__)

class EvalRequest(BaseModel):
    """Request format sent by the AgentBeats platform to host agents."""
    participants: dict[str, HttpUrl] # role -> agent URL
    config: dict[str, Any]


def generate_forecasting_problem():
    """Generates a random time series forecasting problem."""
    freq_rand = np.random.choice(["D", "W", "ME", "QE"])
    periods_per_year = {"D": 365, "W": 52, "ME": 12, "QE": 4, "A": 1}
    
    idx = pd.date_range(start="2000-01-01", periods=25*periods_per_year[freq_rand], freq=freq_rand)
    num_exog_series = np.random.randint(1, 5)
    
    df = pd.DataFrame(index=idx, columns=[f"series{i}" for i in range(num_exog_series)], dtype=float)
    
    ar_terms = np.random.normal(0, 0.5, size=np.random.randint(1, 3))
    coefs = np.random.normal(0, 1, size=num_exog_series)
    
    for i in range(num_exog_series):
        sigma = np.random.normal(0, 1, size=len(df))
        df.iloc[:, i] = sigma.copy()
        for t in range(len(ar_terms), len(df)):
            for k in range(len(ar_terms)):
                df.iloc[t, i] += ar_terms[k] * df.iloc[t-k-1, i]

    df['endog'] = (coefs @ df.iloc[:, :num_exog_series].T).values + np.random.normal(0, 0.5, size=len(df))
    
    # Hide the last 5 year's worth of endog data
    test_periods = 5*periods_per_year[freq_rand]
    df_provided = df.copy()
    df_provided.iloc[-test_periods:, df_provided.columns.get_loc('endog')] = np.nan
    df_solution = df['endog'].iloc[-test_periods:]
    
    return df_provided, df_solution


class Agent:
    required_roles: list[str] = ["forecaster"]
    required_config_keys: list[str] = ["num_rounds"]

    def __init__(self):
        self.messenger = Messenger()

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        missing_roles = set(self.required_roles) - set(request.participants.keys())
        if missing_roles:
            return False, f"Missing roles: {missing_roles}"

        missing_config_keys = set(self.required_config_keys) - set(request.config.keys())
        if missing_config_keys:
            return False, f"Missing config keys: {missing_config_keys}"

        return True, "ok"

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """
        1. Validate request
        2. Generate problem
        3. Send problem to forecaster
        4. Evaluate response
        """
        input_text = get_message_text(message)
        try:
            request: EvalRequest = EvalRequest.model_validate_json(input_text)
            ok, msg = self.validate_request(request)
            if not ok:
                await updater.reject(new_agent_text_message(msg))
                return
        except ValidationError as e:
            await updater.reject(new_agent_text_message(f"Invalid request: {e}"))
            return

        await updater.update_status(
            TaskState.working, new_agent_text_message("Generating forecasting problem...")
        )
        
        df_provided, df_solution = generate_forecasting_problem()
        
        await updater.update_status(
            TaskState.working, new_agent_text_message("Calling forecaster...")
        )

        try:
            # Send the problem as a CSV or JSON string to the participant
            prompt = f"Please provide a forecast for the 'endog' column for the NaN periods in the following data (CSV format):\n\n{df_provided.to_csv()} as a list of float values."
            
            forecaster_url = str(request.participants["forecaster"])
            response_text = await self.messenger.talk_to_agent(prompt, forecaster_url)
            
            await updater.update_status(
                TaskState.working, new_agent_text_message("Evaluating forecast...")
            )

            # Parsing and RMSE Calculation
            try:
                # 1. Try to find a bracketed list first (matches the "list of float values" prompt)
                list_match = re.search(r"\[(.*?)\]", response_text, re.DOTALL)
                if list_match:
                    # Extract all numbers from within the brackets
                    forecast_values = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+", list_match.group(1))]
                else:
                    # 2. Fallback: Extract only numbers with a decimal point to avoid date integers (YYYY-MM-DD)
                    # This specifically targets "float values" as requested in the prompt
                    forecast_values = [float(x) for x in re.findall(r"[-+]?\d*\.\d+", response_text)]
                
                # print(f"Extracted values: {forecast_values}")
                
                if len(forecast_values) != len(df_solution):
                    error_msg = f"Expected {len(df_solution)} values, but got {len(forecast_values)}"
                    rmse = -1.0
                else:
                    # Calculate RMSE using numpy
                    actual = df_solution.values
                    forecast = np.array(forecast_values)
                    rmse = np.sqrt(np.mean((actual - forecast) ** 2))
                    error_msg = f"RMSE calculated: {rmse:.4f}"

                await updater.add_artifact(
                    parts=[
                        Part(root=TextPart(text=f"Evaluation Summary: {error_msg}\nRaw Response: {response_text}")),
                        Part(root=DataPart(data={"rmse": float(rmse), "status": "completed"})),
                    ],
                    name="Evaluation Result",
                )
            except Exception as e:
                await updater.update_status(
                    TaskState.working, new_agent_text_message(f"Error parsing or calculating RMSE: {e}")
                )

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            await updater.failed(new_agent_text_message(f"Evaluation failed: {e}"))
        finally:
            self.messenger.reset()
