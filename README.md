# A2A Green Agent

This is a green agent based on the minimal template for building [A2A (Agent-to-Agent)](https://a2a-protocol.org/latest/) green agents compatible with the [AgentBeats](https://agentbeats.dev) platform.

## Motivation

This green agent aims to test agents' ability to model time series whose underlying structure is empirically discoverable.

Modelling time series data is a very common problem in scientific and business applications. There is a lot of theory available to address the problem and there are many sophisticated libraries that could provide the necessary tools. It remains to be seen, however, how well agents perform on this task.

## Metric of success

The metric of success for this green agent is the Root Mean Square Error (RMSE) of the forecast, whereby a smaller RMSE indicates better performance.

## Project Structure

```
src/
├─ server.py      # Server setup and agent card configuration
├─ executor.py    # A2A request handling
├─ agent.py       # Your agent implementation goes here
└─ messenger.py   # A2A messaging utilities
tests/
└─ test_agent.py  # Agent tests
Dockerfile        # Docker configuration
pyproject.toml    # Python dependencies
.github/
└─ workflows/
   └─ test-and-publish.yml # CI workflow
```
