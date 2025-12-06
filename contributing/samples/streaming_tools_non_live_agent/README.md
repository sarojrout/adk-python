# Streaming Tools Non-Live Agent

This agent demonstrates streaming tools in non-live mode (run_async/SSE).

## Features

- **monitor_stock_price**: Monitors stock prices with real-time updates
- **process_large_dataset**: Processes datasets with progress updates
- **monitor_system_health**: Monitors system health metrics continuously

## Testing

### With ADK Web UI

```bash
cd contributing/samples
adk web .
```

Then try:
- "Monitor the stock price for AAPL"
- "Process a large dataset at /tmp/data.csv"
- "Monitor system health"

### With ADK CLI

```bash
cd contributing/samples/streaming_tools_non_live_agent
adk run .
```

### With API Server (SSE)

```bash
cd contributing/samples
adk api_server .
```

Then send a POST request to `/run_sse` with `streaming: true` to see intermediate Events.
