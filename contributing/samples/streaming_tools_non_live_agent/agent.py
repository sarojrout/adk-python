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

"""Example agent demonstrating streaming tools in non-live mode (run_async/SSE).

This agent shows how to use streaming tools that yield intermediate results
in non-live mode. Streaming tools work with both run_async and SSE endpoints.
"""

from __future__ import annotations

import asyncio
from typing import AsyncGenerator

from google.adk.agents import Agent


async def monitor_stock_price(symbol: str) -> AsyncGenerator[dict, None]:
  """Monitor stock price with real-time updates.

  This is a streaming tool that yields intermediate results as the stock
  price changes. The agent can react to these intermediate results.

  Args:
    symbol: The stock symbol to monitor (e.g., 'AAPL', 'GOOGL').

  Yields:
    Dictionary containing stock price updates with status indicators.
  """
  # Simulate stock price changes
  prices = [100, 105, 110, 108, 112, 115]
  for i, price in enumerate(prices):
    await asyncio.sleep(1)  # Simulate real-time updates
    yield {
        'symbol': symbol,
        'price': price,
        'update': i + 1,
        'status': 'streaming' if i < len(prices) - 1 else 'complete',
    }


async def process_large_dataset(file_path: str) -> AsyncGenerator[dict, None]:
  """Process dataset with progress updates.

  This streaming tool demonstrates how to provide progress feedback
  for long-running operations.

  Args:
    file_path: Path to the dataset file to process.

  Yields:
    Dictionary containing progress information and final result.
  """
  total_rows = 100
  processed = 0

  # Simulate processing in batches
  for batch in range(10):
    await asyncio.sleep(0.5)  # Simulate processing time
    processed += 10
    yield {
        'progress': processed / total_rows,
        'processed': processed,
        'total': total_rows,
        'status': 'streaming',
        'message': f'Processed {processed}/{total_rows} rows',
    }

  # Final result
  yield {
      'result': 'Processing complete',
      'status': 'complete',
      'file_path': file_path,
      'total_processed': total_rows,
  }


async def monitor_system_health() -> AsyncGenerator[dict, None]:
  """Monitor system health metrics with continuous updates.

  This streaming tool demonstrates continuous monitoring that can be
  stopped by the agent when thresholds are reached.

  Yields:
    Dictionary containing system health metrics.
  """
  metrics = [
      {'cpu': 45, 'memory': 60, 'disk': 70},
      {'cpu': 50, 'memory': 65, 'disk': 72},
      {'cpu': 55, 'memory': 70, 'disk': 75},
      {'cpu': 60, 'memory': 75, 'disk': 78},
  ]

  for i, metric in enumerate(metrics):
    await asyncio.sleep(2)  # Check every 2 seconds
    yield {
        'metrics': metric,
        'timestamp': i + 1,
        'status': 'streaming' if i < len(metrics) - 1 else 'complete',
        'alert': 'high' if metric['cpu'] > 55 else 'normal',
    }


root_agent = Agent(
    name='streaming_tools_agent',
    model='gemini-2.5-flash-lite',
    instruction=(
        'You are a helpful assistant that can monitor stock prices, process'
        ' datasets, and monitor system health using streaming tools. When'
        ' using streaming tools, you will receive intermediate results that'
        ' you can react to. For example, if monitoring stock prices, you can'
        ' alert the user when prices change significantly. If processing a'
        ' dataset, you can provide progress updates. If monitoring system'
        ' health, you can alert when metrics exceed thresholds.'
    ),
    tools=[monitor_stock_price, process_large_dataset, monitor_system_health],
)
