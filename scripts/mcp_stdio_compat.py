"""Run the existing MCP SDK over stdio with a periodic event-loop wakeup.

Some restricted Linux worker environments stall SDK stdio without scheduled
timers. This preserves their filesystem/network policy and MCP implementations.
Keep the wakeup alive through executor shutdown as well as protocol handling.
"""
import asyncio
from contextlib import suppress


async def _wakeup():
    while True:
        await asyncio.sleep(.02)


def run_stdio_compatible(server):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    pulse = loop.create_task(_wakeup())
    try:
        loop.run_until_complete(server.run_stdio_async())
    finally:
        pending = [task for task in asyncio.all_tasks(loop) if task is not pulse]
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.run_until_complete(loop.shutdown_default_executor())
        pulse.cancel()
        with suppress(asyncio.CancelledError):
            loop.run_until_complete(pulse)
        asyncio.set_event_loop(None)
        loop.close()
