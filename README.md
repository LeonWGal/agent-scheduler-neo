# Agent Scheduler Neo

Queue `txt2img` and `img2img` jobs in Forge Neo, capture live UI state on `Enqueue`, and monitor task progress, history, and results from the scheduler interface.

Russian: [README.ru-RU.md](./README.ru-RU.md)

## Features

- Adds a queue for `txt2img` and `img2img` tasks, plus task history and API endpoints.
- Uses an Enqueue bridge for Gradio 4: click `Enqueue`, collect live UI arguments, then send them to `/agent-scheduler/v1/queue/ui`.
- Captures or injects runtime data for features such as ControlNet, ADetailer, and TIPO so queued jobs reflect the current UI state.
- Extracts generation info from Forge Neo results and uses PNG metadata as a fallback.
- Applies NeverOOM / inference-tensor compatibility inside the extension through `agent_scheduler/forge_neveroom_compat.py`.
- Shows progress and generated images for the current queued task.
- Persists queue UI grid state in `localStorage`.
- Supports an Enqueue hotkey and queue-related UI actions.

## Installation

1. Place this folder in Forge Neo `extensions/`.
2. Restart the WebUI completely after installing or updating the extension.
3. On first launch, `install.py` installs `sqlalchemy` if it is missing.

## Usage

1. Configure prompts and related scripts such as ControlNet, ADetailer, or TIPO as usual.
2. Click `Enqueue` instead of `Generate` to add the current setup to the queue.
3. Open the Agent Scheduler UI or tab to monitor queue state, progress, and results.
4. Optionally configure and use the Enqueue hotkey from Settings.

## Notes

- A full WebUI restart is required after install or update. Reloading the UI alone is not enough.
- On startup, the log should include `[AgentScheduler] NeverOOM/inference-tensor compat patched`.
- The queue database is stored in `task_scheduler.sqlite3` unless overridden with `--agent-scheduler-sqlite-file`.
- If live script state is missing from a queued task, the extension includes DOM and session fallbacks for some integrations.
- This fork targets Forge Neo / Gradio 4 behavior.

## Development

The backend entry point is `scripts/task_scheduler.py`. Frontend bridge logic lives in `javascript/agent-scheduler.enqueue-bridge.js`, and extension-specific compatibility code lives in `agent_scheduler/`.
