# Local Agent

A Windows desktop assistant that runs its AI model locally through Ollama. No Codex subscription or OpenAI API key is required.

## Getting started

Extract the complete Windows release into a writable folder and open Start Local Agent.cmd. First-time installs offer a Download model button. Existing offline installations can keep their downloaded model.

Choose a project folder and describe a task. Enter sends a message; Shift+Enter adds a new line. Messages sent while the agent works are queued and incorporated at the next safe step. Stop cancels the task and clears unsent follow-ups.

## Working with the agent

The agent can inspect files, make edits, search a project, and run approved PowerShell commands. Commands run with your Windows user permissions. File-tool overwrites keep local backups. Questions and command approvals appear inside the chat. Questions offer choices and a custom answer. Command cards offer Allow once and Reject; the visible Always allow commands in this chat checkbox controls repeated approvals for the current chat.

The working timer shows elapsed time. Token speed is the actual generation rate reported by Ollama for the latest completed model step. Reason more enables the model's extra reasoning mode and can take longer. Plans show progress for larger tasks.

The default model is Qwen3.5 4B with a 16,384-token active context. When working memory becomes full, the app condenses older steps into notes and continues. The full chat stays saved locally, and the agent can retrieve earlier details when needed. Summaries are not unlimited memory and can omit details; project files remain available to reread. A small local model may need corrections and does not guarantee the reasoning quality of larger hosted models.

Productive tasks continue automatically beyond 24 model steps. Repeated work without new progress triggers a normal pause. A longer task budget also provides a stopping point while preserving completed work and chat history; another message can continue the task.

## Models and performance

The inline model and performance controls offer Auto, RTX 5060 desktop, RTX 4070 laptop, RTX 5080 desktop, and CPU profiles. GPU allocation uses detected hardware, available memory, and the selected model. The default active context remains 16,384 tokens when the model supports it; smaller supported model windows receive matching working-memory budgets. Profiles do not change GPU firmware or power settings, and token speed always shows measured performance.

Download another local Ollama model by name, or import a local GGUF file with a chosen model name. Downloads and imports start only when requested. Models need tool support and at least an 8,192-token context for file and command tasks; the app checks advertised capabilities before starting that work. Larger models need more memory and can run more slowly, even on a faster GPU.

## Python projects

Use a project virtual environment for dependencies. The agent can create a .venv folder and then use python -m pip with the same interpreter that runs the project. It must not install packages into the app runtime or its Python Lib folder. Repeated command errors trigger a pause so the approach can be reconsidered.

## Updates

The Update button checks the configured GitHub release channel. It can also install a small update ZIP received from the app's owner. Updates preserve downloaded models, chats, settings, and workspaces. Close other copies of the app before applying an update.

A version-one installation can use Install Update.cmd from the update package once to receive the new Update button.

The owner's development PC automatically commits and pushes saved app source changes to main. Its background source watcher starts with Windows and waits briefly for file writes to settle. There is no manual release step after editing. It retries when the internet connection returns and records its status locally.

Each push starts GitHub tests and packaging. GitHub assigns the release version and publishes the Windows first-install package, small app update, and update feed together after checks pass. A newer push replaces an unfinished older build. Failed checks keep the previous published release available.

The model is downloaded separately on first use, so app releases do not repeatedly transfer model weights. Installed release-version metadata is ignored by the source watcher to prevent an app update from triggering another release. The watcher runs only on the owner's development PC; it is not part of the app distributed to friends.

## Requirements and privacy

Use 64-bit Windows 10 version 22H2 or Windows 11. 16 GB RAM is suggested. An NVIDIA GPU can improve speed; a CPU-only launcher is included. Other project toolchains such as Node.js, Git, Java, and compilers may need separate installation.

Local inference and file work can run offline after the model download. Update checks, model downloads, and commands that contact online services need internet. Personal chats, workspaces, logs, credentials, and downloaded model files are excluded from the source repository and app update packages.

Keep third-party notices with distributed copies. Source component terms are included in the distribution-notices directory.
