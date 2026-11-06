# Local Agent

A Windows desktop assistant that runs its AI model locally through Ollama. No Codex subscription or OpenAI API key is required.

## Getting started

Extract the complete Windows release into a writable folder and open Start Local Agent.cmd. First-time installs offer a Download model button. Existing offline installations can keep their downloaded model.

Choose a project folder and describe a task. Enter sends a message; Shift+Enter adds a new line. Messages sent while the agent works are queued and incorporated at the next safe step. Stop cancels the task and clears unsent follow-ups.

## Working with the agent

The agent can inspect files, make edits, search a project, and run approved PowerShell commands. Commands run with your Windows user permissions. File-tool overwrites keep local backups. Structured questions offer choices and a custom answer when the agent needs important missing information.

The working timer shows elapsed time. Token speed is the actual generation rate reported by Ollama for the latest completed model step. Reason more enables the model's extra reasoning mode and can take longer. Plans show progress for larger tasks.

The default model is Qwen3.5 4B with a 16,384-token active context. When working memory becomes full, the app condenses older steps into notes and continues. The full chat stays saved locally, and the agent can retrieve earlier details when needed. Summaries are not unlimited memory and can omit details; project files remain available to reread. A small local model may need corrections and does not guarantee the reasoning quality of larger hosted models.

## Python projects

Use a project virtual environment for dependencies. The agent can create a .venv folder and then use python -m pip with the same interpreter that runs the project. It must not install packages into the app runtime or its Python Lib folder. Repeated command errors trigger a pause so the approach can be reconsidered.

## Updates

The Update button checks the configured GitHub release channel. It can also install a small update ZIP received from the app's owner. Updates preserve downloaded models, chats, settings, and workspaces. Close other copies of the app before applying an update.

A version-one installation can use Install Update.cmd from the update package once to receive the new Update button.

The repository includes automated tests and release packaging. Pushing a version tag runs tests, builds a Windows first-install package and a small app update, and publishes them together. The model is downloaded separately on first use, so app releases do not repeatedly transfer model weights.

To publish another version, update the version number, commit and push the source changes, then push a matching version tag. The Update button receives published releases; editing local source files alone does not update another computer.

## Requirements and privacy

Use 64-bit Windows 10 version 22H2 or Windows 11. 16 GB RAM is suggested. An NVIDIA GPU can improve speed; a CPU-only launcher is included. Other project toolchains such as Node.js, Git, Java, and compilers may need separate installation.

Local inference and file work can run offline after the model download. Update checks, model downloads, and commands that contact online services need internet. Personal chats, workspaces, logs, credentials, and downloaded model files are excluded from the source repository and app update packages.

Keep third-party notices with distributed copies. Source component terms are included in the distribution-notices directory.
