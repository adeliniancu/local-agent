# Local Agent

A Windows desktop assistant that runs its AI model locally through Ollama. No Codex subscription or OpenAI API key is required.

## Getting started

Extract the complete Windows release into a writable folder and open Start Local Agent.cmd. First-time installs offer a Download model button. Existing offline installations can keep their downloaded model.

Choose a project folder and describe a task. Enter sends a message; Shift+Enter adds a new line. Messages sent while the agent works are queued and incorporated at the next safe step. Stop cancels the task and clears unsent follow-ups.

## Working with the agent

The agent can inspect files, make edits, search a project, and run approved PowerShell commands. Commands run with your Windows user permissions. File-tool overwrites keep local backups. Questions and command approvals stay pinned above the message box, including in smaller windows. Their answer buttons remain visible while the question or command body scrolls. Questions offer choices and a custom answer. Don't ask again in this chat (questions) lets the AI decide routine implementation details; it does not approve commands or invent missing facts. This choice resets with a new or reopened chat. Command cards offer Allow once and Reject; the visible Always allow commands in this chat checkbox controls repeated approvals for the current chat.

The working timer shows elapsed time. Token speed is the actual generation rate reported by Ollama for the latest completed model step. Reason more enables the model's extra reasoning mode and can take longer. Plans show progress for larger tasks.

Attach files beside the message box, or paste an image or copied files with Ctrl+V. Ordinary text paste still works; right-click menus provide copy and paste. Attached files are copied into private local storage, so saved chats can still use them after the original moves. Remove a pending attachment before sending if it is not needed. Attachments can accompany follow-up messages while the agent works.

Supported inputs are UTF-8/UTF-16 text and code, CSV, JSON, Markdown, DOCX text, and PNG/JPEG/GIF/BMP images. Each file can be up to 25 MB; a message can contain eight attachments and up to four images at 16K context, or one image at 8K. Images require a vision-capable model. Large text files start with an excerpt; the agent can read additional exact sections. DOCX layout and embedded images are not extracted, animated images use one frame, and PDF/binary formats show an explanation instead of pretending to read them. Images are resized for local inference. Image generation is not included.

For coding, the agent is instructed to inspect the relevant files, make focused edits, and check the result. Its exact-text edit tool rejects ambiguous matches, and existing-file writes require a current read so stale assumptions do not overwrite newer file changes. These checks reduce avoidable mistakes but do not make a small model equivalent to a larger model.

The default model is Qwen3.5 4B with a 16,384-token active context. When working memory becomes full, the app condenses older steps into notes and continues. The full chat stays saved locally, and the agent can retrieve earlier details when needed. Summaries are not unlimited memory and can omit details; project files remain available to reread. A small local model may need corrections and does not guarantee the reasoning quality of larger hosted models.

Productive tasks continue automatically beyond 24 model steps. Repeated work without new progress triggers a normal pause. A longer task budget also provides a stopping point while preserving completed work and chat history; another message can continue the task.

## Models and performance

The inline model and performance controls offer Auto, RTX 5060 desktop, RTX 4070 laptop, RTX 5080 desktop, and CPU profiles. GPU allocation uses detected hardware, available memory, and the selected model. The default active context remains 16,384 tokens when the model supports it; smaller supported model windows receive matching working-memory budgets. Profiles do not change GPU firmware or power settings, and token speed always shows measured performance.

Download another local Ollama model by name, or import a local GGUF file with a chosen model name. Downloads and imports start only when requested. Models need tool support and at least an 8,192-token context for file and command tasks; the app checks advertised capabilities before starting that work. Larger models need more memory and can run more slowly, even on a faster GPU.

The model panel includes presets for the existing Qwen3.5 4B model (about 3.3 GB) and Qwen3.5 9B (about 6.6 GB). The larger model is an optional quality trial and can be slower on an 8 GB GPU; choosing the preset does not download it until Download is pressed. Actual task quality depends on the model, quantization, reasoning settings, and validation. A faster GPU changes speed, not the model's knowledge. A coding-specialized model may lack vision or extra thinking even when it supports file and command tools.

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

Local inference and file work can run offline after the model download. Update checks, model downloads, and commands that contact online services need internet. Personal chats, attachments, workspaces, logs, credentials, and downloaded model files are excluded from the source repository and app update packages.

Keep third-party notices with distributed copies. Source component terms are included in the distribution-notices directory.
