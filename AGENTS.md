# Development instructions

The owner explicitly requested automatic GitHub pushes every time app source changes are saved. This is standing authorization to commit and push these changes and publish passing builds without asking again. Do not wait for a completed task or choose release milestones.

The owner's development PC runs source_sync.py in the background. It pushes saved source changes to main after a short file-write settling interval. GitHub tests main, assigns the release version, builds the Windows packages, and publishes the update feed. Newer pushes supersede older builds.

Let the watcher handle commits and pushes. Do not manually stage files, bump release numbers, or create release tags during normal work. Check the watcher status and the latest GitHub build before reporting publication complete. If automatic sync is stopped, restore it or use its one-shot mode after inspecting the changes. Do not enable source syncing on a friend's installed app.

Keep README.md completely free of links, including repository, file, and workflow links. The owner's explicit preference applies even though links are generally supported by GitHub.

Never publish local models, runtimes, chats, project workspaces, logs, credentials, downloaded tools, or built archives to the source repository. Keep those generated and personal folders ignored. The source watcher has its own public-source allowlist; add new public source paths deliberately when necessary.

Run checks appropriate to the change. The watcher pushes saved changes automatically; failed GitHub tests must block package publication. Do not disable tests to force a release. Preserve local inference through Ollama and independence from Codex accounts and API keys.
