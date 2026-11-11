"""Local Agent: a standalone, local-first desktop coding assistant."""
from __future__ import annotations

import copy
from datetime import datetime
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter import font as tkfont

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
STATE = DATA / "last-chat.json"
DEFAULT_MODEL = "qwen3.5:4b-q4_K_M"
CODING_V2_MODEL = "qwen3.5:4b-coding-v2"
PROFILE_LABELS = {
    "auto": "Auto", "rtx5060": "RTX 5060 desktop", "rtx4070_laptop": "RTX 4070 laptop",
    "rtx5080": "RTX 5080 desktop", "cpu": "CPU",
}
CODING_MODELS = {
    "Fast / smaller · Qwen 3.5 4B": (DEFAULT_MODEL, "About 3.3 GB. The default for this app."),
    "Coding + images · Qwen 3.5 9B": ("qwen3.5:9b-q4_K_M", "About 6.6 GB. May be slower or use system RAM on an 8 GB GPU."),
}
try:
    from version import __version__ as APP_VERSION
except ImportError:
    APP_VERSION = "2.0.0"
COLORS = {
    "bg": "#101619", "side": "#0b1114", "panel": "#182126",
    "border": "#2a373d", "text": "#e8eeee", "muted": "#95a8ad",
    "mint": "#9bdfc4", "mint_dark": "#254c40", "user": "#a9c9ef",
    "error": "#f1a69e", "tool": "#c0b2df",
}


class LocalAgentApp:
    def __init__(self, root: tk.Tk, *, start_engine: bool = True):
        self.root = root
        self.root.title("Local Agent")
        self.root.geometry("1120x790")
        self.root.minsize(840, 620)
        self.root.configure(bg=COLORS["bg"])
        self.events: queue.Queue = queue.Queue()
        self.closing = False
        self.busy = False
        self.ready = False
        self.connecting = False
        self.background_enabled = start_engine
        self.downloading = False
        self.installing_coding_v2 = False
        self.coding_v2_importing = False
        self.coding_v2_cancel = threading.Event()
        self.coding_v2_worker = None
        self.preserve_model_selection_on_refresh = False
        self.updating = False
        self.cancel = threading.Event()
        self.allow_commands = threading.Event()
        self.incoming: queue.Queue = queue.Queue()
        self.pending_attachments: list[dict] = []
        self.attachment_jobs = 0
        self.attachment_epoch = 0
        self.pending_previews = []
        self.chat_previews = []
        self.queued_count = 0
        self.work_started: float | None = None
        self.elapsed_seconds = 0.0
        self.last_tokens_per_second: float | None = None
        self.reason_more = False
        self.performance_profile = "auto"
        self.settings_expanded = False
        self.available_models: list[str] | None = None
        self.model_tool_support: bool | None = None
        self.model_thinking_support: bool | None = None
        self.model_context_support: bool | None = None
        self.model_vision_support: bool | None = None
        self.model_context_tokens: int | None = None
        self.model_info_name = ""
        self.model_checking = False
        self.model_check_serial = 0
        self.hardware_checking = False
        self.import_path: str | None = None
        self.pending_model_selection: str | None = None
        self.model_operation_label = "Downloading model"
        self.memory_state: dict = {}
        self.history: list[dict] = []
        self.agent = None
        self.worker = None
        self.run_id = 0
        self.stream_started = False
        self.stream_text = ""
        self.approval_window: tk.Frame | None = None
        self.approval_request = None
        self.question_window: tk.Frame | None = None
        self.question_request = None
        self.prompt_panel = None
        self.prompt_body = None
        self.inline_cards: list[tk.Frame] = []
        self.stream_mark: str | None = None
        self.stream_start_mark: str | None = None
        self.stream_end_mark: str | None = None
        self.workspace = ROOT / "workspace"
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.model_name = DEFAULT_MODEL
        self._load_state()
        self._style()
        self._set_icon()
        self._build()
        self._render_history()
        if not self.workspace.is_dir():
            self.status_label.configure(text="Your previous folder is unavailable. Choose a folder before sending a task.")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<Control-Return>", self._send_key)
        self.root.bind("<Escape>", lambda _event: self.stop() if self.busy else None)
        self._poll_id = self.root.after(50, self._poll)
        self._tick_id = self.root.after(250, self._tick)
        if start_engine:
            self.root.after(100, self.connect)

    def _set_icon(self):
        self.icon_photo = None
        self.brand_photo = None
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("LocalAgent.Desktop")
        except (AttributeError, OSError):
            pass
        try:
            icon = ROOT / "assets" / "app.ico"
            if icon.is_file():
                self.root.iconbitmap(str(icon))
            image = ROOT / "assets" / "app.png"
            if image.is_file():
                self.icon_photo = tk.PhotoImage(file=str(image))
                self.root.iconphoto(True, self.icon_photo)
            logo = ROOT / "assets" / "logo.png"
            if logo.is_file():
                photo = tk.PhotoImage(file=str(logo))
                factor = max(1, math.ceil(max(photo.width(), photo.height()) / 40))
                self.brand_photo = photo.subsample(factor)
        except tk.TclError:
            pass

    def _style(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TCombobox", fieldbackground=COLORS["panel"],
                        background=COLORS["panel"], foreground=COLORS["text"],
                        arrowcolor=COLORS["muted"], bordercolor=COLORS["border"],
                        lightcolor=COLORS["border"], darkcolor=COLORS["border"], padding=7)
        style.map("TCombobox", fieldbackground=[("disabled", COLORS["side"])],
                  foreground=[("disabled", COLORS["muted"])])
        self.root.option_add("*TCombobox*Listbox.background", COLORS["panel"])
        self.root.option_add("*TCombobox*Listbox.foreground", COLORS["text"])
        style.configure("Vertical.TScrollbar", background=COLORS["border"],
                        troughcolor=COLORS["bg"], borderwidth=0, arrowsize=12)

    def _label(self, parent, text, *, size=10, color=None, bold=False, bg=None, **kwargs):
        return tk.Label(parent, text=text, bg=bg or parent.cget("bg"),
                        fg=color or COLORS["text"], font=("Segoe UI", size, "bold" if bold else "normal"),
                        **kwargs)

    def _button(self, parent, text, command, *, primary=False, **kwargs):
        return tk.Button(parent, text=text, command=command, relief="flat", bd=0,
                         bg=COLORS["mint"] if primary else COLORS["panel"],
                         fg=COLORS["side"] if primary else COLORS["text"],
                         activebackground="#b8ebd6" if primary else COLORS["border"],
                         activeforeground=COLORS["side"] if primary else COLORS["text"],
                         disabledforeground="#66777d", cursor="hand2", padx=14, pady=10,
                         font=("Segoe UI", 10, "bold" if primary else "normal"), **kwargs)

    def _build(self):
        self.sidebar = tk.Frame(self.root, bg=COLORS["side"], width=238)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)
        # Reserve the footer first so it remains reachable at the minimum size.
        bottom = tk.Frame(self.sidebar, bg=COLORS["side"])
        bottom.pack(side="bottom", fill="x", padx=20, pady=(8, 12))
        self.engine_label = self._label(bottom, "●  Starting local engine", size=9, color=COLORS["muted"], wraplength=190, justify="left")
        self.engine_label.pack(anchor="w", pady=(0, 6))
        footer_actions = tk.Frame(bottom, bg=COLORS["side"])
        footer_actions.pack(fill="x")
        self.retry_button = self._button(footer_actions, "Refresh", self.connect)
        self.retry_button.configure(pady=7, padx=10)
        self.retry_button.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.update_button = self._button(footer_actions, "Update", self.open_update)
        self.update_button.configure(pady=7, padx=10)
        self.update_button.pack(side="left", fill="x", expand=True)
        self._label(bottom, f"v{APP_VERSION} · Saved on this PC", size=8, color=COLORS["muted"]).pack(anchor="w", pady=(10, 0))
        brand = tk.Frame(self.sidebar, bg=COLORS["side"])
        brand.pack(fill="x", padx=20, pady=(20, 16))
        if self.brand_photo is not None:
            tk.Label(brand, image=self.brand_photo, bg=COLORS["side"]).pack(side="left", padx=(0, 9))
        else:
            self._label(brand, "◈", size=25, color=COLORS["mint"]).pack(side="left", padx=(0, 9))
        brand_text = tk.Frame(brand, bg=COLORS["side"])
        brand_text.pack(side="left")
        self._label(brand_text, "Local Agent", size=16, bold=True).pack(anchor="w")
        self._label(brand_text, "Your computer. Your assistant.", size=8, color=COLORS["muted"]).pack(anchor="w")
        self.new_button = self._button(self.sidebar, "+  New chat", self.new_chat)
        self.new_button.configure(pady=7)
        self.new_button.pack(fill="x", padx=20)
        self.saved_button = self._button(self.sidebar, "Saved chats", self.open_saved_chat)
        self.saved_button.configure(pady=7)
        self.saved_button.pack(fill="x", padx=20, pady=(6, 0))
        self._label(self.sidebar, "WORKSPACE", size=8, bold=True, color=COLORS["muted"]).pack(anchor="w", padx=22, pady=(20, 7))
        self.folder_title = self._label(self.sidebar, self._short_name(self.workspace), size=11, bold=True,
                                        anchor="w", wraplength=190, justify="left")
        self.folder_title.pack(fill="x", padx=22)
        self.folder_path = self._label(self.sidebar, self._short_path(self.workspace), size=9, color=COLORS["muted"],
                                       anchor="w", wraplength=190, justify="left")
        self.folder_path.pack(fill="x", padx=22, pady=(5, 9))
        self.folder_button = self._button(self.sidebar, "Choose folder", self.choose_folder)
        self.folder_button.configure(pady=7)
        self.folder_button.pack(fill="x", padx=20)
        self.open_button = self._button(self.sidebar, "Open folder ↗", self.open_folder)
        self.open_button.configure(pady=7)
        self.open_button.pack(fill="x", padx=20, pady=(6, 0))
        self._label(self.sidebar, "MODEL", size=8, bold=True, color=COLORS["muted"]).pack(anchor="w", padx=22, pady=(18, 7))
        self.model_var = tk.StringVar(value=self.model_name)
        self.model_box = ttk.Combobox(self.sidebar, textvariable=self.model_var, font=("Segoe UI", 9))
        self.model_box.pack(fill="x", padx=20)
        self.model_box.bind("<<ComboboxSelected>>", self._model_changed)
        self.model_box.bind("<FocusOut>", self._model_changed)
        self.think_var = tk.BooleanVar(value=self.reason_more)
        self.think_box = tk.Checkbutton(self.sidebar, text="Reason more", variable=self.think_var,
                       command=self._save_state, bg=COLORS["side"], fg=COLORS["text"],
                       selectcolor=COLORS["panel"], activebackground=COLORS["side"],
                       activeforeground=COLORS["text"], font=("Segoe UI", 9), bd=0)
        self.think_box.pack(anchor="w", padx=19, pady=(8, 0))

        main = tk.Frame(self.root, bg=COLORS["bg"])
        main.pack(side="left", fill="both", expand=True)
        # Reserve the composer before the expandable chat/settings area so it
        # remains reachable even at the minimum window size.
        self.footer_frame = tk.Frame(main, bg=COLORS["bg"])
        self.footer_frame.pack(side="bottom", fill="x")
        header = tk.Frame(main, bg=COLORS["bg"])
        header.pack(fill="x", padx=30, pady=(27, 18))
        self._label(header, "Let's get to work.", size=19, bold=True).pack(side="left")
        self.settings_toggle = self._button(header, "Models & performance ▾", self.toggle_settings)
        self.settings_toggle.configure(pady=6, padx=9, font=("Segoe UI", 9))
        self.settings_toggle.pack(side="right")
        tk.Frame(main, bg=COLORS["border"], height=1).pack(fill="x", padx=30)

        self.settings_container = tk.Frame(main, bg=COLORS["panel"])
        self.settings_canvas = tk.Canvas(self.settings_container, height=224, highlightthickness=0, bg=COLORS["panel"])
        settings_scrollbar = ttk.Scrollbar(self.settings_container, orient="vertical", command=self.settings_canvas.yview)
        self.settings_canvas.configure(yscrollcommand=settings_scrollbar.set)
        settings_scrollbar.pack(side="right", fill="y")
        self.settings_canvas.pack(side="left", fill="both", expand=True)
        self.settings_panel = tk.Frame(self.settings_canvas, bg=COLORS["panel"], padx=12, pady=10)
        settings_window = self.settings_canvas.create_window(0, 0, window=self.settings_panel, anchor="nw")
        self.settings_canvas.bind("<Configure>", lambda event: self.settings_canvas.itemconfigure(settings_window, width=event.width))
        self.settings_panel.bind("<Configure>", lambda _event: self.settings_canvas.configure(scrollregion=self.settings_canvas.bbox("all")))
        self.settings_panel.grid_columnconfigure(1, weight=1)
        self._label(self.settings_panel, "Performance", size=9).grid(row=0, column=0, sticky="w", padx=(0, 9))
        self.profile_var = tk.StringVar(value=PROFILE_LABELS[self.performance_profile])
        self.profile_box = ttk.Combobox(self.settings_panel, textvariable=self.profile_var, values=list(PROFILE_LABELS.values()),
                                      state="readonly", font=("Segoe UI", 9), width=24)
        self.profile_box.grid(row=0, column=1, columnspan=2, sticky="ew", padx=(0, 8))
        self.profile_box.bind("<<ComboboxSelected>>", self._profile_changed)
        self._label(self.settings_panel, "Next task", size=8, color=COLORS["muted"]).grid(row=0, column=3, sticky="e")
        self.hardware_label = self._label(self.settings_panel, "Hardware information will appear here.", size=8,
                                         color=COLORS["muted"], wraplength=500, justify="left", anchor="w")
        self.hardware_label.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(7, 8))
        self._label(self.settings_panel, "Download model", size=9).grid(row=2, column=0, sticky="w", padx=(0, 9))
        self.download_name_var = tk.StringVar(value=DEFAULT_MODEL)
        entry_style = dict(bg=COLORS["side"], fg=COLORS["text"], insertbackground=COLORS["mint"],
                           relief="flat", font=("Segoe UI", 9), highlightthickness=1,
                           highlightbackground=COLORS["border"], highlightcolor=COLORS["mint"])
        self.download_name_entry = tk.Entry(self.settings_panel, textvariable=self.download_name_var, **entry_style)
        self.download_name_entry.grid(row=2, column=1, columnspan=2, sticky="ew", ipady=6, padx=(0, 8))
        self.custom_download_button = self._button(self.settings_panel, "Download", self.download_custom_model)
        self.custom_download_button.configure(padx=10, pady=6, font=("Segoe UI", 9))
        self.custom_download_button.grid(row=2, column=3, sticky="ew")
        self._label(self.settings_panel, "Import GGUF", size=9).grid(row=3, column=0, sticky="w", padx=(0, 9), pady=(8, 0))
        self.import_name_var = tk.StringVar(value="my-model")
        self.import_name_entry = tk.Entry(self.settings_panel, textvariable=self.import_name_var, **entry_style)
        self.import_name_entry.grid(row=3, column=1, sticky="ew", ipady=6, padx=(0, 8), pady=(8, 0))
        self.choose_gguf_button = self._button(self.settings_panel, "Choose file", self.choose_gguf)
        self.choose_gguf_button.configure(padx=9, pady=6, font=("Segoe UI", 9))
        self.choose_gguf_button.grid(row=3, column=2, padx=(0, 8), pady=(8, 0))
        self.import_button = self._button(self.settings_panel, "Import", self.import_model, state="disabled")
        self.import_button.configure(padx=10, pady=6, font=("Segoe UI", 9))
        self.import_button.grid(row=3, column=3, sticky="ew", pady=(8, 0))
        self.gguf_label = self._label(self.settings_panel, "Choose a local .gguf file and give it a model name.", size=8,
                                     color=COLORS["muted"], wraplength=500, justify="left", anchor="w")
        self.gguf_label.grid(row=4, column=0, columnspan=4, sticky="ew", pady=(6, 0))
        self.model_capability_label = self._label(self.settings_panel, "Tool support: not checked yet.", size=8,
                                                 color=COLORS["muted"], wraplength=500, justify="left", anchor="w")
        self.model_capability_label.grid(row=5, column=0, columnspan=4, sticky="ew", pady=(7, 0))
        self._label(self.settings_panel, "Coding presets", size=9).grid(row=6, column=0, sticky="w", padx=(0, 9), pady=(8, 0))
        self.coding_preset_var = tk.StringVar(value=next(iter(CODING_MODELS)))
        self.coding_preset_box = ttk.Combobox(self.settings_panel, textvariable=self.coding_preset_var,
                                             values=list(CODING_MODELS), state="readonly", font=("Segoe UI", 9))
        self.coding_preset_box.grid(row=6, column=1, columnspan=3, sticky="ew", pady=(8, 0))
        self.coding_preset_box.bind("<<ComboboxSelected>>", self._coding_preset_changed)
        self.coding_preset_note = self._label(self.settings_panel, "Choose a preset to fill the download name. Download starts only when you click Download.",
                                             size=8, color=COLORS["muted"], wraplength=500, justify="left", anchor="w")
        self.coding_preset_note.grid(row=7, column=0, columnspan=4, sticky="ew", pady=(5, 0))
        # Keep the trained model discoverable near the top of the existing
        # scrollable model controls, without changing the selected model.
        for child in self.settings_panel.winfo_children():
            info = child.grid_info()
            if info and int(info["row"]) >= 2:
                child.grid_configure(row=int(info["row"]) + 1)
        coding_v2 = tk.Frame(self.settings_panel, bg=COLORS["panel"])
        coding_v2.grid(row=2, column=0, columnspan=4, sticky="ew", pady=(0, 10))
        coding_v2_actions = tk.Frame(coding_v2, bg=COLORS["panel"])
        coding_v2_actions.pack(fill="x")
        self._label(coding_v2_actions, "Coding v2 · Experimental", size=9, bold=True).pack(side="left")
        self.install_coding_v2_button = self._button(coding_v2_actions, "Install coding v2", self.install_coding_v2)
        self.install_coding_v2_button.configure(padx=9, pady=6, font=("Segoe UI", 9))
        self.install_coding_v2_button.pack(side="right")
        self.cancel_coding_v2_button = self._button(coding_v2_actions, "Cancel", self.cancel_coding_v2_install, state="disabled")
        self.cancel_coding_v2_button.configure(padx=8, pady=6, font=("Segoe UI", 9))
        self.cancel_coding_v2_button.pack(side="right", padx=(6, 6))
        self._label(coding_v2, "Up to 3.4 GB download · 8 GB free storage required.\nExperimental. The original scored better in our coding tests.",
                    size=8, color=COLORS["muted"], wraplength=480, justify="left", anchor="w").pack(fill="x", pady=(5, 0))
        self.coding_v2_status = self._label(coding_v2, CODING_V2_MODEL, size=8, color=COLORS["muted"],
                                           wraplength=480, justify="left", anchor="w")
        self.coding_v2_status.pack(fill="x", pady=(4, 0))
        self.coding_v2_progress = ttk.Progressbar(coding_v2, mode="determinate", maximum=100)

        self.setup_bar = tk.Frame(main, bg=COLORS["panel"], padx=12, pady=10)
        self._label(self.setup_bar, "Download your local model\nonce to get started.", size=9,
                    color=COLORS["muted"], justify="left").pack(side="left")
        self.download_button = self._button(self.setup_bar, "Download model (3.3 GB)", self.download_model, primary=True)
        self.download_button.pack(side="right", padx=(8, 0))

        transcript_frame = tk.Frame(main, bg=COLORS["bg"])
        self.transcript_frame = transcript_frame
        transcript_frame.pack(fill="both", expand=True, padx=(27, 16), pady=(8, 0))
        self.transcript = tk.Text(transcript_frame, wrap="word", height=1, width=1, bd=0, highlightthickness=0,
                                  bg=COLORS["bg"], fg=COLORS["text"], insertbackground=COLORS["text"],
                                  font=("Segoe UI", 11), padx=3, pady=12, spacing1=3, spacing3=4,
                                  state="disabled", cursor="arrow", takefocus=True)
        scrollbar = ttk.Scrollbar(transcript_frame, orient="vertical", command=self.transcript.yview)
        self.transcript.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.transcript.pack(side="left", fill="both", expand=True)
        self.transcript.bind("<Configure>", self._resize_inline_cards)
        for name, opts in {
            "welcome_title": {"font": ("Segoe UI", 24, "bold"), "foreground": COLORS["text"], "spacing1": 35, "spacing3": 14},
            "muted": {"foreground": COLORS["muted"]},
            "user_role": {"foreground": COLORS["user"], "font": ("Segoe UI", 9, "bold"), "spacing1": 17},
            "agent_role": {"foreground": COLORS["mint"], "font": ("Segoe UI", 9, "bold"), "spacing1": 17},
            "tool_role": {"foreground": COLORS["tool"], "font": ("Segoe UI", 9, "bold"), "spacing1": 10},
            "tool": {"foreground": COLORS["muted"], "font": ("Consolas", 9), "lmargin1": 12, "lmargin2": 12},
            "error": {"foreground": COLORS["error"], "spacing1": 9},
            "body": {"foreground": COLORS["text"], "spacing3": 9},
        }.items():
            self.transcript.tag_configure(name, **opts)

        self.status_label = self._label(self.footer_frame, "Choose a folder, then tell your agent what to do.", size=9, color=COLORS["muted"], anchor="w")
        self.status_label.pack(fill="x", padx=31, pady=(5, 2))
        self.metrics_label = self._label(self.footer_frame, "Elapsed 00:00 · Last generation — tokens/s", size=8, color=COLORS["muted"], anchor="w")
        self.metrics_label.pack(fill="x", padx=31, pady=(0, 8))
        self.memory_label = self._label(self.footer_frame, "Working memory summarized · Full chat saved", size=8,
                                        color=COLORS["mint"], anchor="w")
        composer_border = tk.Frame(self.footer_frame, bg=COLORS["border"], padx=1, pady=1)
        self.composer_border = composer_border
        composer_border.pack(fill="x", padx=30)
        composer = tk.Frame(composer_border, bg=COLORS["panel"])
        composer.pack(fill="x")
        self.input = tk.Text(composer, height=3, wrap="word", bd=0, highlightthickness=0,
                             bg=COLORS["panel"], fg=COLORS["text"], insertbackground=COLORS["mint"],
                             font=("Segoe UI", 11), padx=13, pady=12, undo=True)
        self.input.pack(fill="x")
        self.input.bind("<Return>", self._send_key)
        self.input.bind("<Shift-Return>", self._newline_key)
        self.input.bind("<Control-Return>", self._send_key)
        self.input.bind("<Control-v>", self._paste_input)
        self.input.bind("<Control-V>", self._paste_input)
        self.input.bind("<<Paste>>", self._paste_input)
        self.input.bind("<Button-3>", lambda event: self._text_menu(event, self.input, editable=True))
        self.transcript.bind("<Button-3>", lambda event: self._text_menu(event, self.transcript))
        self.pending_frame = tk.Frame(composer, bg=COLORS["panel"])
        self.pending_canvas = tk.Canvas(self.pending_frame, height=38, highlightthickness=0, bg=COLORS["panel"])
        self.pending_canvas.pack(side="left", fill="x", expand=True)
        self.pending_chips = tk.Frame(self.pending_canvas, bg=COLORS["panel"])
        self.pending_canvas.create_window(0, 0, window=self.pending_chips, anchor="nw")
        self.pending_chips.bind("<Configure>", lambda _event: self.pending_canvas.configure(scrollregion=self.pending_canvas.bbox("all")))
        self.pending_canvas.bind("<MouseWheel>", lambda event: self.pending_canvas.xview_scroll(-1 if event.delta > 0 else 1, "units"))
        self.clear_attachments_button = self._button(self.pending_frame, "Clear", self.clear_attachments)
        self.clear_attachments_button.configure(pady=4, padx=8, font=("Segoe UI", 8))
        self.clear_attachments_button.pack(side="right", padx=(5, 0))
        self.always_allow_var = tk.BooleanVar(value=False)
        self.always_allow_box = tk.Checkbutton(composer, text="Always allow commands in this chat",
                         variable=self.always_allow_var, command=self._toggle_command_permission,
                         bg=COLORS["panel"], fg=COLORS["muted"], activebackground=COLORS["panel"],
                         activeforeground=COLORS["text"], selectcolor=COLORS["side"],
                         font=("Segoe UI", 9), bd=0)
        self.always_allow_box.pack(anchor="w", padx=10, pady=(0, 6))
        self.decide_questions_var = tk.BooleanVar(value=False)
        self.decide_questions_box = tk.Checkbutton(composer, text="Don't ask again in this chat (questions)",
                         variable=self.decide_questions_var, command=self._toggle_question_permission,
                         bg=COLORS["panel"], fg=COLORS["muted"], activebackground=COLORS["panel"],
                         activeforeground=COLORS["text"], selectcolor=COLORS["side"], font=("Segoe UI", 9), bd=0)
        self.decide_questions_box.pack(anchor="w", padx=10, pady=(0, 6))
        composer_actions = tk.Frame(composer, bg=COLORS["panel"])
        composer_actions.pack(fill="x", padx=12, pady=(0, 10))
        self.attach_button = self._button(composer_actions, "Attach files", self.attach_files)
        self.attach_button.configure(padx=9, pady=8, font=("Segoe UI", 9))
        self.attach_button.pack(side="left")
        self._label(composer_actions, "Enter sends · Shift+Enter newline", size=8, color=COLORS["muted"]).pack(side="left", padx=(8, 0))
        self.send_button = self._button(composer_actions, "Send  ↑", self.send, primary=True, state="disabled")
        self.send_button.pack(side="right")
        self.stop_button = self._button(composer_actions, "Stop", self.stop, state="disabled")
        self.stop_button.pack(side="right", padx=(0, 8))
        self._label(self.footer_frame, "Files · Code · Terminal", size=8, color=COLORS["muted"]).pack(pady=(10, 16))
        self._refresh_memory_indicator()
        self.input.focus_set()

    def _append(self, text, tag="body"):
        self.transcript.configure(state="normal")
        self.transcript.insert("end", text, tag)
        self.transcript.configure(state="disabled")
        self.transcript.see("end")

    def _text_menu(self, event, widget, *, editable=False):
        menu = tk.Menu(self.root, tearoff=False)
        menu.add_command(label="Copy", command=lambda: widget.event_generate("<<Copy>>"))
        if editable:
            menu.add_command(label="Cut", command=lambda: widget.event_generate("<<Cut>>"))
            menu.add_command(label="Paste", command=self._paste_input)
        menu.add_separator()
        menu.add_command(label="Select all", command=lambda: widget.tag_add("sel", "1.0", "end-1c"))
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()
        return "break"

    def attach_files(self):
        if self.closing or self.updating:
            return
        paths = filedialog.askopenfilenames(parent=self.root, title="Attach files to your message",
            filetypes=[("Images, text, code, and Word documents", "*.png *.jpg *.jpeg *.gif *.bmp *.txt *.md *.csv *.json *.py *.js *.ts *.html *.css *.docx"),
                       ("All files", "*.*")])
        if paths:
            self._prepare_attachments(list(paths))

    def _prepare_attachments(self, paths=None, *, clipboard=False, fallback_text="", paste_marks=None):
        if self.closing:
            return
        if not clipboard:
            paths = list(paths or [])
            available = max(0, 8 - len(self.pending_attachments))
            if len(paths) > available:
                self.status_label.configure(text=f"Choose up to {available} more file(s). A message holds at most 8 attachments.")
                return
        epoch = self.attachment_epoch
        self.attachment_jobs += 1
        self.status_label.configure(text="Preparing attachments… Images, text/code, CSV, JSON, and DOCX are supported.")
        self._update_model_controls()

        def worker():
            items, errors = [], []
            try:
                import runtime
                if clipboard:
                    items = runtime.clipboard_attachments()
                else:
                    for path in paths or []:
                        try:
                            items.append(runtime.prepare_attachment(path))
                        except Exception as exc:
                            errors.append(f"{Path(path).name}: {exc}")
            except Exception as exc:
                errors.append(str(exc))
            self.events.put(("attachments_prepared", {"epoch": epoch, "items": items, "errors": errors,
                             "paste": fallback_text if clipboard and not items and not errors else "",
                             "paste_marks": paste_marks}, None))
        threading.Thread(target=worker, daemon=True).start()

    def _paste_input(self, _event=None):
        if self.closing or self.updating:
            return "break"
        try:
            plain_text = self.root.clipboard_get()
        except tk.TclError:
            plain_text = ""
        prefix = f"attachment_paste_{time.monotonic_ns()}"
        marks = {"insert": prefix}
        self.input.mark_set(prefix, "insert")
        self.input.mark_gravity(prefix, "left")
        try:
            for key, index in (("start", "sel.first"), ("end", "sel.last")):
                marks[key] = prefix + "_" + key
                self.input.mark_set(marks[key], index)
                self.input.mark_gravity(marks[key], "left")
        except tk.TclError:
            marks.pop("start", None)
            marks.pop("end", None)
        self._prepare_attachments(clipboard=True, fallback_text=plain_text, paste_marks=marks)
        return "break"

    def _add_attachments(self, items):
        if items and self.settings_expanded:
            self.toggle_settings()
        for item in items:
            if not isinstance(item, dict) or not item.get("id"):
                continue
            if any(existing.get("id") == item["id"] for existing in self.pending_attachments):
                continue
            if len(self.pending_attachments) >= 8:
                self._append("A message can contain up to 8 attachments. Send these first, then attach more.\n", "error")
                break
            if item.get("kind") == "image" and sum(entry.get("kind") == "image" for entry in self.pending_attachments) >= 4:
                self._append("A message can contain up to 4 images. Send these first, then attach more.\n", "error")
                continue
            self.pending_attachments.append(copy.deepcopy(item))
            if item.get("warning"):
                self._append(str(item["warning"]) + "\n", "muted")
        self._render_pending_attachments()

    def clear_attachments(self):
        self.pending_attachments = []
        self.attachment_epoch += 1
        self.attachment_jobs = 0
        for mark in self.input.mark_names():
            if mark.startswith("attachment_paste_"):
                self.input.mark_unset(mark)
        self._render_pending_attachments()
        self._update_model_controls()

    def _remove_attachment(self, attachment_id):
        self.pending_attachments = [item for item in self.pending_attachments if item.get("id") != attachment_id]
        self._render_pending_attachments()

    def _attachment_photo(self, item, *, width=220, height=150):
        if item.get("kind") != "image":
            return None
        try:
            import runtime
            path = runtime.resolve_attachment_path(item, preview=True)
            photo = tk.PhotoImage(file=str(path))
            return photo.subsample(max(1, math.ceil(max(photo.width() / width, photo.height() / height))))
        except (OSError, ValueError, KeyError, AttributeError, tk.TclError):
            return None

    def _render_pending_attachments(self):
        for child in self.pending_chips.winfo_children():
            child.destroy()
        self.pending_previews = []
        if not self.pending_attachments:
            self.pending_frame.pack_forget()
            return
        self.pending_frame.pack(fill="x", padx=12, pady=(0, 5), before=self.always_allow_box)
        for item in self.pending_attachments:
            chip = tk.Frame(self.pending_chips, bg=COLORS["border"], padx=5, pady=3)
            chip.pack(side="left", padx=(0, 5))
            photo = self._attachment_photo(item, width=28, height=28)
            if photo:
                self.pending_previews.append(photo)
                tk.Label(chip, image=photo, bg=COLORS["border"]).pack(side="left", padx=(0, 5))
            name = str(item.get("name", "Attachment"))
            self._label(chip, name if len(name) < 25 else name[:22] + "…", size=8).pack(side="left")
            button = self._button(chip, "×", lambda key=item["id"]: self._remove_attachment(key))
            button.configure(pady=0, padx=5, font=("Segoe UI", 10))
            button.pack(side="left", padx=(3, 0))
        self.pending_canvas.xview_moveto(0)

    def _open_attachment(self, item):
        try:
            import runtime
            path = runtime.resolve_attachment_path(item)
            if item.get("kind") == "image":
                if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".gif", ".bmp"}:
                    raise ValueError("The stored file is not an image. Attach the original image again.")
                os.startfile(str(path))
            elif item.get("kind") == "text":
                notepad = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "notepad.exe"
                subprocess.Popen([str(notepad), str(path)])
            else:
                explorer = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "explorer.exe"
                subprocess.Popen([str(explorer), "/select,", str(path)])
        except (OSError, ValueError, KeyError) as exc:
            self.status_label.configure(text=f"Could not open attachment: {exc}")
            self._append(f"Could not open attachment: {exc}\n", "error")

    def _render_attachments(self, items):
        if not items:
            return
        card = self._inline_card("Attachments")
        for item in items:
            if not isinstance(item, dict):
                continue
            row = tk.Frame(card, bg=COLORS["panel"])
            row.pack(fill="x", pady=(0, 4))
            label = str(item.get("name", "Attachment"))
            self._label(row, label, size=9, wraplength=310, justify="left").pack(side="left")
            action = "Open image" if item.get("kind") == "image" else "Open in Notepad" if item.get("kind") == "text" else "Show file"
            button = self._button(row, action, lambda attachment=copy.deepcopy(item): self._open_attachment(attachment))
            button.configure(padx=8, pady=4, font=("Segoe UI", 8))
            button.pack(side="right", padx=(8, 0))
            photo = self._attachment_photo(item)
            if photo:
                self.chat_previews.append(photo)
                preview = tk.Label(card, image=photo, bg=COLORS["panel"], cursor="hand2")
                preview.pack(anchor="w", pady=(0, 8))
                preview.bind("<Button-1>", lambda _event, attachment=copy.deepcopy(item): self._open_attachment(attachment))
        self._resize_inline_cards()

    def _append_user(self, prompt, items, *, queued=False):
        self._append("YOU · QUEUED\n" if queued else "YOU\n", "user_role")
        if prompt:
            self._append(prompt + "\n\n")
        self._render_attachments(items)

    def _coding_preset_changed(self, _event=None):
        if self.busy or self.downloading or self.updating:
            return
        selected = CODING_MODELS.get(self.coding_preset_var.get())
        if selected:
            self.download_name_var.set(selected[0])
            self.coding_preset_note.configure(text=selected[1] + " Click Download to add it.")

    def _prompt_card(self, title):
        """Pin active decisions above the composer; only their body scrolls."""
        if self.settings_expanded:
            self.toggle_settings()
        if self.prompt_panel is not None:
            self.prompt_panel.destroy()
        panel = tk.Frame(self.footer_frame, bg=COLORS["panel"], padx=10, pady=7,
                         highlightthickness=1, highlightbackground=COLORS["mint_dark"])
        panel.pack(fill="x", padx=30, pady=(0, 6), before=self.composer_border)
        self.prompt_panel = panel
        self._label(panel, title, bold=True, color=COLORS["mint"]).pack(anchor="w")
        viewport = tk.Frame(panel, bg=COLORS["panel"])
        viewport.pack(fill="x", pady=4)
        canvas = tk.Canvas(viewport, height=90, width=1, bg=COLORS["panel"], highlightthickness=0)
        scroll = ttk.Scrollbar(viewport, orient="vertical", command=canvas.yview)
        scroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="x", expand=True)
        canvas.configure(yscrollcommand=scroll.set)
        body = tk.Frame(canvas, bg=COLORS["panel"])
        item = canvas.create_window(0, 0, window=body, anchor="nw")
        self.prompt_body = body
        self.prompt_actions = tk.Frame(panel, bg=COLORS["panel"])
        self.prompt_actions.pack(fill="x")
        self.input.configure(height=2)

        def wheel(event):
            canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")
            return "break"

        def fit(event=None):
            if not panel.winfo_exists():
                return
            width = max(200, canvas.winfo_width())
            canvas.itemconfigure(item, width=width)
            def visit(widget):
                for child in widget.winfo_children():
                    if isinstance(child, (tk.Label, tk.Radiobutton)):
                        child.configure(wraplength=max(180, width - 16))
                    if not isinstance(child, tk.Text):
                        child.bind("<MouseWheel>", wheel)
                    visit(child)
            visit(body)
            canvas.configure(scrollregion=canvas.bbox("all"),
                             height=min(120, max(60, self.root.winfo_height() // 7)))
        canvas.bind("<Configure>", fit)
        body.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<MouseWheel>", wheel)
        self.root.after_idle(fit)
        return body

    def _toggle_question_permission(self):
        if self.decide_questions_var.get() and self.question_request is not None and not self.cancel.is_set():
            self._answer_question(self._delegated_question_answer())

    @staticmethod
    def _delegated_question_answer():
        return ("I selected Don't ask again in this chat for clarification questions. "
                "Use your best judgment for routine, reversible implementation choices and state your assumptions. "
                "Do not invent missing facts or credentials. If an essential fact cannot be inferred, explain the blocker. "
                "This does not approve commands, destructive actions, purchases, or external sharing.")

    def _inline_card(self, title):
        card = tk.Frame(self.transcript, bg=COLORS["panel"], highlightbackground=COLORS["border"],
                        highlightthickness=1, padx=12, pady=10)
        self._label(card, title, size=11, bold=True, color=COLORS["mint"]).pack(anchor="w", pady=(0, 8))
        self.transcript.configure(state="normal")
        self.transcript.window_create("end", window=card, padx=2, pady=6)
        self.transcript.insert("end", "\n\n")
        self.transcript.configure(state="disabled")
        self.inline_cards.append(card)
        self.transcript.see("end")
        return card

    def _resize_inline_cards(self, _event=None):
        width = max(270, self.transcript.winfo_width() - 48)

        def resize(widget):
            for child in widget.winfo_children():
                if isinstance(child, (tk.Label, tk.Radiobutton)):
                    child.configure(wraplength=max(240, width - 28))
                elif isinstance(child, tk.Text):
                    char_width = max(1, tkfont.Font(font=child.cget("font")).measure("0"))
                    child.configure(width=max(24, int((width - 30) / char_width)))
                elif isinstance(child, tk.Canvas):
                    child.configure(width=max(240, width - 30))
                resize(child)
        self.inline_cards = [card for card in self.inline_cards if card.winfo_exists()]
        for card in self.inline_cards:
            resize(card)

    def _resolve_inline_card(self, card, text):
        if card is not None and card is self.prompt_body:
            self.prompt_panel.destroy()
            self.prompt_panel = self.prompt_body = None
            self.input.configure(height=3)
            self._append(text + "\n", "muted")
            return
        if card is None or not card.winfo_exists():
            return
        for child in card.winfo_children():
            child.destroy()
        self._label(card, text, size=9, color=COLORS["muted"], justify="left", wraplength=450).pack(anchor="w")
        self._resize_inline_cards()

    def _toggle_command_permission(self):
        if self.always_allow_var.get():
            self.allow_commands.set()
            if self.approval_request is not None and not self.cancel.is_set():
                self._answer_approval(True, allow_chat=True)
        else:
            self.allow_commands.clear()

    def _reset_command_permission(self):
        self.allow_commands.clear()
        self.always_allow_var.set(False)
        self.decide_questions_var.set(False)

    @staticmethod
    def _short_path(path):
        text = str(path)
        return text if len(text) <= 62 else "…" + text[-61:]

    @staticmethod
    def _short_name(path):
        text = Path(path).name or str(path)
        return text if len(text) <= 36 else text[:35] + "…"

    def _refresh_memory_indicator(self):
        if self.memory_state:
            self.memory_label.pack(fill="x", padx=31, pady=(0, 8), before=self.composer_border)
        else:
            self.memory_label.pack_forget()

    @staticmethod
    def _clean_memory_state(value):
        if not isinstance(value, dict):
            return {}
        count = value.get("compactions")
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            return {}
        covered = value.get("covered_messages", 0)
        return {"compactions": count, "covered_messages": covered if isinstance(covered, int) and covered >= 0 else 0}

    @classmethod
    def _history_memory_state(cls, history):
        for message in history:
            if message.get("role") == "system":
                marker = message.get("_local_agent_context")
                if isinstance(marker, dict) and marker.get("schema") == 1:
                    return cls._clean_memory_state(marker)
        return {}

    def _can_send(self):
        return (self.ready and not self.updating and not self.downloading and not self.model_checking and not self._model_block_reason()
                and not self.attachment_jobs and not self.closing and self.workspace.is_dir() and not (self.busy and self.cancel.is_set()))

    def _model_block_reason(self):
        if self.model_info_name != self.model_var.get().strip():
            return ""
        if self.model_tool_support is False:
            return "This model does not support tools. Choose a tool-capable model for agent work."
        if self.model_context_support is False:
            return "This model's context is too small. Agent work needs at least 8,192 tokens of context."
        return ""

    def toggle_settings(self):
        self.settings_expanded = not self.settings_expanded
        self.settings_toggle.configure(text="Models & performance ▴" if self.settings_expanded else "Models & performance ▾")
        if self.settings_expanded:
            self.setup_bar.pack_forget()
            self.settings_container.pack(fill="x", padx=30, pady=(10, 0), before=self.transcript_frame)
            self.refresh_hardware()
            self.refresh_model_info()
        else:
            self.settings_container.pack_forget()
            if self.available_models == []:
                self.setup_bar.pack(fill="x", padx=30, pady=(12, 0), before=self.transcript_frame)
        self._update_model_controls()

    def _selected_profile(self):
        return next((key for key, label in PROFILE_LABELS.items() if label == self.profile_var.get()), "auto")

    def _profile_changed(self, _event=None):
        if self.busy or self.downloading or self.updating:
            return
        self.performance_profile = self._selected_profile()
        self._save_state()
        self.status_label.configure(text="Performance profile saved. It applies to your next task.")

    def _update_model_controls(self):
        blocked = self.busy or self.downloading or self.connecting or self.updating or self.closing
        state = "disabled" if blocked else "normal"
        for widget in (self.download_name_entry, self.custom_download_button, self.import_name_entry, self.choose_gguf_button):
            widget.configure(state=state)
        self.import_button.configure(state="normal" if self.import_path and not blocked else "disabled")
        self.profile_box.configure(state="disabled" if blocked else "readonly")
        self.coding_preset_box.configure(state="disabled" if blocked else "readonly")
        self.model_box.configure(state=state)
        self.think_box.configure(state="disabled" if blocked or self.model_thinking_support is False else "normal")
        self.download_button.configure(state=state)
        self.send_button.configure(state="normal" if self._can_send() else "disabled")
        self.attach_button.configure(state="disabled" if self.updating or self.closing else "normal")
        installed = CODING_V2_MODEL in (self.available_models or [])
        available = self._coding_v2_installer_available()
        self.install_coding_v2_button.configure(state="disabled" if blocked or installed or not available else "normal",
                                                text="Installed" if installed else "Install coding v2")
        if not available and not installed:
            self.coding_v2_status.configure(text="Update Local Agent to enable this installer.")
        self.cancel_coding_v2_button.configure(state="normal" if self.installing_coding_v2 and not self.coding_v2_importing
                                               and not self.coding_v2_cancel.is_set() and not self.closing else "disabled")

    @staticmethod
    def _coding_v2_installer_available():
        try:
            import runtime
            return callable(getattr(runtime, "install_coding_v2", None))
        except ImportError:
            return False

    def install_coding_v2(self):
        if self.busy or self.downloading or self.connecting or self.updating or self.closing:
            return
        if not self._coding_v2_installer_available():
            self.coding_v2_status.configure(text="Update Local Agent to enable this installer.")
            self._update_model_controls()
            return
        if CODING_V2_MODEL in (self.available_models or []):
            self.coding_v2_status.configure(text=f"Already installed. Choose {CODING_V2_MODEL} in the model list.")
            return
        self.installing_coding_v2 = self.downloading = True
        self.coding_v2_importing = False
        self.coding_v2_cancel = threading.Event()
        cancel = self.coding_v2_cancel
        self.pending_model_selection = None
        self.coding_v2_status.configure(text="Preparing the experimental model installation…")
        self.coding_v2_progress.configure(value=0)
        self.coding_v2_progress.pack(fill="x", pady=(5, 0))
        self.retry_button.configure(state="disabled")
        self.update_button.configure(state="disabled")
        self.status_label.configure(text="Installing experimental coding v2. Choose it from the model list when ready.")
        self._update_model_controls()

        def worker():
            try:
                import runtime
                runtime.install_coding_v2(progress=lambda event: self.events.put(("coding_v2_progress", event, None)),
                                          cancel_event=cancel)
            except InterruptedError as exc:
                self.events.put(("coding_v2_finished", {"cancelled": True, "text": str(exc)}, None))
            except Exception as exc:
                self.events.put(("coding_v2_finished", {"error": str(exc)}, None))
            else:
                self.events.put(("coding_v2_finished", {"success": True}, None))
        # Keep Python alive until model registration and cleanup have finished.
        self.coding_v2_worker = threading.Thread(target=worker, daemon=False)
        self.coding_v2_worker.start()

    def cancel_coding_v2_install(self):
        if self.installing_coding_v2 and not self.coding_v2_importing:
            self.coding_v2_cancel.set()
            self.coding_v2_status.configure(text="Cancelling the download and cleaning up…")
            self._update_model_controls()

    def refresh_hardware(self):
        if not self.background_enabled or self.hardware_checking or self.closing:
            return
        self.hardware_checking = True

        def worker():
            try:
                import runtime
                self.events.put(("hardware_info", runtime.get_hardware_info(), None))
            except Exception as exc:
                self.events.put(("hardware_error", {"text": str(exc)}, None))
        threading.Thread(target=worker, daemon=True).start()

    def refresh_model_info(self):
        if not self.background_enabled or not self.ready or self.busy or self.closing:
            return
        model = self.model_var.get().strip()
        if not model:
            return
        self.model_check_serial += 1
        serial = self.model_check_serial
        self.model_checking = True
        self.model_capability_label.configure(text=f"Checking tool support for {model}…", fg=COLORS["muted"])
        self.send_button.configure(state="disabled")

        def worker():
            try:
                import runtime
                info = runtime.get_model_info(model)
                self.events.put(("model_info", {"model": model, "serial": serial, "info": info}, None))
            except Exception as exc:
                self.events.put(("model_info_error", {"model": model, "serial": serial, "text": str(exc)}, None))
        threading.Thread(target=worker, daemon=True).start()

    def choose_gguf(self):
        if self.busy or self.downloading or self.updating or self.connecting:
            return
        path = filedialog.askopenfilename(parent=self.root, title="Choose a local GGUF model",
                                         filetypes=[("GGUF models", "*.gguf")])
        if path:
            self.import_path = path
            self.gguf_label.configure(text=Path(path).name)
            self._update_model_controls()

    def download_custom_model(self):
        name = self.download_name_var.get().strip()
        if not name:
            self.status_label.configure(text="Enter an Ollama model name to download.")
            return
        self._start_model_operation("download", name)

    def import_model(self):
        name = self.import_name_var.get().strip()
        if not name or not self.import_path:
            self.status_label.configure(text="Choose a GGUF file and enter a model name first.")
            return
        self._start_model_operation("import", name, self.import_path)

    def _start_model_operation(self, operation, model, path=None):
        if self.downloading or self.connecting or self.updating or self.busy or self.closing:
            return
        self.downloading = True
        self.pending_model_selection = model
        self.model_operation_label = "Importing model" if operation == "import" else "Downloading model"
        self.download_button.configure(text="Downloading…" if operation == "download" else "Importing…")
        self.retry_button.configure(state="disabled")
        self.update_button.configure(state="disabled")
        self._update_model_controls()
        self.status_label.configure(text=f"{self.model_operation_label}: {model}. Keep this window open.")

        def worker():
            try:
                import runtime
                progress = lambda event: self.events.put(("model_download_status", event, None))
                if operation == "import":
                    runtime.import_gguf(model, path, progress=progress)
                else:
                    runtime.pull_model(model, progress=progress)
                self.events.put(("model_download_complete", {"model": model}, None))
            except Exception as exc:
                self.events.put(("model_download_error", {"text": str(exc)}, None))
        threading.Thread(target=worker, daemon=True).start()

    def _tick(self):
        if self.closing:
            return
        elapsed = self.elapsed_seconds
        if self.busy and self.work_started is not None:
            elapsed = time.monotonic() - self.work_started
        seconds = max(0, int(elapsed))
        clock = f"{seconds // 60:02d}:{seconds % 60:02d}" if seconds < 3600 else f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"
        speed = "—" if self.last_tokens_per_second is None else f"{self.last_tokens_per_second:.1f}"
        queued = f" · {self.queued_count} queued" if self.queued_count else ""
        self.metrics_label.configure(text=f"Elapsed {clock} · Last generation {speed} tokens/s{queued}")
        self._tick_id = self.root.after(250, self._tick)

    def open_update(self):
        if self.busy or self.connecting or self.downloading or self.updating or self.closing:
            return
        try:
            import updater
            self.updating = True
            self._update_model_controls()
            self.send_button.configure(state="disabled")
            self.update_button.configure(state="disabled")
            self.retry_button.configure(state="disabled")
            dialog = updater.open_update_dialog(self.root)

            def finished(event):
                if event.widget is dialog and not self.closing and self.update_button.winfo_exists():
                    self.updating = False
                    self.update_button.configure(state="normal")
                    self.retry_button.configure(state="normal")
                    self._update_model_controls()
            dialog.bind("<Destroy>", finished, add=True)
        except Exception as exc:
            self.updating = False
            self.update_button.configure(state="normal")
            self.retry_button.configure(state="normal")
            self._update_model_controls()
            messagebox.showerror("Couldn't open updater", str(exc), parent=self.root)

    def download_model(self):
        self._start_model_operation("download", DEFAULT_MODEL)

    def _render_history(self):
        for card in self.inline_cards:
            if card.winfo_exists():
                card.destroy()
        self.inline_cards = []
        self.chat_previews = []
        self.transcript.configure(state="normal")
        self.transcript.delete("1.0", "end")
        self.transcript.configure(state="disabled")
        visible = [m for m in self.history if m.get("role") in ("user", "assistant") and (m.get("content") or m.get("attachments"))]
        if not visible:
            self._append("A little help.\nA lot done.\n", "welcome_title")
            self._append("Ask your agent to explore a project, write code,\nfix a problem, or run a command.\n\n", "muted")
            self._append("TRY A TASK\n", "agent_role")
            self._append("“Look through this folder and explain the project.”\n", "body")
            self._append("“Create a small Python app and test it.”\n", "body")
            self._append("“Find the cause of this error and fix it.”\n\n", "body")
            self._append("Your selected folder is the starting point.\nYou'll see file actions and command results here.", "muted")
        else:
            for message in self.history:
                if message.get("role") in ("user", "assistant") and (message.get("content") or message.get("attachments")):
                    is_user = message["role"] == "user"
                    self._append("YOU\n" if is_user else "LOCAL AGENT\n", "user_role" if is_user else "agent_role")
                    content = message.get("user_text", message.get("_display_content", message.get("content", "")))
                    if content:
                        self._append(str(content) + "\n\n")
                    self._render_attachments(message.get("attachments", []))
                for call in message.get("tool_calls", []):
                    function = call.get("function", {})
                    if function.get("name") == "ask_user":
                        arguments = function.get("arguments", {})
                        if isinstance(arguments, dict) and arguments.get("question"):
                            self._append("QUESTION FOR YOU\n", "agent_role")
                            self._append(str(arguments["question"]) + "\n\n")
                if message.get("role") == "tool" and message.get("tool_name") == "ask_user":
                    try:
                        result = json.loads(message.get("content", "{}"))
                        if result.get("answer"):
                            self._append("YOU · ANSWER\n", "user_role")
                            self._append(str(result["answer"]) + "\n\n")
                    except (ValueError, TypeError, AttributeError):
                        pass

    def _load_state(self):
        try:
            saved = json.loads(STATE.read_text(encoding="utf-8"))
            workspace = Path(saved.get("workspace", str(self.workspace)))
            self.workspace = workspace.resolve()
            # A moved/deleted project must not erase its saved conversation or
            # silently attach it to a different working folder.
            history = saved.get("history", [])
            if isinstance(history, list) and all(isinstance(m, dict) for m in history):
                self.history = history
            self.model_name = str(saved.get("model") or DEFAULT_MODEL)
            self.reason_more = bool(saved.get("reason_more", False))
            profile = saved.get("performance_profile", "auto")
            self.performance_profile = profile if isinstance(profile, str) and profile in PROFILE_LABELS else "auto"
            if self.history:
                self.memory_state = self._history_memory_state(self.history) or self._clean_memory_state(saved.get("memory_state"))
        except (OSError, ValueError, TypeError):
            pass

    def _save_state(self):
        try:
            DATA.mkdir(parents=True, exist_ok=True)
            target = STATE.with_suffix(".tmp")
            target.write_text(json.dumps({"workspace": str(self.workspace), "model": self.model_var.get().strip() or DEFAULT_MODEL,
                                          "reason_more": self.think_var.get(), "memory_state": self.memory_state,
                                          "performance_profile": self._selected_profile(),
                                          "history": self.history}, ensure_ascii=False, indent=2), encoding="utf-8")
            target.replace(STATE)
        except OSError as exc:
            self.status_label.configure(text=f"Could not save chat: {exc}")

    def _model_changed(self, _event=None):
        if not self.busy:
            self.model_name = self.model_var.get().strip() or DEFAULT_MODEL
            if self.model_name in (self.available_models or []):
                self.ready = True
            self._save_state()
            if self.model_name != self.model_info_name:
                self.model_info_name = ""
                self.model_tool_support = None
                self.model_thinking_support = None
                self.model_context_support = None
                self.model_vision_support = None
                self.model_context_tokens = None
                self.model_capability_label.configure(text="Tool support: not checked yet.", fg=COLORS["muted"])
                self.refresh_model_info()
                self._update_model_controls()

    def new_chat(self):
        if self.busy:
            return
        if not self._archive_history():
            return
        self.history = []
        self.clear_attachments()
        self._reset_command_permission()
        self.elapsed_seconds = 0.0
        self.last_tokens_per_second = None
        self.memory_state = {}
        self._refresh_memory_indicator()
        self._render_history()
        self._save_state()
        self.status_label.configure(text="New chat. What would you like to do?")
        self.send_button.configure(state="normal" if self._can_send() else "disabled")
        if not self.workspace.is_dir():
            self.status_label.configure(text="Choose an existing folder before sending a task.")
        self.input.focus_set()

    def _archive_history(self):
        if not self.history:
            return True
        try:
            directory = DATA / "chats"
            directory.mkdir(parents=True, exist_ok=True)
            filename = datetime.now().strftime("%Y-%m-%d %H-%M-%S-%f") + ".json"
            (directory / filename).write_text(json.dumps({"workspace": str(self.workspace),
                "model": self.model_var.get().strip() or DEFAULT_MODEL,
                "memory_state": self.memory_state, "history": self.history},
                ensure_ascii=False, indent=2), encoding="utf-8")
            return True
        except OSError as exc:
            messagebox.showerror("Couldn't save this chat", str(exc), parent=self.root)
            return False

    def open_saved_chat(self):
        if self.busy:
            return
        directory = DATA / "chats"
        directory.mkdir(parents=True, exist_ok=True)
        filename = filedialog.askopenfilename(parent=self.root, title="Open a saved chat", initialdir=str(directory),
                                               filetypes=[("Saved chats", "*.json")])
        if not filename:
            return
        try:
            saved = json.loads(Path(filename).read_text(encoding="utf-8"))
            workspace = Path(saved["workspace"]).resolve()
            history = saved["history"]
            if not isinstance(history, list) or not all(isinstance(m, dict) for m in history):
                raise ValueError("This file doesn't contain a saved chat.")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            messagebox.showerror("Couldn't open chat", str(exc), parent=self.root)
            return
        if not self._archive_history():
            return
        self.workspace = workspace
        self.history = history
        self.clear_attachments()
        self.memory_state = self._history_memory_state(history) or self._clean_memory_state(saved.get("memory_state"))
        self._refresh_memory_indicator()
        self._reset_command_permission()
        self.model_var.set(str(saved.get("model") or DEFAULT_MODEL))
        self.model_info_name = ""
        self.model_tool_support = None
        self.model_thinking_support = None
        self.model_context_support = None
        self.model_vision_support = None
        self.model_context_tokens = None
        self.folder_title.configure(text=self._short_name(workspace))
        self.folder_path.configure(text=self._short_path(workspace))
        self._render_history()
        self._save_state()
        self.send_button.configure(state="normal" if self._can_send() else "disabled")
        self.status_label.configure(text="Saved chat opened. You can continue the conversation." if workspace.is_dir()
                                    else "Chat restored. Its folder is unavailable; choose a folder before sending a task.")
        self.refresh_model_info()

    def choose_folder(self):
        if self.busy:
            return
        path = filedialog.askdirectory(parent=self.root, initialdir=str(self.workspace), title="Choose your agent's working folder", mustexist=True)
        if path:
            selected = Path(path).resolve()
            if selected != self.workspace:
                if not self._archive_history():
                    return
                self.history = []
                self.workspace = selected
                self.folder_title.configure(text=self._short_name(selected))
                self.folder_path.configure(text=self._short_path(selected))
                self.new_chat()
                self.status_label.configure(text=f"Ready to work in {selected.name}.")

    def open_folder(self):
        try:
            os.startfile(str(self.workspace))
        except OSError as exc:
            messagebox.showerror("Could not open folder", str(exc), parent=self.root)

    def connect(self):
        if self.connecting or self.busy or self.downloading or self.updating or self.closing:
            return
        self.connecting = True
        self.ready = False
        self.send_button.configure(state="disabled")
        self.retry_button.configure(state="disabled")
        self.update_button.configure(state="disabled")
        self._update_model_controls()
        self.engine_label.configure(text="●  Starting local engine", fg=COLORS["muted"])
        self.status_label.configure(text="Connecting to your local model…")

        def worker():
            try:
                import runtime
                base_url = runtime.ensure_server()
                models = runtime.list_models()
                self.events.put(("connected", {"base_url": base_url, "models": models}, None))
            except Exception as exc:
                self.events.put(("connection_error", {"text": str(exc)}, None))
        threading.Thread(target=worker, daemon=True).start()

    def _set_busy(self, busy):
        self.busy = busy
        for widget in (self.new_button, self.saved_button, self.folder_button, self.retry_button, self.update_button):
            widget.configure(state="disabled" if busy else "normal")
        self.model_box.configure(state="disabled" if busy else "normal")
        self.think_box.configure(state="disabled" if busy else "normal")
        self.send_button.configure(state="normal" if self._can_send() else "disabled", text="Send follow-up" if busy else "Send  ↑")
        self.stop_button.configure(state="normal" if busy else "disabled")
        self.input.configure(state="normal")
        self._update_model_controls()

    def _send_key(self, _event):
        self.send()
        return "break"

    def _newline_key(self, _event):
        self.input.insert("insert", "\n")
        return "break"

    def _drain_incoming(self):
        messages = []
        while True:
            try:
                messages.append(self.incoming.get_nowait())
            except queue.Empty:
                return messages

    def send(self):
        if not self.ready or self.updating or self.closing:
            return
        if self.downloading:
            self.status_label.configure(text="Wait for the model download or import to finish.")
            return
        if self.attachment_jobs:
            self.status_label.configure(text="Wait for the attachments to finish preparing, then send.")
            return
        if self.model_checking:
            self.status_label.configure(text="Checking the selected model's tool support…")
            return
        if self._model_block_reason():
            self.status_label.configure(text=self._model_block_reason())
            return
        if self.background_enabled and not self.busy and self.model_info_name != self.model_var.get().strip():
            self.refresh_model_info()
            self.status_label.configure(text="Checking the selected model's tool support. Send again when it is ready.")
            return
        if not self.workspace.is_dir():
            self.status_label.configure(text="Your previous folder is unavailable. Choose a folder before sending a task.")
            return
        prompt = self.input.get("1.0", "end-1c").strip()
        attachments = copy.deepcopy(self.pending_attachments)
        model = self.model_var.get().strip()
        if not prompt and not attachments:
            return
        images = sum(item.get("kind") == "image" for item in attachments)
        if images and self.model_vision_support is False:
            self.status_label.configure(text="This model cannot read images. Choose a vision model or remove the image attachments.")
            return
        if images > 1 and self.model_context_tokens and self.model_context_tokens < 16384:
            self.status_label.configure(text="This model's context supports one image per message. Remove extra images or choose a 16K model.")
            return
        if not model:
            self.status_label.configure(text="Choose a local model first.")
            return
        if self.busy:
            if self.cancel.is_set():
                self.status_label.configure(text="Wait for the task to stop, then send your next message.")
                return
            self.input.delete("1.0", "end")
            if self.question_request is not None:
                if attachments:
                    self.incoming.put({"content": prompt, "attachments": attachments})
                    self.queued_count += 1
                    self._append_user(prompt, attachments, queued=True)
                    self.clear_attachments()
                self._answer_question(prompt or "Please use the attached files.")
                return
            self.incoming.put({"content": prompt, "attachments": attachments} if attachments else prompt)
            self.queued_count += 1
            self._append_user(prompt, attachments, queued=True)
            self.clear_attachments()
            self.status_label.configure(text="Message queued. Your agent will read it before its next step.")
            return
        if not self.history:
            self.transcript.configure(state="normal")
            self.transcript.delete("1.0", "end")
            self.transcript.configure(state="disabled")
        self._append_user(prompt, attachments)
        self.input.delete("1.0", "end")
        self.clear_attachments()
        if attachments:
            self._start_run(prompt, attachments=attachments)
        else:
            self._start_run(prompt)

    def _start_run(self, prompt, *, attachments=None, continuation=False):
        self.cancel = threading.Event()
        self.run_id += 1
        run_id = self.run_id
        history = copy.deepcopy(self.history)
        previous_user_count = sum(message.get("role") == "user" for message in history)
        self.stream_started = False
        self.stream_text = ""
        self.stream_mark = None
        self.stream_start_mark = None
        self.stream_end_mark = None
        if not continuation or self.work_started is None:
            self.work_started = time.monotonic()
            self.elapsed_seconds = 0.0
            self.last_tokens_per_second = None
        model = self.model_var.get().strip()
        think = bool(self.think_var.get()) and self.model_thinking_support is not False
        performance_profile = self._selected_profile()
        vision_supported = self.model_vision_support
        workspace = str(self.workspace)
        base_url = self.base_url
        cancel = self.cancel
        incoming = self.incoming
        self._set_busy(True)
        self.status_label.configure(text="Working on your task…")

        def emit(event, payload):
            self.events.put((event, payload, run_id))

        def approve(command, cwd):
            if cancel.is_set() or self.closing:
                return False
            if self.allow_commands.is_set():
                return True
            request = {"command": command, "cwd": cwd, "event": threading.Event(), "allowed": False}
            emit("approval", request)
            while not request["event"].wait(0.1):
                if cancel.is_set() or self.closing:
                    return False
            return bool(request["allowed"]) and not cancel.is_set()

        def ask_user(question, options):
            if cancel.is_set() or self.closing:
                return None
            request = {"question": str(question), "options": list(options or []),
                       "event": threading.Event(), "answer": None}
            emit("question", request)
            while not request["event"].wait(0.1):
                if cancel.is_set() or self.closing:
                    return None
            return request["answer"] if not cancel.is_set() else None

        def poll_messages():
            messages = []
            if cancel.is_set() or self.closing:
                return messages
            while True:
                try:
                    messages.append(incoming.get_nowait())
                except queue.Empty:
                    return messages

        def worker():
            final_history = history
            try:
                from agent_core import Agent
                agent = Agent(workspace, emit, approve, cancel, model=model, base_url=base_url,
                              ask_user=ask_user, poll_messages=poll_messages, think=think,
                              performance_profile=performance_profile, vision_supported=vision_supported)
                self.agent = agent
                agent.history = history
                agent.run(prompt, attachments=attachments)
                final_history = agent.history
            except Exception as exc:
                emit("error", {"text": str(exc)})
                if self.agent is not None:
                    final_history = self.agent.history
            finally:
                result = {"history": final_history}
                if attachments and not cancel.is_set() and sum(message.get("role") == "user" for message in final_history) <= previous_user_count:
                    result["retry_draft"] = {"content": prompt, "attachments": attachments}
                emit("worker_complete", result)
        self.agent = None
        self.worker = threading.Thread(target=worker, daemon=True)
        self.worker.start()

    def stop(self):
        if not self.busy:
            return
        self.cancel.set()
        self._answer_approval(False)
        self._answer_question(None)
        dropped = self._drain_incoming()
        self.queued_count = 0
        if dropped:
            self._append(f"Stopped: {len(dropped)} queued message(s) were not sent to the agent.\n\n", "muted")
        self.stop_button.configure(state="disabled")
        self.send_button.configure(state="disabled")
        self.status_label.configure(text="Stopping…")

    def _show_approval(self, request):
        if self.cancel.is_set() or self.closing:
            request["event"].set()
            return
        self.approval_request = request
        if self.allow_commands.is_set():
            self._answer_approval(True)
            return
        content = self._prompt_card("Run this command?")
        self.approval_window = content
        self._label(content, str(request["cwd"]), size=9, color=COLORS["muted"],
                    wraplength=450, justify="left").pack(anchor="w", pady=(0, 8))
        command_frame = tk.Frame(content, bg=COLORS["panel"])
        command_frame.pack(fill="x")
        command_box = tk.Text(command_frame, height=3, width=48, wrap="word", relief="flat", bg=COLORS["side"], fg=COLORS["text"],
                              font=("Consolas", 10), padx=8, pady=8)
        scrollbar = ttk.Scrollbar(command_frame, orient="vertical", command=command_box.yview)
        command_box.configure(yscrollcommand=scrollbar.set)
        command_box.insert("1.0", request["command"])
        command_box.configure(state="disabled")
        scrollbar.pack(side="right", fill="y")
        command_box.pack(side="left", fill="both", expand=True)
        self._label(content, "Commands run with your Windows account and can access files outside the selected folder.",
                    size=8, color=COLORS["muted"], wraplength=450, justify="left").pack(anchor="w", pady=(8, 0))
        tk.Checkbutton(self.prompt_actions, text="Don't ask again in this chat (commands)",
                       variable=self.always_allow_var, command=self._toggle_command_permission,
                       bg=COLORS["panel"], fg=COLORS["text"], selectcolor=COLORS["side"],
                       activebackground=COLORS["panel"], font=("Segoe UI", 9)).pack(anchor="w")
        buttons = tk.Frame(self.prompt_actions, bg=COLORS["panel"])
        buttons.pack(fill="x", pady=(9, 0))
        self._button(buttons, "Allow once", lambda: self._answer_approval(True), primary=True).pack(side="right")
        self._button(buttons, "Reject", lambda: self._answer_approval(False)).pack(side="right", padx=8)
        self._resize_inline_cards()
        self.transcript.see("end")
        self.status_label.configure(text="Review the command in chat, or enable always allow below.")

    def _answer_approval(self, allowed, allow_chat=False):
        request = self.approval_request
        if request:
            request["allowed"] = allowed
            if allowed and allow_chat:
                self.allow_commands.set()
                self.always_allow_var.set(True)
            request["event"].set()
            self.approval_request = None
        if self.approval_window:
            outcome = "Command allowed for this chat." if allowed and self.allow_commands.is_set() else "Command allowed once." if allowed else "Command cancelled." if self.cancel.is_set() else "Command rejected."
            self._resolve_inline_card(self.approval_window, outcome)
            self.approval_window = None

    def _show_question(self, request):
        if self.cancel.is_set() or self.closing:
            request["event"].set()
            return
        self.question_request = request
        self._append("QUESTION FOR YOU\n", "agent_role")
        self._append(request["question"] + "\n\n")
        if self.decide_questions_var.get():
            self._answer_question(self._delegated_question_answer())
            return
        content = self._prompt_card("Choose an answer" if request.get("options") else "Your answer")
        self._label(content, request["question"], justify="left", wraplength=450).pack(fill="x", pady=(0, 6))
        self.question_window = content
        selected = tk.StringVar(value="")
        if request.get("options"):
            choices = content
            for option in request.get("options", [])[:6]:
                tk.Radiobutton(choices, text=str(option), variable=selected, value=str(option),
                               bg=COLORS["panel"], fg=COLORS["text"], activebackground=COLORS["panel"],
                               activeforeground=COLORS["text"], selectcolor=COLORS["side"],
                               font=("Segoe UI", 10), wraplength=420, justify="left", anchor="w").pack(fill="x", pady=3)
        self._label(content, "Or type your own answer in the message box below.", size=9,
                    color=COLORS["muted"], wraplength=450, justify="left").pack(anchor="w", pady=(8, 0))

        def submit(_event=None):
            text = self.input.get("1.0", "end-1c").strip() or selected.get().strip()
            if text:
                self.input.delete("1.0", "end")
                self._answer_question(text)
            return "break"
        buttons = tk.Frame(self.prompt_actions, bg=COLORS["panel"])
        buttons.pack(fill="x", pady=(9, 0))
        self._button(buttons, "Send answer", submit, primary=True).pack(side="right")
        self._button(buttons, "Skip", lambda: self._answer_question(None)).pack(side="right", padx=8)
        self.send_button.configure(text="Send answer")
        self.status_label.configure(text="Your agent is waiting for your answer.")
        self._resize_inline_cards()
        self.transcript.see("end")
        self.input.focus_set()

    def _answer_question(self, answer):
        request = self.question_request
        if request is not None:
            text = str(answer).strip() if answer is not None else None
            request["answer"] = text or None
            request["event"].set()
            self.question_request = None
            if text:
                self._append("YOU · ANSWER\n", "user_role")
                self._append(text + "\n\n")
                self.status_label.configure(text="Answer sent. Your agent is continuing.")
            else:
                self._append("Question skipped.\n\n", "muted")
            self.send_button.configure(text="Send follow-up" if self.busy else "Send  ↑")
        if self.question_window is not None:
            self._resolve_inline_card(self.question_window, "Answer received." if answer is not None and str(answer).strip() else "Question skipped.")
            self.question_window = None

    def _finish_stream(self):
        for mark in (self.stream_mark, self.stream_start_mark, self.stream_end_mark):
            if mark is not None:
                self.transcript.mark_unset(mark)
        self.stream_mark = None
        self.stream_start_mark = None
        self.stream_end_mark = None
        self.stream_started = False
        self.stream_text = ""

    def _stream_token(self, text):
        if not self.stream_started:
            start = self.transcript.index("end-1c")
            self._append("LOCAL AGENT\n", "agent_role")
            self.transcript.configure(state="normal")
            position = self.transcript.index("end-1c")
            self.transcript.insert("end", "\n\n", "body")
            self.stream_mark = f"stream_{self.run_id}"
            self.stream_start_mark = f"stream_start_{self.run_id}"
            self.stream_end_mark = f"stream_end_{self.run_id}"
            self.transcript.mark_set(self.stream_mark, position)
            self.transcript.mark_gravity(self.stream_mark, "right")
            self.transcript.mark_set(self.stream_start_mark, start)
            self.transcript.mark_gravity(self.stream_start_mark, "left")
            self.transcript.mark_set(self.stream_end_mark, "end-1c")
            self.transcript.mark_gravity(self.stream_end_mark, "left")
            self.transcript.configure(state="disabled")
            self.stream_started = True
        self.stream_text += text
        self.transcript.configure(state="normal")
        self.transcript.insert(self.stream_mark, text, "body")
        self.transcript.configure(state="disabled")
        self.transcript.see("end")

    def _discard_stream(self):
        if self.stream_started and self.stream_start_mark and self.stream_end_mark:
            self.transcript.configure(state="normal")
            self.transcript.delete(self.stream_start_mark, self.stream_end_mark)
            self.transcript.configure(state="disabled")
        self._finish_stream()

    def _handle_event(self, event, payload, run_id):
        if run_id is not None and run_id != self.run_id:
            return
        if event == "coding_v2_progress":
            if not self.installing_coding_v2:
                return
            kind = payload.get("kind")
            if kind == "progress":
                done, total = payload.get("done", 0), payload.get("total", 0)
                percent = max(0, min(100, 100 * done / total)) if total else 0
                self.coding_v2_progress.configure(value=percent)
                if not self.coding_v2_cancel.is_set():
                    self.coding_v2_status.configure(text=f"Downloading and verifying: {percent:.0f}% · {done / 1e9:.2f} / {total / 1e9:.2f} GB")
            elif kind == "importing":
                self.coding_v2_importing = True
                self.coding_v2_status.configure(text="Registering the verified model. This final step must finish before closing.")
                self._update_model_controls()
            elif kind == "success":
                self.coding_v2_status.configure(text="Finishing installation and cleaning up…")
            elif not self.coding_v2_cancel.is_set():
                self.coding_v2_status.configure(text=str(payload.get("text", "Installing coding v2…")))
        elif event == "coding_v2_finished":
            self.installing_coding_v2 = self.downloading = False
            self.coding_v2_importing = False
            self.coding_v2_progress.pack_forget()
            self.retry_button.configure(state="normal")
            self.update_button.configure(state="normal")
            if payload.get("success"):
                self.coding_v2_status.configure(text=f"Installed. Choose {CODING_V2_MODEL} in the model list to try it.")
                self.preserve_model_selection_on_refresh = True
                self._update_model_controls()
                self.connect()
            else:
                description = "Installation cancelled." if payload.get("cancelled") else str(payload.get("error", "Installation failed."))
                self.coding_v2_status.configure(text=description)
                self.status_label.configure(text=description)
                if not payload.get("cancelled"):
                    self._append("Coding v2 installation: " + description + "\n", "error")
                self._update_model_controls()
        elif event == "attachments_prepared":
            if payload.get("epoch") != self.attachment_epoch:
                return
            self.attachment_jobs = max(0, self.attachment_jobs - 1)
            self._add_attachments(payload.get("items", []))
            marks = payload.get("paste_marks") or {}
            if payload.get("paste"):
                if marks.get("start") and marks.get("end"):
                    self.input.delete(marks["start"], marks["end"])
                destination = marks.get("insert", "insert")
                # Several quick pastes can finish out of order. Later paste
                # positions at this same cursor belong after this insertion.
                later_marks = []
                if destination.startswith("attachment_paste_"):
                    sequence = int(destination.split("_")[2])
                    for name in self.input.mark_names():
                        if (name.startswith("attachment_paste_") and int(name.split("_")[2]) > sequence
                                and self.input.compare(name, "==", destination)):
                            self.input.mark_gravity(name, "right")
                            later_marks.append(name)
                self.input.insert(destination, payload["paste"])
                for name in later_marks:
                    self.input.mark_gravity(name, "left")
            for mark in marks.values():
                self.input.mark_unset(mark)
            errors = payload.get("errors", [])
            for error in errors:
                self._append(str(error) + "\n", "error")
            self.status_label.configure(text="Could not attach some files. See the message in chat." if errors
                                        else f"{len(self.pending_attachments)} attachment(s) ready to send." if self.pending_attachments
                                        else "Text pasted. Ready to send.")
            self._update_model_controls()
        elif event == "connected":
            self.connecting = False
            self.base_url = payload["base_url"]
            models = payload["models"]
            self.available_models = list(models)
            self.model_box.configure(values=models)
            self.ready = bool(models)
            preserve_selection = self.preserve_model_selection_on_refresh
            self.preserve_model_selection_on_refresh = False
            requested = self.pending_model_selection
            if requested:
                selected = next((name for name in models if name == requested or name == requested + ":latest"), None)
                if selected:
                    self.model_var.set(selected)
            self.pending_model_selection = None
            if self.model_var.get() not in models and models and not preserve_selection:
                self.model_var.set(DEFAULT_MODEL if DEFAULT_MODEL in models else models[0])
            if preserve_selection and self.model_var.get() not in models:
                self.ready = False
            self.model_info_name = ""
            self.model_tool_support = self.model_thinking_support = self.model_context_support = None
            self.model_vision_support = None
            self.model_context_tokens = None
            self.model_checking = False
            self.model_check_serial += 1
            self.engine_label.configure(text="●  Local engine ready" if models else "●  No local models installed",
                                        fg=COLORS["mint"] if models else COLORS["error"])
            self.retry_button.configure(state="normal")
            self.update_button.configure(state="normal")
            if models or self.settings_expanded:
                self.setup_bar.pack_forget()
            else:
                self.setup_bar.pack(fill="x", padx=30, pady=(12, 0), before=self.transcript_frame)
            if models and not self.workspace.is_dir():
                self.status_label.configure(text="Your previous folder is unavailable. Choose a folder before sending a task.")
            else:
                self.status_label.configure(text="Choose an installed model from the model list to start." if models and not self.ready
                                            else "Ready. Tell your agent what to do." if models else "Download the local model once, then you can work offline.")
            self._update_model_controls()
            self.refresh_hardware()
            self.refresh_model_info()
            self._save_state()
        elif event == "connection_error":
            self.connecting = False
            self.ready = False
            self.engine_label.configure(text="●  Local engine offline", fg=COLORS["error"])
            self.retry_button.configure(state="normal")
            self.update_button.configure(state="normal")
            self.status_label.configure(text="Couldn't connect. Use Refresh connection to try again.")
            self._append("Connection: " + payload["text"] + "\n", "error")
            self._update_model_controls()
        elif event == "hardware_info":
            self.hardware_checking = False
            gpus = payload.get("gpus") or []
            lines = [f"{gpu.get('name', 'GPU')} · {gpu.get('free_mb', 0) / 1024:.1f} / "
                     f"{gpu.get('total_mb', 0) / 1024:.1f} GB VRAM available" for gpu in gpus]
            if os.environ.get("LOCAL_AGENT_GPU_LAYERS") == "0":
                lines.append("CPU launcher active")
            else:
                lines.append("Ollama can split models across compatible GPUs.")
            self.hardware_label.configure(text="\n".join(lines))
        elif event == "hardware_error":
            self.hardware_checking = False
            self.hardware_label.configure(text="Hardware details unavailable. CPU inference remains available.")
        elif event in {"model_info", "model_info_error"}:
            if payload.get("serial") != self.model_check_serial or payload.get("model") != self.model_var.get().strip():
                return
            self.model_checking = False
            self.model_info_name = payload["model"]
            info = payload.get("info") or {}
            capabilities = info.get("capabilities")
            known = info.get("capabilities_known", bool(capabilities)) and isinstance(capabilities, list)
            self.model_tool_support = "tools" in capabilities if known else None
            self.model_thinking_support = "thinking" in capabilities if known else None
            self.model_vision_support = "vision" in capabilities if known else None
            context = info.get("context_length")
            self.model_context_tokens = context if isinstance(context, int) and context > 0 else None
            self.model_context_support = context >= 8192 if isinstance(context, int) and context > 0 else None
            if self.model_thinking_support is False:
                self.think_var.set(False)
            problem = self._model_block_reason()
            if problem:
                description = problem
            elif self.model_tool_support is None:
                description = "Tool support is unknown. This model may not work with file and command tools."
            else:
                description = "Tools supported"
                if self.model_thinking_support is False:
                    description += " · Reason more is unavailable for this model"
                elif self.model_thinking_support:
                    description += " · Reason more supported"
                if isinstance(context, int) and context > 0:
                    description += f" · Up to {min(16384, context):,} context tokens"
                description += " · Reads images" if self.model_vision_support else " · Text only" if self.model_vision_support is False else ""
            self.model_capability_label.configure(text=description, fg=COLORS["error"] if problem else COLORS["muted"])
            if problem or self.model_tool_support is None:
                self.status_label.configure(text=description)
            self._update_model_controls()
            self._save_state()
        elif event == "model_download_status":
            total = payload.get("total", 0)
            completed = payload.get("completed", 0)
            progress = f" {max(0, min(100, int(100 * completed / total)))}%" if total else ""
            self.status_label.configure(text=f"{self.model_operation_label}: {payload.get('status', 'Working')}{progress}")
            self.download_button.configure(text=f"{self.model_operation_label}{progress}…")
        elif event == "model_download_complete":
            self.downloading = False
            self.pending_model_selection = payload.get("model", self.pending_model_selection)
            self.download_button.configure(state="normal", text="Download model (3.3 GB)")
            self.connect()
        elif event == "model_download_error":
            self.downloading = False
            self.pending_model_selection = None
            self.download_button.configure(state="normal", text="Download model (3.3 GB)")
            self.retry_button.configure(state="normal")
            self.update_button.configure(state="normal")
            self.status_label.configure(text=f"{self.model_operation_label} stopped. Check the message in chat, then try again.")
            self._append(payload.get("text", "Model operation failed.") + "\n\n", "error")
            self._update_model_controls()
        elif event == "status":
            self.status_label.configure(text=payload.get("text", ""))
        elif event == "token":
            self._stream_token(payload.get("text", ""))
        elif event == "stream_reset":
            self._discard_stream()
            self.status_label.configure(text=payload.get("text", "Retrying after memory compaction…"))
        elif event == "metrics":
            value = payload.get("tokens_per_second")
            self.last_tokens_per_second = float(value) if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0 else None
        elif event == "memory":
            previous = self.memory_state.get("compactions", 0)
            count = payload.get("compactions", previous + 1)
            count = count if isinstance(count, int) and count > 0 else previous + 1
            first_notice = not self.memory_state or count > previous
            covered = payload.get("covered_messages", 0)
            self.memory_state = {"compactions": count,
                                 "covered_messages": covered if isinstance(covered, int) and covered >= 0 else 0}
            self._refresh_memory_indicator()
            if first_notice:
                self._append("Working memory summarized so the task can continue. Your full chat remains saved.\n\n", "muted")
            self.status_label.configure(text="Working memory summarized. Continuing…")
        elif event == "steering":
            self.queued_count = max(0, self.queued_count - 1)
            self.status_label.configure(text="Your follow-up message has reached the agent.")
        elif event == "plan":
            steps = payload.get("steps", [])
            if steps:
                self._append("PLAN\n", "tool_role")
                symbols = {"completed": "✓", "done": "✓", "in_progress": "→", "pending": "○"}
                for step in steps:
                    self._append(f"{symbols.get(step.get('status'), '○')}  {step.get('text', '')}\n", "muted")
                self._append("\n")
        elif event == "assistant":
            text = payload.get("text", "")
            if not self.stream_started and text:
                self._append("LOCAL AGENT\n", "agent_role")
                self._append(text + "\n\n")
            elif self.stream_started:
                self._finish_stream()
        elif event == "tool":
            self._finish_stream()
            name = payload.get("name", "tool")
            if name in {"ask_user", "set_plan"}:
                return
            arguments = payload.get("arguments", {})
            labels = {"list_files": "LOOKING THROUGH FILES", "read_file": "READING FILE", "write_file": "WRITING FILE",
                      "replace_text": "EDITING FILE", "search_files": "SEARCHING FILES", "run_command": "RUNNING COMMAND",
                      "ask_user": "ASKING A QUESTION", "set_plan": "UPDATING THE PLAN", "launch_python_app": "LAUNCHING APP"}
            self._append(labels.get(name, name.upper()) + "\n", "tool_role")
            if isinstance(arguments, dict):
                preview = arguments.get("command") or arguments.get("path") or arguments.get("query") or arguments.get("question") or "."
            else:
                preview = str(arguments)
            self._append(str(preview)[:1200] + "\n", "tool")
        elif event == "tool_result":
            if payload.get("name") in {"ask_user", "set_plan"}:
                return
            result = payload.get("result", "")
            if not isinstance(result, str):
                result = json.dumps(result, ensure_ascii=False)
            self._append(result[:1800] + ("\n[Output shortened in chat]" if len(result) > 1800 else "") + "\n\n", "tool")
        elif event == "approval":
            self._show_approval(payload)
        elif event == "question":
            self._show_question(payload)
        elif event == "error":
            self._finish_stream()
            self._append(payload.get("text", "Something went wrong.") + "\n\n", "error")
        elif event == "worker_complete":
            self._finish_stream()
            self._answer_approval(False)
            self._answer_question(None)
            self.history = payload.get("history", self.history)
            retry_draft = payload.get("retry_draft")
            if retry_draft and not self.cancel.is_set() and not self.closing:
                self._add_attachments(retry_draft.get("attachments", []))
                if not self.input.get("1.0", "end-1c").strip():
                    self.input.insert("1.0", retry_draft.get("content", ""))
            pending = self._drain_incoming()
            self.queued_count = 0
            if pending and not self.cancel.is_set() and not self.closing:
                self._save_state()
                first, *remaining = pending
                for message in remaining:
                    self.incoming.put(message)
                self.queued_count = len(remaining)
                if isinstance(first, dict):
                    self._start_run(str(first.get("content", "")), attachments=first.get("attachments"), continuation=True)
                else:
                    self._start_run(first, continuation=True)
                return
            if self.work_started is not None:
                self.elapsed_seconds = time.monotonic() - self.work_started
                self.work_started = None
            self._set_busy(False)
            self._save_state()
            self.status_label.configure(text="Stopped. You can continue with another message." if self.cancel.is_set()
                                        else "Attachment request returned to the composer. Check the error above, then retry." if retry_draft
                                        else "Ready for your next message.")
            self.input.focus_set()

    def _poll(self):
        if self.closing:
            return
        try:
            for _ in range(150):
                event, payload, run_id = self.events.get_nowait()
                self._handle_event(event, payload, run_id)
        except queue.Empty:
            pass
        self._poll_id = self.root.after(40, self._poll)

    def close(self):
        if self.closing:
            return
        self.closing = True
        for callback_id in (self._poll_id, self._tick_id):
            self.root.after_cancel(callback_id)
        self.cancel.set()
        self.coding_v2_cancel.set()
        self._answer_approval(False)
        self._answer_question(None)
        self._drain_incoming()
        self.queued_count = 0
        self.status_label.configure(text="Stopping the current task and saving your chat…")
        for widget in (self.send_button, self.stop_button, self.new_button, self.saved_button,
                       self.folder_button, self.retry_button, self.update_button, self.download_button):
            widget.configure(state="disabled")
        self.input.configure(state="disabled")
        self.always_allow_box.configure(state="disabled")
        self.model_box.configure(state="disabled")
        self._update_model_controls()
        self._close_started = time.monotonic()
        self._wait_for_close()

    def _wait_for_close(self):
        if self.coding_v2_worker is not None and self.coding_v2_worker.is_alive():
            self.status_label.configure(text="Finishing the model installer's cleanup and saving your chat…")
            self.root.after(100, self._wait_for_close)
            return
        # Let the agent cancel HTTP reads and kill any running command's process
        # tree before ending Python; its worker is deliberately a daemon.
        if self.worker is not None and self.worker.is_alive() and time.monotonic() - self._close_started < 15:
            self.root.after(100, self._wait_for_close)
            return
        if self.agent is not None and self.busy:
            self.history = copy.deepcopy(self.agent.history)
        self._save_state()
        self.root.destroy()


def main():
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        pass
    root = tk.Tk()
    smoke = "--smoke-test" in sys.argv
    if smoke:
        root.withdraw()
    app = LocalAgentApp(root, start_engine=not smoke)
    if smoke:
        root.update_idletasks()
        root.update()
        app.closing = True
        root.destroy()
        print("Desktop UI smoke test passed.")
    else:
        root.mainloop()


if __name__ == "__main__":
    main()
