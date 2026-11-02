"""Blind ETT reading tool: a thin tkinter window over ``tools/reading_core.py``.

Run from the repository root (PROTOCOL.md section 2; docs/ETT_reference_sheet.md):

    python tools/read_tool.py --practice --list items/practice_list.csv
    python tools/read_tool.py --practice                      (synthetic demo, keys only)
    python tools/read_tool.py --list items/reading_list.csv --out items/verified_items.csv
    python tools/read_tool.py --mode retest --list items/retest_list.csv --out items/retest.csv \
        --first-read items/verified_items.csv

Keys: P present, A absent, S skip; 1 ambiguous, 2 tracheostomy, 3 out of field (toggle);
N note; Enter confirm (and, after the last image, finalise); Backspace back one image;
Z 100% view with scrollbars (drag or scroll to move); C contrast (original, autocontrast,
equalised). The window shows "k of N" and the read_id only: never a filename, a path or
anything about the arm. Every confirmed answer is autosaved; closing the window and
running the same command again resumes where you stopped.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import tkinter as tk  # noqa: E402
from tkinter import messagebox, simpledialog  # noqa: E402

from PIL import Image, ImageTk  # noqa: E402

from tools import reading_core as rc  # noqa: E402

KEY_HELP = (
    "P present   A absent   S skip   |   1 ambiguous   2 tracheostomy   3 out of field   |   "
    "N note   Enter confirm   Backspace back one   |   Z 100% view   C contrast"
)
FLAG_LABELS = {"ambiguous": "1 ambiguous", "tracheostomy": "2 tracheostomy", "out_of_field": "3 out of field"}

# keysym (lower case) -> (action, argument)
KEY_ACTIONS: dict[str, tuple[str, str | None]] = {
    "p": ("select", "present"), "a": ("select", "absent"), "s": ("select", "skip"),
    "1": ("flag", "ambiguous"), "2": ("flag", "tracheostomy"), "3": ("flag", "out_of_field"),
    "kp_1": ("flag", "ambiguous"), "kp_2": ("flag", "tracheostomy"), "kp_3": ("flag", "out_of_field"),
    "kp_end": ("flag", "ambiguous"), "kp_down": ("flag", "tracheostomy"), "kp_next": ("flag", "out_of_field"),
    "n": ("note", None), "backspace": ("back", None), "z": ("zoom", None), "c": ("contrast", None),
    "return": ("confirm", None), "kp_enter": ("confirm", None),
}
_CONTROL_MASK = 0x0004


class TkDialogs:
    """Modal dialogs; tests replace this object with a fake."""

    def __init__(self, root: tk.Misc) -> None:
        self.root = root

    def ask_yes_no(self, title: str, message: str) -> bool:
        return bool(messagebox.askyesno(title, message, parent=self.root))

    def ask_ok_cancel(self, title: str, message: str) -> bool:
        return bool(messagebox.askokcancel(title, message, parent=self.root))

    def ask_string(self, title: str, prompt: str, initial: str) -> str | None:
        return simpledialog.askstring(title, prompt, initialvalue=initial, parent=self.root)

    def info(self, title: str, message: str) -> None:
        messagebox.showinfo(title, message, parent=self.root)

    def error(self, title: str, message: str) -> None:
        messagebox.showerror(title, message, parent=self.root)


class ReaderApp:
    """The reading window. All decisions are delegated to ``ReadingSession``; this class
    only draws ``session.view()`` and the current image and maps keys to session calls.
    ``root`` is the program's ``tk.Tk`` (or a ``tk.Toplevel``, as in the tests); it is
    destroyed when the reading is finalised or the reader quits."""

    BG = "#000000"
    PANEL = "#1e1e1e"
    FG = "#e6e6e6"
    DIM = "#8c8c8c"
    ON_BG = "#ffd166"
    ON_FG = "#000000"
    ERR = "#ff8c8c"

    def __init__(
        self,
        root: tk.Tk | tk.Toplevel,
        session: rc.ReadingSession,
        *,
        plan: rc.RunPlan | None = None,
        dialogs: object | None = None,
    ) -> None:
        self.root = root
        self.session = session
        self.plan = plan
        self.dialogs = dialogs if dialogs is not None else TkDialogs(root)
        self.contrast = rc.CONTRAST_PRESETS[0]
        self.zoom_100 = False
        self.finished = False
        self.result: rc.FinaliseResult | None = None
        self._base_image: Image.Image | None = None
        self._image_error: str | None = None
        self._contrast_cache: dict[str, Image.Image] = {}
        self._photo: ImageTk.PhotoImage | None = None
        self._resize_job: str | None = None
        self._build()
        self.refresh(reload_image=True)

    # ------------------------------------------------------------ layout
    def _build(self) -> None:
        root = self.root
        title = "Blind ETT read"
        if self.plan is not None and self.plan.practice_dir is not None:
            title += " - PRACTICE"
        if self.session.mode == "retest":
            title += " - RETEST"
        root.title(title)
        root.configure(bg=self.BG)
        width = min(1400, max(800, root.winfo_screenwidth() - 80))
        height = min(1300, max(700, root.winfo_screenheight() - 120))
        root.geometry(f"{width}x{height}")

        menubar = tk.Menu(root)
        session_menu = tk.Menu(menubar, tearoff=0)
        session_menu.add_command(label="Mark all unanswered images as skip...", command=self.do_skip_remaining)
        session_menu.add_separator()
        session_menu.add_command(label="Quit (answers so far are saved)", command=self.on_close)
        menubar.add_cascade(label="Session", menu=session_menu)
        root.config(menu=menubar)

        top = tk.Frame(root, bg=self.PANEL)
        top.pack(side="top", fill="x")
        self.progress_var = tk.StringVar(master=root)
        self.read_id_var = tk.StringVar(master=root)
        self.display_var = tk.StringVar(master=root)
        tk.Label(top, textvariable=self.progress_var, bg=self.PANEL, fg=self.FG,
                 font=("Segoe UI", 16, "bold")).pack(side="left", padx=12, pady=6)
        tk.Label(top, textvariable=self.read_id_var, bg=self.PANEL, fg=self.FG,
                 font=("Consolas", 14)).pack(side="left", padx=12)
        tk.Label(top, textvariable=self.display_var, bg=self.PANEL, fg=self.DIM,
                 font=("Segoe UI", 11)).pack(side="right", padx=12)

        help_bar = tk.Label(root, text=KEY_HELP, bg=self.PANEL, fg=self.DIM, font=("Segoe UI", 10), anchor="w")
        help_bar.pack(side="bottom", fill="x", ipady=3)
        self.message_var = tk.StringVar(master=root)
        self.message_label = tk.Label(root, textvariable=self.message_var, bg=self.PANEL, fg=self.FG,
                                      font=("Segoe UI", 11), anchor="w")
        self.message_label.pack(side="bottom", fill="x", ipady=3)
        answer_bar = tk.Frame(root, bg=self.PANEL)
        answer_bar.pack(side="bottom", fill="x")
        self.answer_var = tk.StringVar(master=root)
        tk.Label(answer_bar, textvariable=self.answer_var, bg=self.PANEL, fg=self.FG, width=34, anchor="w",
                 font=("Segoe UI", 14, "bold")).pack(side="left", padx=12, pady=6)
        self.flag_labels: dict[str, tk.Label] = {}
        for flag in rc.FLAGS:
            label = tk.Label(answer_bar, text=FLAG_LABELS[flag], bg=self.PANEL, fg=self.DIM,
                             font=("Segoe UI", 12), padx=8, pady=2)
            label.pack(side="left", padx=4)
            self.flag_labels[flag] = label
        self.note_var = tk.StringVar(master=root)
        tk.Label(answer_bar, textvariable=self.note_var, bg=self.PANEL, fg=self.FG, anchor="w",
                 font=("Segoe UI", 11)).pack(side="left", padx=12, fill="x", expand=True)

        frame = tk.Frame(root, bg=self.BG)
        frame.pack(side="top", fill="both", expand=True)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(frame, bg=self.BG, highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vbar = tk.Scrollbar(frame, orient="vertical", command=self.canvas.yview)
        self.hbar = tk.Scrollbar(frame, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(xscrollcommand=self.hbar.set, yscrollcommand=self.vbar.set)

        root.bind("<KeyPress>", self._on_key)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Shift-MouseWheel>", self._on_shift_wheel)
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ------------------------------------------------------------ keys
    def _on_key(self, event: tk.Event) -> None:
        if int(getattr(event, "state", 0)) & _CONTROL_MASK:
            return
        self.handle_key(event.keysym)

    def handle_key(self, keysym: str) -> bool:
        """Run the action bound to ``keysym``; returns False for keys without an action."""
        action = KEY_ACTIONS.get(keysym.lower())
        if action is None:
            return False
        if self.finished:
            return True
        name, argument = action
        handler = getattr(self, f"do_{name}")
        if argument is None:
            handler()
        else:
            handler(argument)
        return True

    # ------------------------------------------------------------ actions
    def set_message(self, text: str, error: bool = False) -> None:
        self.message_var.set(text)
        self.message_label.configure(fg=self.ERR if error else self.FG)

    def do_select(self, response: str) -> None:
        try:
            self.session.select(response)
            self.set_message("")
        except rc.ReadingError as exc:
            self.set_message(str(exc), error=True)
        self.refresh()

    def do_flag(self, flag: str) -> None:
        try:
            self.session.toggle_flag(flag)
            self.set_message("")
        except rc.ReadingError as exc:
            self.set_message(str(exc), error=True)
        self.refresh()

    def do_note(self) -> None:
        view = self.session.view()
        if view.complete or view.finalised:
            return
        text = self.dialogs.ask_string("Note", f"Note for {view.read_id} (optional, one line):", view.note)
        if text is not None:
            self.session.set_note(text)
        self.refresh()

    def do_confirm(self) -> None:
        view = self.session.view()
        if view.complete:
            self.do_finalise()
            return
        try:
            self.session.confirm()
        except rc.ReadingError as exc:
            # The session may already have moved on (the answer is kept in memory when only the
            # autosave failed), so reload the image whenever the position changed.
            moved = self.session.position != view.position
            if isinstance(exc, (rc.AutosaveError, rc.ConcurrentWriteError)):
                self.dialogs.error("Autosave failed", str(exc))
                self.set_message("Answer kept in memory; saving is retried with the next answer.", error=True)
            else:
                self.set_message(str(exc), error=True)
            self.refresh(reload_image=moved)
            return
        if self.session.complete:
            self.set_message("All images answered. Press Enter to finalise and save, or Backspace to go back one image.")
        else:
            self.set_message(f"Image {view.position + 1} saved.")
        self.refresh(reload_image=True)

    def do_back(self) -> None:
        if self.session.back():
            self.set_message("Back one image: change the answer if needed, then press Enter.")
            self.refresh(reload_image=True)
        elif self.session.position == 0:
            self.set_message("This is the first image.", error=True)
        else:
            self.set_message("You can go back one image only; press Enter to confirm this one.", error=True)

    def do_zoom(self) -> None:
        self.zoom_100 = not self.zoom_100
        self.refresh()
        if self.zoom_100 and self._base_image is not None:
            width = self._base_image.width
            visible = max(1, self.canvas.winfo_width())
            self.canvas.xview_moveto(max(0.0, (width - visible) / 2.0 / width))
            self.canvas.yview_moveto(0.0)

    def do_contrast(self) -> None:
        self.contrast = rc.next_contrast(self.contrast)
        self.refresh()

    def do_skip_remaining(self) -> None:
        if self.finished:
            return
        remaining = len(self.session.unanswered())
        if remaining == 0:
            self.set_message("Every image already has an answer.")
            return
        if not self.dialogs.ask_yes_no(
            "End early",
            f"Mark the {remaining} unanswered images as skip? Skipped images are excluded from the "
            "analysis. This cannot be undone.",
        ):
            return
        try:
            self.session.skip_remaining()
            self.set_message("Remaining images marked as skip. Press Enter to finalise and save.")
        except rc.ReadingError as exc:
            self.dialogs.error("Autosave failed" if isinstance(exc, rc.AutosaveError) else "Error", str(exc))
            self.set_message(str(exc), error=True)
        self.refresh(reload_image=True)

    def do_finalise(self) -> None:
        if not self.dialogs.ask_yes_no(
            "Finalise",
            "Save the final file now? It is fingerprinted (SHA-256) and cannot be changed afterwards.",
        ):
            return
        try:
            result = self.session.finalise()
        except (rc.ReadingError, OSError) as exc:
            self.dialogs.error("Cannot finalise", str(exc))
            return
        self.result = result
        lines = result.summary_lines()
        if self.plan is not None and self.plan.practice_dir is not None:
            lines.append(f"Practice folder (not part of the study): {self.plan.practice_dir}")
        print("\n".join(lines))
        self.dialogs.info("Saved", "\n".join(lines))
        self.finished = True
        self.root.destroy()

    def on_close(self) -> None:
        if self.finished or self.session.finalised:
            self.finished = True
            self.root.destroy()
            return
        if self.dialogs.ask_yes_no(
            "Quit",
            "Every confirmed answer is saved in the autosave file. Quit now? "
            "Run the same command again to resume.",
        ):
            self.finished = True
            self.root.destroy()

    # ------------------------------------------------------------ drawing
    def refresh(self, reload_image: bool = False) -> None:
        """Redraw the status texts and the image from ``session.view()``."""
        view = self.session.view()
        if reload_image:
            self._load_image()
            self.zoom_100 = False
        self.progress_var.set(view.progress)
        self.read_id_var.set(f"read_id {view.read_id}" if view.read_id else "")
        self.display_var.set(f"contrast: {self.contrast}   view: {'100%' if self.zoom_100 else 'fit'}")
        if view.complete:
            self.answer_var.set("")
            self.note_var.set("")
        else:
            answer = view.response.upper() if view.response else "-"
            suffix = "  (previous image)" if view.stepped_back else ""
            self.answer_var.set(f"Answer: {answer}{suffix}")
            self.note_var.set(f"Note: {view.note}" if view.note else "")
        for flag, label in self.flag_labels.items():
            on = flag in view.flags and not view.complete
            label.configure(bg=self.ON_BG if on else self.PANEL, fg=self.ON_FG if on else self.DIM)
        self._render(view)

    def _load_image(self) -> None:
        self._base_image, self._image_error = None, None
        self._contrast_cache = {}
        if self.session.complete:
            return
        try:
            self._base_image = self.session.load_image()
        except rc.ImageUnavailableError as exc:
            self._image_error = str(exc)

    def _shown_image(self) -> Image.Image | None:
        if self._base_image is None:
            return None
        if self.contrast not in self._contrast_cache:
            self._contrast_cache[self.contrast] = rc.apply_contrast(self._base_image, self.contrast)
        return self._contrast_cache[self.contrast]

    def _show_scrollbars(self, show: bool) -> None:
        if show:
            self.vbar.grid(row=0, column=1, sticky="ns")
            self.hbar.grid(row=1, column=0, sticky="ew")
        else:
            self.vbar.grid_remove()
            self.hbar.grid_remove()

    def _centre_text(self, text: str, colour: str | None = None) -> None:
        width, height = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        self.canvas.configure(scrollregion=(0, 0, width, height))
        self.canvas.create_text(width // 2, height // 2, text=text, fill=colour or self.FG,
                                font=("Segoe UI", 16), justify="center", width=max(200, width - 80))

    def _render(self, view: rc.ViewState) -> None:
        self.canvas.delete("all")
        self._photo = None
        if view.complete:
            self._show_scrollbars(False)
            self._centre_text(
                f"All {view.n} images answered.\n\nEnter: finalise and save the file.\n"
                "Backspace: go back one image."
            )
            return
        image = self._shown_image()
        if image is None:
            self._show_scrollbars(False)
            problem = self._image_error or view.image_problem or "image unavailable"
            self._centre_text(
                f"Image unavailable for {view.read_id} ({problem}).\n\nPress S, then Enter, to skip it.",
                colour=self.ERR,
            )
            return
        if self.zoom_100:
            self._show_scrollbars(True)
            self.canvas.configure(scrollregion=(0, 0, image.width, image.height))
            self._photo = ImageTk.PhotoImage(image, master=self.root)
            self.canvas.create_image(0, 0, anchor="nw", image=self._photo)
        else:
            self._show_scrollbars(False)
            width, height = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
            self.canvas.configure(scrollregion=(0, 0, width, height))
            self._photo = ImageTk.PhotoImage(rc.fit_image(image, width, height), master=self.root)
            self.canvas.create_image(width // 2, height // 2, anchor="center", image=self._photo)

    # ------------------------------------------------------------ canvas events
    def _on_canvas_configure(self, _event: tk.Event) -> None:
        if self.zoom_100 or self.finished:
            return
        if self._resize_job is not None:
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(80, self._redraw_after_resize)

    def _redraw_after_resize(self) -> None:
        self._resize_job = None
        if not self.finished:
            self._render(self.session.view())

    @staticmethod
    def _wheel_units(delta: int) -> int:
        """Scroll units for a mouse-wheel delta (Windows reports multiples of 120)."""
        if delta == 0:
            return 0
        return int(-delta / 120) or (-1 if delta > 0 else 1)

    def _on_wheel(self, event: tk.Event) -> None:
        units = self._wheel_units(int(event.delta))
        if self.zoom_100 and units:
            self.canvas.yview_scroll(units, "units")

    def _on_shift_wheel(self, event: tk.Event) -> None:
        units = self._wheel_units(int(event.delta))
        if self.zoom_100 and units:
            self.canvas.xview_scroll(units, "units")

    def _on_press(self, event: tk.Event) -> None:
        if self.zoom_100:
            self.canvas.scan_mark(event.x, event.y)

    def _on_drag(self, event: tk.Event) -> None:
        if self.zoom_100:
            self.canvas.scan_dragto(event.x, event.y, gain=1)


# ---------------------------------------------------------------- command line
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse --list, --out, --mode, --practice and --first-read."""
    parser = argparse.ArgumentParser(
        prog="read_tool.py",
        description="Blind ETT reading tool (PROTOCOL.md section 2). Shows one image at a time "
                    "with 'k of N' and the read_id only.",
    )
    parser.add_argument("--list", type=Path, help="reading list CSV (read_id, nih_filename, image_path)")
    parser.add_argument("--out", type=Path, help="output CSV, e.g. items/verified_items.csv (must not exist yet)")
    parser.add_argument("--mode", choices=rc.MODES, default="read", help="read (default) or retest")
    parser.add_argument("--practice", action="store_true",
                        help="practice run: the first 10 images of --list (or 10 synthetic demo images "
                             "without --list); answers go to a new temporary folder")
    parser.add_argument("--first-read", type=Path, default=None,
                        help="retest mode: the finalised first-read output (verified_items.csv); the tool "
                             "refuses to start earlier than 48 h after it was saved")
    args = parser.parse_args(argv)
    if not args.practice and (args.list is None or args.out is None):
        parser.error("--list and --out are required unless --practice is given")
    if args.first_read is not None and args.mode != "retest":
        parser.error("--first-read applies to --mode retest only")
    return args


def _enable_dpi_awareness() -> None:
    """On Windows, ask for per-monitor DPI awareness so that the 100% view is 1:1."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:  # older Windows or no shcore: fall back to the system-aware call
        try:
            import ctypes

            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def _startup_error(message: str) -> None:
    print(f"read_tool: {message}", file=sys.stderr)
    try:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Blind ETT read", message, parent=root)
        root.destroy()
    except tk.TclError:
        pass


def main(argv: list[str] | None = None) -> int:
    """Run the tool. Exit codes: 0 finalised, 1 closed before finalising (answers saved
    in the autosave file), 2 could not start."""
    args = parse_args(argv)
    try:
        plan = rc.plan_run(args.list, args.out, args.mode, practice=args.practice, first_read=args.first_read)
        session = rc.ReadingSession(plan.list_path, plan.out_path, plan.mode, limit=plan.limit)
    except (rc.ReadingError, ValueError, OSError) as exc:
        _startup_error(str(exc))
        return 2
    _enable_dpi_awareness()
    root = tk.Tk()
    root.withdraw()
    app = ReaderApp(root, session, plan=plan)
    text = "\n".join([*plan.messages, *session.summary_lines(), "", KEY_HELP.replace("   |   ", "\n"),
                      "", "Press OK to start."])
    if not app.dialogs.ask_ok_cancel("Blind ETT read", text):
        root.destroy()
        return 1
    root.deiconify()
    root.focus_force()
    session.restart_timer()
    root.mainloop()
    return 0 if session.finalised else 1


if __name__ == "__main__":
    sys.exit(main())
