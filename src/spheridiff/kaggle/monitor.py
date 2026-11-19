from pathlib import Path


class LogTail:
    def __init__(self, path):
        self.path = Path(path)

    def text(self):
        if not self.path.exists():
            return ""
        return self.path.read_bytes()[-20000:].decode("utf-8", errors="replace")

    def last_line(self):
        lines = self.text().replace("\r", "\n").splitlines()
        lines = [line.strip() for line in lines if line.strip()]
        return lines[-1] if lines else ""


def format_report(name, gpu, seconds, line):
    return f"[{seconds / 60:7.1f} min] [gpu {gpu}] [{name}] {line[:170]}"
