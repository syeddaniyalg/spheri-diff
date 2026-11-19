from pathlib import Path


class LogStream:
    def __init__(self, path):
        self.path = Path(path)
        self.offset = 0
        self.pending = ""

    def read_new_text(self):
        if not self.path.exists():
            return ""
        with open(self.path, "rb") as handle:
            handle.seek(self.offset)
            data = handle.read()
        self.offset += len(data)
        return data.decode("utf-8", errors="replace")

    def lines(self, text):
        return [line.strip() for line in text.replace("\r", "\n").split("\n") if line.strip()]

    def read(self):
        text = self.pending + self.read_new_text()
        cut = max(text.rfind("\n"), text.rfind("\r")) + 1
        self.pending = text[cut:]
        return self.lines(text[:cut])

    def flush(self):
        remainder, self.pending = self.pending, ""
        return self.lines(remainder)


def format_report(name, gpu, seconds, line):
    return f"[{seconds / 60:7.1f} min] [gpu {gpu}] [{name}] {line}"
