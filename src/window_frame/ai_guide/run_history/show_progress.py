"""Live terminal progress, with sparse plain-text progress in saved logs."""
import logging
import shutil
import sys
import time


def format_duration(seconds):
    if seconds is None:
        return 'calculating'
    seconds = max(0, int(seconds))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return (f'{days}d ' if days else '') + (f'{hours}h ' if hours or days else '') + f'{minutes}m {seconds}s'


class Progress:
    def __init__(self, label, logger, total=0, stream=None, log_interval=30):
        self.label, self.logger, self.total = label, logger, total
        self.stream = sys.stderr if stream is None else stream
        self.interactive = self.stream.isatty()
        self.log_interval = log_interval
        self.started = time.monotonic()
        self.last_draw = self.last_log = self.started
        self.done = 0
        self.loss = None
        self.total_remaining = None
        self.closed = False
        self.line_width = 0
        self._log(self._text())
        self.update(0)

    def _text(self, status=None):
        elapsed = time.monotonic() - self.started
        fraction = min(1, self.done / self.total) if self.total else 0
        filled = int(20 * fraction)
        bar = '#' * filled + '-' * (20 - filled)
        eta = elapsed * (self.total - self.done) / self.done if self.done else None
        remaining = f'{eta:.0f}s' if eta is not None else '--'
        text = (f'{self.label} [{bar}] {fraction:5.1%} {self.done}/{self.total or "?"}'
                f' | elapsed {elapsed:.0f}s | ETA {remaining}')
        if self.loss is not None:
            text += f' | loss {self.loss:.4f}'
        if self.total_remaining is not None:
            text += f' | all epochs ~{format_duration(self.total_remaining)}'
        if status:
            text += f' | {status}'
        return text

    def _log(self, text):
        if not self.interactive:
            self.logger.info(text)
            return
        # Save milestones without sending a duplicate line to the terminal.
        record = self.logger.makeRecord(self.logger.name, logging.INFO, __file__, 0, text, (), None)
        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.handle(record)

    def _draw(self, text, newline=False):
        width = max(20, shutil.get_terminal_size((120, 20)).columns - 1)
        # Don't let a long progress line wrap and leave stale bars on screen.
        if len(text) > width:
            text = text.replace('elapsed ', '').replace(' | ', ' ').replace('loss ', 'L=')
            text = text.replace('[' + '#' * 20 + ']', '[done]') if self.done == self.total else text
        if len(text) > width:
            text = text[:width // 2 - 2] + '...' + text[-(width - width // 2 - 1):]
        self.stream.write('\r' + text.ljust(min(self.line_width, width)) + ('\n' if newline else ''))
        self.stream.flush()
        self.line_width = len(text)

    def update(self, done, total=None, loss=None, remaining_seconds=None):
        self.done = done
        if remaining_seconds is not None:
            self.total_remaining = remaining_seconds
        if total is not None:
            self.total = total
        if loss is not None:
            self.loss = loss
        now = time.monotonic()
        if self.interactive and (now - self.last_draw >= .2 or self.done in (0, self.total)):
            self._draw(self._text())
            self.last_draw = now
        if now - self.last_log >= self.log_interval:
            self._log(self._text())
            self.last_log = now

    def close(self, status='done'):
        if self.closed:
            return
        self.closed = True
        text = self._text(status)
        if self.interactive:
            self._draw(text, newline=True)
        self._log(text)

    def __enter__(self):
        return self

    def __exit__(self, error_type, error, traceback):
        self.close('stopped' if error_type else 'done')
