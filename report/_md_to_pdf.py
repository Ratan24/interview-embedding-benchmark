"""Render a markdown file to a styled, print-ready HTML, then to PDF
via headless Chrome.

Usage:
    python3 _md_to_pdf.py <input.md> <output.pdf>
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import markdown


CSS = """
@page {
    size: Letter;
    margin: 0.85in 0.95in;
}
* { box-sizing: border-box; }
html, body {
    font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue",
                 "Segoe UI", Helvetica, Arial, sans-serif;
    font-size: 11pt;
    color: #1a1a1a;
    line-height: 1.55;
    margin: 0;
    padding: 0;
    background: #ffffff;
}
h1 {
    font-size: 22pt;
    font-weight: 700;
    color: #0f1f3a;
    margin: 0 0 0.15em 0;
    padding-bottom: 0.25em;
    border-bottom: 2px solid #0f1f3a;
    line-height: 1.2;
}
h1 + h3 {
    margin-top: 0.1em;
    color: #4a4a4a;
    font-weight: 500;
    font-size: 13pt;
    border-bottom: none;
}
h2 {
    font-size: 15pt;
    font-weight: 700;
    color: #0f1f3a;
    margin: 1.6em 0 0.5em 0;
    padding-bottom: 0.2em;
    border-bottom: 1px solid #d0d4dc;
    page-break-after: avoid;
}
h3 {
    font-size: 12.5pt;
    font-weight: 600;
    color: #1f3a5f;
    margin: 1.3em 0 0.4em 0;
    page-break-after: avoid;
}
p {
    margin: 0 0 0.7em 0;
    text-align: left;
}
strong { color: #0f1f3a; }
em { color: #333; }
ul, ol { margin: 0.4em 0 0.9em 1.2em; padding: 0; }
li { margin: 0.15em 0; }
blockquote {
    margin: 0.9em 0;
    padding: 0.6em 1em;
    border-left: 3px solid #1f3a5f;
    background: #f4f6fa;
    color: #1a2438;
    font-style: italic;
    page-break-inside: avoid;
}
hr {
    border: none;
    border-top: 1px solid #d0d4dc;
    margin: 1.6em 0;
}
code {
    font-family: "SF Mono", Menlo, Consolas, monospace;
    font-size: 9.5pt;
    background: #f1f3f7;
    padding: 1px 5px;
    border-radius: 3px;
    color: #1a2438;
}
pre {
    background: #f1f3f7;
    border: 1px solid #d8dde6;
    border-radius: 4px;
    padding: 0.7em 0.9em;
    font-size: 9pt;
    line-height: 1.4;
    overflow-x: auto;
    page-break-inside: avoid;
}
pre code {
    background: transparent;
    padding: 0;
    border-radius: 0;
}
table {
    border-collapse: collapse;
    width: 100%;
    margin: 0.7em 0 1em 0;
    font-size: 10pt;
    page-break-inside: avoid;
}
thead {
    background: #1f3a5f;
    color: #ffffff;
}
th, td {
    text-align: left;
    padding: 0.45em 0.7em;
    border: 1px solid #d0d4dc;
    vertical-align: top;
}
th { font-weight: 600; letter-spacing: 0.01em; }
tbody tr:nth-child(even) { background: #f5f7fb; }
img {
    display: block;
    max-width: 100%;
    margin: 0.8em auto 0.4em auto;
    page-break-inside: avoid;
    page-break-before: auto;
    page-break-after: auto;
}
a { color: #1f3a5f; text-decoration: none; border-bottom: 1px solid #b8c2d4; }
a:hover { color: #0f1f3a; }
.meta {
    color: #555;
    font-size: 10pt;
    margin-bottom: 0.4em;
}
"""

HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>{css}</style>
</head>
<body>
{body}
</body>
</html>
"""


def absolutise_image_paths(html, md_path):
    """Rewrite <img src="../figures/x.png"> to file:// absolute paths
    so headless Chrome can find them when rendering from a temp file."""
    base = md_path.resolve().parent

    def repl(match):
        src = match.group(1)
        if src.startswith(("http://", "https://", "data:", "file://")):
            return match.group(0)
        abs_path = (base / src).resolve()
        return f'src="file://{abs_path}"'

    return re.sub(r'src="([^"]+)"', repl, html)


def render_html(md_path):
    text = md_path.read_text(encoding="utf-8")
    body = markdown.markdown(
        text,
        extensions=[
            "tables",
            "fenced_code",
            "sane_lists",
            "attr_list",
            "toc",
        ],
        output_format="html5",
    )
    body = absolutise_image_paths(body, md_path)
    title = md_path.stem
    return HTML_TEMPLATE.format(title=title, css=CSS, body=body)


def find_chrome():
    """Locate a Chrome/Chromium binary across macOS, Linux, and Windows.

    The CHROME_BINARY env var, if set, takes precedence.
    """
    override = os.environ.get("CHROME_BINARY")
    if override:
        if Path(override).exists():
            return override
        raise FileNotFoundError(
            f"CHROME_BINARY={override!r} does not exist."
        )

    candidates = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ]
    for cmd in (
        "google-chrome", "google-chrome-stable",
        "chromium", "chromium-browser", "chrome",
    ):
        path = shutil.which(cmd)
        if path:
            candidates.append(path)
    for c in candidates:
        if c and Path(c).exists():
            return c
    raise RuntimeError(
        "Could not locate a Chrome or Chromium binary.  Install one, "
        "or set the CHROME_BINARY environment variable to its path."
    )


def html_to_pdf(html_path, pdf_path, timeout_sec=120):
    chrome = find_chrome()
    cmd = [
        chrome,
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        "--no-margins",
        "--virtual-time-budget=10000",
        f"--print-to-pdf={pdf_path}",
        f"file://{html_path}",
    ]
    subprocess.run(
        cmd, check=True, capture_output=True, timeout=timeout_sec
    )


def main(md_arg, pdf_arg):
    md_path = Path(md_arg).resolve()
    pdf_path = Path(pdf_arg).resolve()
    html = render_html(md_path)
    html_path = md_path.with_suffix(".html")
    html_path.write_text(html, encoding="utf-8")
    html_to_pdf(html_path, pdf_path)
    size_kb = pdf_path.stat().st_size / 1024
    print(f"OK  {pdf_path}  ({size_kb:.0f} KB)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
