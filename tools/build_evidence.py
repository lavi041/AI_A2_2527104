"""Build evidence/AI_A2_2527104_TEST_LOG.pdf (real pytest output) and ..._DECLARATION.pdf.

Run from the project root:  python tools/build_evidence.py
"""
from __future__ import annotations

import datetime as dt
import platform
import subprocess
import sys
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Preformatted

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "evidence"
NAME, REG, USER, CODE = "Aisha Niyonsaba", "25/27104", "ug2527104@ines.ac.rw", "AI-A2-7104"
ss = getSampleStyleSheet()
MONO = ParagraphStyle("m", fontName="Courier", fontSize=7.6, leading=9.4)


def run(cmd):
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return (r.stdout + r.stderr).rstrip(), r.returncode


def test_log():
    out_q, rc_q = run([sys.executable, "-m", "pytest", "-q"])
    out_v, rc_v = run([sys.executable, "-m", "pytest", "-v", "-p", "no:cacheprovider"])
    stamp = dt.datetime.now().strftime("%A %d %B %Y, %H:%M:%S")
    story = [Paragraph(f"Test log - {CODE}", ss["Title"]),
             Paragraph(f"{NAME} | Reg. {REG} | {USER}", ss["Normal"]),
             Paragraph(f"Generated: {stamp} (machine local time) | Python {platform.python_version()} | {platform.system()} {platform.release()}", ss["Normal"]),
             Spacer(1, 8), Paragraph("$ pytest -q", ss["Heading3"]), Preformatted(out_q, MONO),
             Paragraph(f"exit code: {rc_q}", ss["Normal"]), Spacer(1, 8),
             Paragraph("$ pytest -v", ss["Heading3"]), Preformatted(out_v, MONO),
             Paragraph(f"exit code: {rc_v}", ss["Normal"])]
    SimpleDocTemplate(str(EV / "AI_A2_2527104_TEST_LOG.pdf"), pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm,
                      title=f"Test log {CODE}", author=NAME).build(story)
    print("test log written; exit codes", rc_q, rc_v)


def declaration():
    B = ParagraphStyle("b", parent=ss["BodyText"], fontSize=10.5, leading=15, spaceAfter=6)
    items = [
        "This submission is my own individual work for SWE 3513, Individual Assignment 2.",
        f"My dataset, seed (7104), verification code ({CODE}), metrics, plots and model behaviour were produced by running my own code with my own registration digits.",
        "The dataset is synthetic. I have not used real farmer data, and I have not edited the generated CSV by hand.",
        "I used an AI assistant (Claude, Anthropic) to help generate code, tests, documentation and a first draft of the report. This use is fully disclosed in evidence/AI_USE.md.",
        "I have read, understood and tested the submitted code and I can explain it and make a small live modification during the defence.",
        "I wrote or rewrote in my own words the interpretation of one regression error, one classification error and one farm cluster.",
        "I have not shared my seed-specific files or fingerprint with other students and I did not copy another student's work.",
        "I understand that mismatched seeds, copied fingerprints, identical explanations or unexplained code may be referred for academic integrity review.",
    ]
    story = [Paragraph("Student declaration", ss["Title"]),
             Paragraph("Institut d'Enseignement Superieur de Ruhengeri - SWE 3513 Artificial Intelligence", ss["Normal"]),
             Spacer(1, 10),
             Paragraph(f"<b>Name:</b> {NAME}<br/><b>Registration number:</b> {REG}<br/><b>Institutional username:</b> {USER}<br/>"
                       f"<b>Verification code:</b> {CODE}", B), Spacer(1, 6),
             Paragraph("I declare that:", B)]
    story += [Paragraph(f"{i}. {escape(t)}", B) for i, t in enumerate(items, 1)]
    story += [Spacer(1, 30), Paragraph("Signature: ______________________________", B),
              Paragraph("Date: ______________________________", B)]
    SimpleDocTemplate(str(EV / "AI_A2_2527104_DECLARATION.pdf"), pagesize=A4, title=f"Declaration {CODE}", author=NAME).build(story)
    print("declaration written")


if __name__ == "__main__":
    test_log()
    declaration()
