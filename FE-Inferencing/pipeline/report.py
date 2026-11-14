"""End-user crack report: a labelled facade image + a table mapping each label
to its dimensions, as a PDF inspection report (A4 landscape, page numbers),
the same content as one self-contained HTML file, and the table as CSV.

PDF: WeasyPrint renders the HTML, so the two never drift apart. It needs
Pango (macOS: `brew install pango`); without it the PDF is skipped with a
warning and HTML/CSV are still written.

Crack #N is the same crack in the image, the table and the CSV: all three
use quantify.review_order (on-wall first, widest first). Outline colour =
BRE Digest 251 severity for on-wall cracks; off-wall detections (aircon,
ledges, ground, by 3D depth) get their own colour so they read as "review".
Each table row carries a close-up thumbnail, because at facade scale the
labels of hundreds of cracks are too small to find by eye.
"""
import base64
import csv
import html
import os
import sys
from datetime import date
from pathlib import Path

import cv2
import numpy as np

from quantify import IMPLAUSIBLE_WIDTH_MM, crack_table, review_order, scale_note

# BGR outline colours by BRE category (on-wall); off-wall separate
SEVERITY_BGR = {0: (60, 170, 60), 1: (60, 170, 60), 2: (0, 200, 240), 3: (0, 128, 255), 4: (40, 40, 230),
                5: (40, 40, 230), None: (150, 150, 150)}
OFFWALL_BGR = (200, 60, 200)
REPORT_IMG_W = 2400   # embedded facade width (px): legible on screen, keeps the HTML a few MB
THUMB_PX = 150


def _colour(row: dict, loc: str) -> tuple[int, int, int]:
    return OFFWALL_BGR if loc != "wall" else SEVERITY_BGR.get(row["bre_category"], SEVERITY_BGR[None])


def draw_labelled(image: np.ndarray, found: list) -> np.ndarray:
    """Outline every crack in its severity colour and stamp its crack # (white box, coloured text).
    Text is sized for the REPORT_IMG_W-wide embedded copy, so it stays readable after downscaling."""
    out = image.copy()
    k = image.shape[1] / REPORT_IMG_W
    thick, fs = max(2, round(2 * k)), 0.55 * k
    for i, (c, r, loc) in enumerate(review_order(found)):
        col = _colour(r, loc)
        cv2.polylines(out, [c["contour"].astype(np.int32).reshape(-1, 1, 2)], True, col, thick)
        label = str(i + 1)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, thick)
        x, y = int(c["bbox"][0]), max(int(c["bbox"][1]) - 4, th + 4)
        cv2.rectangle(out, (x, y - th - 4), (x + tw + 4, y + 2), (255, 255, 255), -1)
        cv2.putText(out, label, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, fs, col, thick)
    return out


def _thumb(image: np.ndarray, c: dict, col) -> np.ndarray:
    x1, y1, x2, y2 = (int(v) for v in c["bbox"])
    m = max(40, (x2 - x1 + y2 - y1) // 4)
    X1, Y1 = max(x1 - m, 0), max(y1 - m, 0)
    crop = image[Y1:y2 + m, X1:x2 + m].copy()
    cv2.polylines(crop, [(c["contour"] - [X1, Y1]).astype(np.int32).reshape(-1, 1, 2)], True, col, 2)
    s = THUMB_PX / max(crop.shape[:2])
    return cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def _b64(img: np.ndarray, q: int) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return base64.b64encode(buf.tobytes()).decode()


def _hex(bgr) -> str:
    return "#%02x%02x%02x" % (bgr[2], bgr[1], bgr[0])


def summary(found: list) -> dict:
    """Headline numbers shared by the HTML/PDF report and the wizard screen.
    Longest/total length leave out "verify" rows (pipes/joints would inflate them)."""
    wall = [r for _, r, loc in found if loc == "wall"]
    plausible = [r for r in wall if not r["max_width_mm"] > IMPLAUSIBLE_WIDTH_MM]
    by_cat: dict = {}
    for r in wall:
        key = (f"BRE {r['bre_category']}" if r["width_resolved"]
               else f"BRE ≤{r['bre_category']}" if r["n_views"] else "no width")
        by_cat[key] = by_cat.get(key, 0) + 1
    return {"wall": len(wall), "offwall": len(found) - len(wall), "verify": len(wall) - len(plausible),
            "longest_m": max((r["length_mm"] for r in plausible), default=0.0) / 1000,
            "total_length_m": sum(r["length_mm"] for r in plausible) / 1000,
            "by_category": dict(sorted(by_cat.items(), key=lambda kv: (kv[0].replace("≤", ""), kv[0])))}


def write_pdf(html_path: Path, pdf_path: Path) -> Path | None:
    """HTML -> PDF with WeasyPrint. None (with a warning) if its system libraries are missing."""
    if sys.platform == "darwin" and Path("/opt/homebrew/lib").exists():  # Homebrew's Pango isn't on dyld's path
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join(
            p for p in ("/opt/homebrew/lib", os.environ.get("DYLD_FALLBACK_LIBRARY_PATH")) if p)
    try:
        from weasyprint import HTML
    except (ImportError, OSError) as e:
        print(f"[report] PDF skipped ({e.__class__.__name__}): pip install weasyprint; macOS: brew install pango",
              file=sys.stderr)
        return None
    HTML(filename=str(html_path)).write_pdf(str(pdf_path))
    return pdf_path


def write_report(out_dir: Path, image: np.ndarray, found: list, scale: dict,
                 title: str = "Facade crack report") -> dict:
    """Writes cracks.csv, facade_labelled.jpg, report.html into out_dir. Returns their paths + the table."""
    out_dir = Path(out_dir)
    table = crack_table(found)
    ordered = review_order(found)

    csv_path = out_dir / "cracks.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]) if table else ["crack #"])
        w.writeheader()
        w.writerows(table)

    labelled = draw_labelled(image, found)
    img_path = out_dir / "facade_labelled.jpg"
    cv2.imwrite(str(img_path), labelled, [cv2.IMWRITE_JPEG_QUALITY, 90])
    small = cv2.resize(labelled, None, fx=REPORT_IMG_W / labelled.shape[1], fy=REPORT_IMG_W / labelled.shape[1],
                       interpolation=cv2.INTER_AREA) if labelled.shape[1] > REPORT_IMG_W else labelled

    sm = summary(found)

    rows_html = []
    for (c, r, loc), t in zip(ordered, table):
        col = _colour(r, loc)
        cells = [f'<td class="num" style="color:{_hex(col)}">{t["crack #"]}</td>',
                 f'<td><img src="data:image/jpeg;base64,{_b64(_thumb(image, c, col), 70)}" alt="crack {t["crack #"]}"></td>',
                 f"<td>{html.escape(loc)}</td>",
                 f'<td><span class="sev" style="background:{_hex(col)}"></span>{html.escape(str(t["BRE cat"]))} · '
                 f'{html.escape(t["BRE"])}</td>',
                 f'<td class="r">{t["length (mm)"]:.0f}</td>',
                 f'<td class="r">{html.escape(t["width (mm)"])}</td>',
                 f'<td class="r">{html.escape(t["max width (mm)"])}</td>',
                 f'<td>{t["tier"]} ({t["score"]:.2f})</td>', f'<td class="r">{t["views"]}</td>',
                 f'<td class="warn">{html.escape(t["note"])}</td>']
        rows_html.append("<tr>" + "".join(cells) + "</tr>")

    legend = "".join(f'<span class="key"><span class="sev" style="background:{_hex(SEVERITY_BGR[k])}"></span>{lbl}</span>'
                     for k, lbl in [(1, "BRE 0–1 (≤1 mm)"), (2, "BRE 2 (≤5 mm)"), (3, "BRE 3 (≤15 mm)"),
                                    (4, "BRE 4–5 (>15 mm)")])
    legend += f'<span class="key"><span class="sev" style="background:{_hex(OFFWALL_BGR)}"></span>off-wall (review)</span>'
    cats = " · ".join(f"{k}: {v}" for k, v in sm["by_category"].items())

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>
<style>
body{{font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif;margin:24px;color:#1d1d1f;background:#fff}}
h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:16px;margin:28px 0 8px}} .muted{{color:#666}}
.cards{{display:flex;flex-wrap:wrap;gap:12px;margin:16px 0}}
.card{{border:1px solid #ddd;border-radius:8px;padding:10px 14px;min-width:150px}}
.card b{{display:block;font-size:22px}}
.facade{{width:100%;height:auto;border:1px solid #ddd}}
table{{border-collapse:collapse;width:100%}} th,td{{border-bottom:1px solid #eee;padding:4px 8px;text-align:left;vertical-align:middle}}
th{{position:sticky;top:0;background:#fafafa}} td.r{{text-align:right;font-variant-numeric:tabular-nums}}
td.num{{font-weight:700;font-size:16px}} td img{{display:block;border-radius:4px}}
.sev{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:middle}}
.key{{margin-right:16px;white-space:nowrap}} td.warn{{color:#a15c00;font-size:12px}} .note{{background:#f6f6f6;border-radius:8px;padding:10px 14px}}
@page{{size:A4 landscape;margin:10mm 10mm 14mm;
  @bottom-left{{content:"{html.escape(title)}";font:9px system-ui;color:#888}}
  @bottom-right{{content:"Page " counter(page) " of " counter(pages);font:9px system-ui;color:#888}}}}
@media print{{body{{margin:0;font-size:11px}} th{{position:static}} tr{{break-inside:avoid}} thead{{display:table-header-group}}
  h2.facade-h,h2.cracks{{break-before:page}} .facade{{max-height:160mm;width:auto;max-width:100%}}
  td img{{height:16mm;width:auto}} td{{padding:2px 6px}} td.num{{font-size:13px}}}}
</style></head><body>
<h1>{html.escape(title)}</h1>
<div class="muted">Generated {date.today().isoformat()} · camera {html.escape(scale.get("camera", "DJI Mavic Air 2 (FC3170)"))}</div>
<div class="cards">
<div class="card">On-wall cracks<b>{sm["wall"]}</b>{sm["verify"]} marked “verify”</div>
<div class="card">Off-wall, to review<b>{sm["offwall"]}</b></div>
<div class="card">Total on-wall length<b>{sm["total_length_m"]:.1f} m</b>excl. “verify”</div>
<div class="card">Longest on-wall<b>{sm["longest_m"]:.2f} m</b>excl. “verify”</div>
<div class="card">Distance to wall<b>{scale["wall_distance_m"]:.2f} m</b>{"measured" if scale.get("fit") else "assumed"}</div>
</div>
<p class="note">{html.escape(scale_note(scale))} Severity by width: {html.escape(cats)}.</p>

<h2 class="facade-h">Facade</h2>
<p>{legend}</p>
<img class="facade" src="data:image/jpeg;base64,{_b64(small, 82)}" alt="Facade with numbered cracks">

<h2 class="cracks">Cracks</h2>
<p class="muted">Numbers match the labels on the facade. Sorted on-wall first, plausible widths before
“verify” rows, then by confidence (most likely real first), then widest first. Rows marked “verify” are wider than 25 mm, which on a facade is
usually a pipe, joint or edge rather than a crack; they are kept for review, not presented as confirmed.
Width “&lt; x” means the crack is narrower than the camera can resolve (2 pixels); its true width is
below x, and its BRE category is an upper bound. Confidence tier: high ≥ 0.25, medium ≥ 0.10, low &lt; 0.10;
low-confidence findings are kept on purpose (missing a crack costs more than a false alarm).</p>
<table><thead><tr><th>#</th><th>Close-up</th><th>Location</th><th>Severity (BRE)</th><th>Length (mm)</th>
<th>Width (mm)</th><th>Max width (mm)</th><th>Confidence</th><th>Views</th><th>Note</th></tr></thead>
<tbody>{"".join(rows_html)}</tbody></table>

<h2>Method and limits</h2>
<p class="muted">Facade: COLMAP structure-from-motion, frames projected onto the fitted wall plane.
Cracks: SAHI-tiled segmentation at confidence ≥ 0.05, copies merged by connected-component analysis.
Length: crack skeleton on the facade. Width: distance transform on each source frame's mask, median over views.
Scale: {html.escape(scale_note(scale))} Severity: BRE Digest 251. Widths are upper estimates
(blur and mask edges widen thin cracks); confirm critical cracks on site with a crack-width gauge.</p>
</body></html>"""
    html_path = out_dir / "report.html"
    html_path.write_text(page, encoding="utf-8")
    pdf_path = write_pdf(html_path, out_dir / "report.pdf")
    return {"table": table, "csv": csv_path, "labelled": img_path, "html": html_path, "pdf": pdf_path,
            "summary": sm}


def _self_check() -> None:
    """ponytail: two synthetic cracks -> files exist, #1 is the wider on-wall one in image, table and HTML."""
    import tempfile
    img = np.full((400, 600, 3), 230, np.uint8)
    mk = lambda x1, y1, x2, y2: {"bbox": (x1, y1, x2, y2), "score": 0.5,
                                 "contour": np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)}
    row = lambda w, cat: {"length_mm": 100.0, "width_mm": w, "max_width_mm": w, "width_resolved": True,
                          "bre_category": cat, "bre_desc": "x", "n_views": 2}
    found = [(mk(50, 50, 150, 60), row(2.0, 2), "wall"), (mk(300, 200, 400, 230), row(9.0, 3), "wall"),
             (mk(100, 300, 200, 320), row(30.0, 5), "off-wall")]
    with tempfile.TemporaryDirectory() as d:
        out = write_report(Path(d), img, found, {"wall_distance_m": 4.8, "fit": None})
        assert out["html"].exists() and out["csv"].exists() and out["labelled"].exists()
        assert out["table"][0]["max width (mm)"] == "9.0" and out["table"][2]["location"] == "off-wall", out["table"]
        page = out["html"].read_text()
        assert page.count("<tr>") == 3 + 1 and "ASSUMED" in page
        sm = out["summary"]
        assert (sm["wall"], sm["offwall"], sm["verify"]) == (2, 1, 0) and abs(sm["total_length_m"] - 0.2) < 1e-9
        assert sm["by_category"] == {"BRE 2": 1, "BRE 3": 1}, sm["by_category"]
        if out["pdf"] is not None:
            assert out["pdf"].read_bytes()[:5] == b"%PDF-", "not a PDF"
        lab = cv2.imread(str(out["labelled"]))
        assert np.abs(lab[200, 330].astype(int) - SEVERITY_BGR[3]).sum() < 60, "crack #1 outline colour"
    print("[report] self-check OK", file=sys.stderr)


if __name__ == "__main__":
    _self_check()
