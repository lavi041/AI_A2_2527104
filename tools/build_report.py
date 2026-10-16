"""Build docs/AI_A2_2527104_REPORT.pdf from the generated artifacts (numbers are read from JSON).

Run AFTER run_all.py:  python tools/build_report.py
Edit the STUDENT_* text blocks below so the interpretations are in YOUR own words, then rebuild.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import (KeepTogether, Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,)  # noqa: E402

from app import app  # noqa: E402

NAME, REG, USER, SEED, CODE = "Aisha Niyonsaba", "25/27104", "ug2527104@ines.ac.rw", 7104, "AI-A2-7104"
OUT = ROOT / "docs" / "AI_A2_2527104_REPORT.pdf"
A = ROOT / "artifacts"
reg = json.loads((A / "regression_metrics.json").read_text())
clf = json.loads((A / "classification_metrics.json").read_text())
clu = json.loads((A / "clustering_report.json").read_text())
dq = json.loads((A / "data_quality_report.json").read_text())
meta = json.loads((ROOT / "models" / "metadata.json").read_text())
sha = dq["sha256"]

# ---------------- STUDENT INTERPRETATIONS (rewrite in your own words, then rebuild) -------------

STUDENT_REG_ERROR = (
    "The largest regression error is farm row 244. The actual yield was 15.07 t but the model "
    "predicted 12.75 t, so it under-predicted by 2.32 t. This farm has low rainfall (about 702 mm) "
    "but a large field (3.45 ha), and its soil moisture value was missing, so the pipeline filled it "
    "with the training median. I see two possible causes. First, the data has random noise with "
    "standard deviation 0.80 t, and an error of 2.32 t is about three times that, so this farm was "
    "probably just lucky. Second, if the real soil moisture was higher than the median, the imputed "
    "value made the prediction too low. The model itself is not biased, because most residuals are "
    "small and centred around zero."
)
STUDENT_CLF_ERROR = (
    "The most serious classification error is a false negative: farm row 174 is truly high-risk "
    "(5.52 t on 1.57 ha) but the model gave it only 0.36 probability, so it predicted not high-risk. "
    "Several features of this farm point to risk: low rainfall (695 mm), very low fertilizer (14.5 kg), "
    "a hot average temperature (25.6 C, far from the best 18 C) and season B. However, it also receives "
    "cooperative support and has fairly good soil moisture (58 percent), and these pushed the probability "
    "down. A false negative is worse than a false alarm for a field officer, because a farm that needs "
    "help would be missed. For this reason, I recommend that farms with probabilities between about "
    "0.3 and 0.7 are reviewed by a person instead of trusting the 0.5 threshold alone."
)
STUDENT_CLUSTER = (
    "I interpret Cluster 4 (25 farms): low rainfall (mean 715 mm), little fertilizer (mean 58 kg) and "
    "small fields (mean 1.33 ha). After clustering, this group has the lowest mean yield (8.13 t). "
    "Its high-risk share is only 40 percent, because risk is measured per hectare, so these farms "
    "produce little mostly because they are small, not only because they are inefficient. A field "
    "officer could visit this group to discuss fertilizer access and water management. I also note "
    "that the clusters are weak (silhouette 0.165), so this is a practical grouping, not a natural "
    "segment."
)
STUDENT_REFLECTION = (
    "With real data I would first check data quality with the cooperative, because spreadsheets can "
    "have wrong units and mixed formats. I would add features such as potato variety, planting date, "
    "pests and soil tests, and I would test the model on a later season instead of a random split. "
    "I would also calibrate the risk probabilities and compare errors across farmer groups to check "
    "fairness, and I would ask field officers if the risk flags are useful in practice."
)
# -------------------------------------------------------------------------------------------------

ss = getSampleStyleSheet()
B = ParagraphStyle("B", parent=ss["BodyText"], fontName="Helvetica", fontSize=9.6, leading=13)
SM = ParagraphStyle("SM", parent=B, fontSize=8.4, leading=11)
CODEF = ParagraphStyle("C", parent=B, fontName="Courier", fontSize=7.6, leading=9.6)
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=15, spaceAfter=6, keepWithNext=1, textColor=colors.HexColor("#1b5e20"))
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=11.5, spaceBefore=8, spaceAfter=3, keepWithNext=1)


def P(t, s=B):
    return Paragraph(t, s)


def table(rows, widths, head=True, fs=8.4):
    cell = ParagraphStyle("cell", parent=B, fontSize=fs, leading=fs + 2.2)
    data = [[Paragraph(str(c), cell) for c in r] for r in rows]
    t = Table(data, colWidths=widths, repeatRows=1 if head else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]
    if head:
        st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#c8e6c9")))
    t.setStyle(TableStyle(st))
    return t


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.drawString(2 * cm, 1.1 * cm, f"{NAME} | Reg. {REG} | {CODE} | SWE 3513 Assignment 2 | SYNTHETIC DATA - classroom prototype")
    canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"Page {doc.page}")
    canvas.restoreState()


def img(path, w):
    from PIL import Image as PI
    iw, ih = PI.open(path).size
    return Image(str(path), width=w, height=w * ih / iw)


def f(x):
    return "n/a" if x is None else x


S = []
# ---------------- Page 1
S += [P("Musanze Harvest Intelligence Service", H1),
      P("SWE 3513 Artificial Intelligence - Individual Assignment 2 - Technical Report (INES Ruhengeri)", B), Spacer(1, 6),
      table([["Full name", NAME], ["Registration number", REG + " (REGDIGITS 2527104)"], ["Institutional username", USER],
             ["Seed (LLLL)", str(SEED)], ["Verification code", f"<b>{CODE}</b>"],
             ["Dataset fingerprint (SHA-256)", f"<font name='Courier' size='7.5'>{sha}</font>"]], [5 * cm, 11.5 * cm], head=False, fs=9),
      P("1. Identity and decision problem", H2),
      P("<b>Stakeholder.</b> The field officer of a potato cooperative in Musanze, who receives farm records in inconsistent spreadsheets "
        "and needs an offline, evidence-based tool. A simple API allows later integration with a mobile application."),
      P("<b>Prediction questions.</b>"),
      table([["Task", "Question for the field officer", "Target"],
             ["Regression", "How many tons will this farm harvest?", "yield_tons"],
             ["Classification", "Is this farm at high production risk (yield per hectare below the median)?", "high_risk (0/1)"],
             ["Clustering", "Which farms have similar growing conditions and can be visited or advised together?", "none (unsupervised)"]],
            [3 * cm, 10 * cm, 3.5 * cm]),
      P("<b>Ethical boundary.</b> The system is a classroom decision-support prototype trained on <b>synthetic</b> data. It does not replace "
        "an agronomist and must not be used to decide access to finance, insurance or public services. Every prediction needs human review. "
        "Inputs are rejected when they fall outside the ranges the model was trained on."),
      P("<b>System overview.</b> One command (<font name='Courier'>python run_all.py --student-id ug2527104@ines.ac.rw --seed 7104</font>) "
        "generates the dataset, validates and cleans it, trains and evaluates three kinds of machine learning, saves all artifacts and models, "
        "and the same saved pipelines are used by <font name='Courier'>predict.py</font> and the FastAPI service."),
      PageBreak()]
# ---------------- Page 2
S += [P("2. Data and reproducibility", H1),
      table([["Column", "Generation rule", "Role"],
             ["rainfall_mm", "Uniform 600-1400", "numeric feature"], ["avg_temp_c", "Uniform 12-26", "numeric feature"],
             ["soil_moisture_pct", "Uniform 20-80", "numeric feature"], ["fertilizer_kg", "Uniform 0-250", "numeric feature"],
             ["field_size_ha", "Uniform 0.2-4.0", "numeric feature"], ["season", "Random A or B", "categorical feature"],
             ["cooperative_support", "Random 0 or 1", "binary feature"],
             ["yield_tons", "0.004 rain + 0.06 soil + 0.02 fert + 1.30 size - 0.25 |temp-18| + 0.70 support + season effect (+0.50 A, -0.20 B) + N(0, 0.80), clipped at 0.50", "regression target"],
             ["high_risk", "1 if yield_tons / field_size_ha is below its median", "classification target"]],
            [3.4 * cm, 10.2 * cm, 2.9 * cm]),
      P("<b>Personal seed rule.</b> The seed is LLLL = 7104 (last four digits of 25/27104) and the generator uses "
        "<font name='Courier'>numpy.random.default_rng(7104)</font>. Base rows = 240 + (7104 mod 41) = 240 + 11 = <b>251</b>. "
        "The number of deliberate missing values is m = round(0.05 x 251) = <b>13</b> in soil_moisture_pct and 13 in fertilizer_kg "
        "(different random positions); then 3 exact duplicates are appended, giving <b>254</b> raw rows. "
        "<font name='Courier'>run_all.py</font> rejects any other seed."),
      P("Data-quality evidence (artifacts/data_quality_report.json)", H2),
      table([["Check", "Result"],
             ["Raw rows", dq["rows_raw"]], ["Exact duplicate rows found / removed", f"{dq['duplicate_rows_found']} / {dq['duplicate_rows_removed']}"],
             ["Missing values in raw CSV", f"soil_moisture_pct = {dq['missing_values_raw']['soil_moisture_pct']}, fertilizer_kg = {dq['missing_values_raw']['fertilizer_kg']} "
              "(13 + 13 planted; the 3 duplicated rows copied 2 + 1 of them)"],
             ["Missing values after de-duplication", f"soil_moisture_pct = {dq['missing_values_after_dedup']['soil_moisture_pct']}, fertilizer_kg = {dq['missing_values_after_dedup']['fertilizer_kg']}"],
             ["Rows after cleaning", dq["rows_after_cleaning"]], ["Share of high_risk = 1", dq["high_risk_share"]],
             ["Checks passed", ", ".join(dq["checks_passed"])]], [6 * cm, 10.5 * cm]),
      P("Handling", H2),
      P("Duplicates are dropped (first occurrence kept). Missing values are <b>not</b> filled on the whole dataset; they are median-imputed inside the "
        "pipeline, so the medians come from the training split only. The CSV is written with four decimals and LF line endings, so its bytes "
        "(and SHA-256) are the same on any operating system."),
      P("SHA-256 fingerprint (artifacts/data_fingerprint.txt)", H2),
      P(f"<font name='Courier' size='8.5'>{sha}</font>"),
      P("Re-running run_all.py twice reproduces this fingerprint and the metrics exactly (verified by an automated test). To recreate the CSV alone: "
        "<font name='Courier'>python data/generate_data.py --student-id ug2527104@ines.ac.rw --seed 7104</font>."),
      PageBreak()]
# ---------------- Page 3
S += [P("3. Pipeline design", H1),
      P("<b>Train/test separation.</b> After cleaning (251 rows) the data is split once with <font name='Courier'>train_test_split(test_size=0.25, "
        f"random_state=7104, stratify=high_risk)</font> into <b>{reg['train_rows']} training</b> and <b>{reg['test_rows']} test</b> farms. The same split "
        "is used for all three tasks. The test set is touched only for final reporting."),
      P("<b>Preprocessing flow (one ColumnTransformer inside every Pipeline).</b>"),
      table([["Feature group", "Columns", "Transformation (fit on training rows only)"],
             ["Numeric", "rainfall_mm, avg_temp_c, soil_moisture_pct, fertilizer_kg, field_size_ha", "SimpleImputer(median) then StandardScaler"],
             ["Categorical", "season", "OneHotEncoder(drop='if_binary', handle_unknown='error')"],
             ["Binary", "cooperative_support", "passthrough (0/1)"]], [2.6 * cm, 6.9 * cm, 7 * cm]),
      P("<b>Leakage controls.</b> yield_tons, yield per hectare and high_risk are never model inputs (<font name='Courier'>assert_no_leakage</font> "
        "runs when the preprocessor is built, and the API schema rejects extra fields such as yield_tons). high_risk is derived from the yield, so using it "
        "(or yield) as a feature would hide the real difficulty of the task. Imputers, scalers and encoders are never fitted on the full dataset. "
        "Clustering uses only the seven input features."),
      P("<b>Model candidates and selection rule.</b>"),
      table([["Task", "Baseline", "Candidates", "Selection (training data only)"],
             ["Regression", "Mean predictor", "Linear regression, Random forest (200 trees), Gradient boosting", "Lowest 5-fold CV RMSE"],
             ["Classification", "Majority class", "Logistic regression, Random forest (200 trees), Gradient boosting", "Highest 5-fold CV F1"],
             ["Clustering", "-", "KMeans, K = 2..6 (n_init=10)", "Highest silhouette score"]], [2.5 * cm, 2.6 * cm, 7 * cm, 4.4 * cm]),
      P("All random states equal 7104. Final models are the selected pipelines fitted on the training split, so the reported test metrics describe exactly "
        "the saved models (<font name='Courier'>models/*.joblib</font>). The same pipeline objects are loaded by predict.py and app.py, so training and "
        "inference apply identical transformations."),
      P("<b>Code structure.</b> src/config.py (constants, identity), validation.py (seed, schema, ranges, leakage), preprocessing.py, modeling.py, "
        "clustering.py, evaluation.py (plots, predictions), schemas.py (typed input), inference.py (artifact loading, prediction), pipeline.py "
        "(orchestration); entry points run_all.py, predict.py, app.py."),
      PageBreak()]
# ---------------- Page 4
rr, cc = reg["models"], clf["models"]
S += [P("4. Regression and classification", H1),
      P("Regression (63 held-out test farms)", H2),
      table([["Model", "CV RMSE (train)", "Test MAE", "Test RMSE", "Test R2"]] +
            [[n + (" <b>(selected)</b>" if n == reg["selected_model"] else ""), m["cv_rmse_mean"], m["test_mae"], m["test_rmse"], m["test_r2"]] for n, m in rr.items()],
            [5.5 * cm, 2.8 * cm, 2.5 * cm, 2.8 * cm, 2.9 * cm]),
      P(f"<b>Selection.</b> {reg['selection_reason']} Linear regression wins because the true yield formula is almost linear; flexible tree models need more than "
        "188 rows to match it. The mean baseline has R2 below zero on the test set, so the selected model is far better than guessing the average."),
      img(A / "plots" / "residuals.png", 10.5 * cm),
      P(f"<b>Residual interpretation and one important error (my analysis).</b> Residuals are centred on zero with no strong curve against the prediction, "
        f"so the linear model is not systematically biased. {STUDENT_REG_ERROR}", SM),
      P("Classification (positive class = high_risk)", H2),
      table([["Model", "CV F1", "Precision", "Recall", "F1", "ROC-AUC", "Confusion [[TN,FP],[FN,TP]]"]] +
            [[n + (" <b>(selected)</b>" if n == clf["selected_model"] else ""), m["cv_f1_mean"], m["test_precision"], m["test_recall"], m["test_f1"], f(m["test_roc_auc"]), m["confusion_matrix"]]
             for n, m in cc.items()], [3.9 * cm, 1.5 * cm, 1.8 * cm, 1.6 * cm, 1.5 * cm, 1.8 * cm, 4.4 * cm], fs=8),
      P(f"<b>Selection.</b> {clf['selection_reason']} The majority baseline never predicts high risk (precision, recall and F1 = 0; AUC 0.5). "
        "Accuracy alone would be misleading, so precision, recall, F1 and ROC-AUC are all reported.", SM)]
# second part continues on page 5? keep confusion + error on page 5 start
sel = cc[clf["selected_model"]]
conf_text = [
    P(f"<b>Confusion-matrix interpretation.</b> The selected model made 1 false positive and 1 false negative among 63 farms "
      f"(precision {sel['test_precision']}, recall {sel['test_recall']}). The classes are balanced (about 50% high risk), so the result is not driven by class imbalance. "
      "Test-set size is small, so one more error changes recall by about 3 percentage points: the metrics are indicative, not exact.", SM),
    Spacer(1, 4),
    P(f"<b>One important error (my analysis).</b> {STUDENT_CLF_ERROR}", SM),
]
side = Table([[img(A / "plots" / "confusion_matrix.png", 5.4 * cm), conf_text]], colWidths=[5.8 * cm, 10.7 * cm])
side.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
S += [Spacer(1, 4), side, Spacer(1, 6)]
# ---------------- Page 6 clustering
S += [P("5. Clustering", H1),
      table([["K"] + [str(k) for k in clu["silhouette_by_k"]], ["Silhouette"] + [str(v) for v in clu["silhouette_by_k"].values()]],
            [2.4 * cm] + [2.8 * cm] * 5),
      P(f"<b>Selected K = {clu['selected_k']}</b> (silhouette {clu['selected_silhouette']}). {clu['interpretation']} The scores for K = 2..6 are close "
        "(0.140 to 0.165), which is expected because the generator draws every feature independently and uniformly: the data has no natural clusters. "
        "I report this honestly instead of claiming strong segments."),
      img(A / "plots" / "clusters.png", 8.6 * cm),
      KeepTogether([
      P("Cluster profiles (training farms; yield and risk are shown after clustering and were not used to build it)", H2),
      table([["Cluster", "n", "Profile (vs overall average)", "Mean yield (t)", "High-risk share"]] +
            [[p["cluster"], p["size"], p["label"].split(": ", 1)[1], p["posthoc_mean_yield_tons"], p["posthoc_high_risk_share"]] for p in clu["profiles"]],
            [1.5 * cm, 1.0 * cm, 8.4 * cm, 2.6 * cm, 3.0 * cm], fs=8)]),
      P(f"<b>One cluster and its operational meaning (my analysis).</b> {STUDENT_CLUSTER}", SM)]
# ---------------- Page 7 API + testing (+ model card)
cl = TestClient(app)
valid = json.loads((ROOT / "samples" / "valid.json").read_text())
invalid = json.loads((ROOT / "samples" / "invalid.json").read_text())
rv = cl.post("/predict", json=valid)
ri = cl.post("/predict", json=invalid)
sv = json.dumps({k: v for k, v in rv.json().items() if k != "disclaimer"}, indent=1)
si = json.dumps(ri.json(), indent=1)
S += [P("6. API and testing", H1),
      table([["Endpoint", "Purpose", "Responses"],
             ["GET /health", "Liveness and model availability, includes verification code", "200 ok; 503 if models missing"],
             ["GET /model-info", "Student id, seed, AI-A2-7104, dataset fingerprint, selected models, metrics, cluster profiles", "200"],
             ["POST /predict", "Typed JSON body (7 fields, strict types, ranges); returns yield estimate, risk, probability, cluster, verification code", "200; 422 validation error; 503"]],
            [3 * cm, 10 * cm, 3.5 * cm], fs=8),
      P("Valid request (samples/valid.json) and response (HTTP %d)" % rv.status_code, H2),
      P(sv.replace("\n", "<br/>").replace(" ", "&nbsp;"), CODEF),
      P("Invalid request (samples/invalid.json) and response (HTTP %d)" % ri.status_code, H2),
      P(si.replace("\n", "<br/>").replace(" ", "&nbsp;"), CODEF),
      P("<b>Failure behaviour.</b> Missing field, wrong type, out-of-range value, unknown category, extra field and malformed JSON all return 422 with a compact list of "
        "field/message pairs; submitted values are not echoed back. predict.py prints the same error list and exits with code 2. "
        "<b>Tests:</b> 11 pytest tests pass (<font name='Courier'>pytest -q</font>): valid inference, missing field, wrong type, range error, unknown category / extra field, "
        "reproducibility (two full runs give identical fingerprint and metrics), dataset signature (254 rows, 13 missing, 3 duplicates), wrong seed rejection, leakage guard, "
        "API health/model-info/predict, API 422 errors.")]
S += [P("7. Model card and reflection", H1),
      table([["Item", "Statement"],
             ["Intended use", "Classroom demonstration of a decision-support tool: rough harvest estimate, high-risk flag and farm grouping to help a field officer prioritise visits."],
             ["Data", "Fully SYNTHETIC (seed 7104, 254 raw / 251 clean rows). It reflects the assignment formula, not real Musanze farms."],
             ["Performance", f"Regression R2 {rr[reg['selected_model']]['test_r2']}, MAE {rr[reg['selected_model']]['test_mae']} t; classification F1 {sel['test_f1']}, ROC-AUC {sel['test_roc_auc']}; "
              f"clusters weak (silhouette {clu['selected_silhouette']}). Test set has only 63 farms."],
             ["Limitations", "Synthetic uniform features and an almost linear target make results look better than real life; no temporal validation; small test set; weak clusters; "
              "inputs outside the trained ranges are rejected."],
             ["Risks", "Over-trusting a prediction; wrong ranking of farms; unfair treatment if used for support decisions; stale model if climate or practices change."],
             ["Prohibited use", "Deciding access to loans, insurance, subsidies or any public service; replacing an agronomist; use on real personal data without consent and a new validation."],
             ["Human oversight", "A field officer or agronomist must review every output and may override it. Borderline probabilities (about 0.3-0.7) need a site visit."]],
            [3.2 * cm, 13.3 * cm], fs=8.4),
      P("Reflection - what I would improve with real data", H2),
      P(STUDENT_REFLECTION),
      P(f"<b>Reproduce:</b> <font name='Courier'>python run_all.py --student-id ug2527104@ines.ac.rw --seed 7104</font> | verification code {CODE} | SHA-256 {sha[:16]}...")]

doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.6 * cm, bottomMargin=1.8 * cm,
                        title=f"AI_A2_2527104_REPORT {CODE}", author=NAME)
OUT.parent.mkdir(exist_ok=True)
doc.build(S, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT)
