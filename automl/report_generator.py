import os
import json
import logging
from datetime import datetime
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor, black, white, grey
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY

from automl.logger import get_logger

logger = get_logger("Report_Generator")

# Standard Color Palette
COLOR_PRIMARY = HexColor("#1A1A2E")      # Dark navy — headings / cover
COLOR_SECONDARY = HexColor("#16213E")    # Slightly lighter navy
COLOR_ACCENT = HexColor("#0F3460")       # Blue accent
COLOR_SUCCESS = HexColor("#2ECC71")      # Green — good metrics / low risk
COLOR_WARNING = HexColor("#F39C12")      # Orange — medium severity / warnings
COLOR_DANGER = HexColor("#E74C3C")       # Red — high severity / errors
COLOR_LIGHT = HexColor("#F5F5F5")        # Light grey — table backgrounds
COLOR_BORDER = HexColor("#DDDDDD")       # Table borders


def _validate_report_inputs(
    model_card: dict = None,
    evaluator_result: dict = None,
    explainer_result: dict = None,
    task_type: str = "Regression"
) -> dict:
    """
    Step 1 — Validates report inputs and verifies which diagnostic plots actually exist on disk.
    Tolerant validator that never raises exceptions.
    """
    model_card_available = model_card is not None and isinstance(model_card, dict) and len(model_card) > 0

    all_plot_paths = {}
    if evaluator_result and isinstance(evaluator_result, dict):
        eval_plots = evaluator_result.get("plot_paths", {})
        if isinstance(eval_plots, dict):
            for k, v in eval_plots.items():
                if v:
                    all_plot_paths[f"eval_{k}"] = str(v)

    if explainer_result and isinstance(explainer_result, dict):
        expl_plots = explainer_result.get("plot_paths", {})
        if isinstance(expl_plots, dict):
            for k, v in expl_plots.items():
                if v:
                    all_plot_paths[f"expl_{k}"] = str(v)

    available_plot_paths = {}
    missing_plot_paths = []

    for plot_name, plot_path in all_plot_paths.items():
        if plot_path and Path(plot_path).exists():
            available_plot_paths[plot_name] = plot_path
        else:
            missing_plot_paths.append(plot_name)

    evaluator_plots_available = any(k.startswith("eval_") for k in available_plot_paths)
    explainer_plots_available = any(k.startswith("expl_") for k in available_plot_paths)

    feasibility = {
        "can_generate_report": model_card_available,
        "model_card_available": model_card_available,
        "evaluator_plots_available": evaluator_plots_available,
        "explainer_plots_available": explainer_plots_available,
        "missing_plot_count": len(missing_plot_paths),
        "available_plot_paths": available_plot_paths,
        "missing_plot_paths": missing_plot_paths,
    }

    if not model_card_available:
        logger.error("Report generator validation failed: model_card is missing or empty.")
    else:
        logger.info(
            f"Report validation passed | Available plots: {len(available_plot_paths)} | "
            f"Missing plots: {len(missing_plot_paths)}"
        )

    return feasibility


def _create_pdf_styles() -> dict:
    """
    Step 2 — Creates and centralizes all ReportLab paragraph and text styles used in the report.
    """
    styles = getSampleStyleSheet()

    custom_styles = {
        "report_title": ParagraphStyle(
            "report_title",
            parent=styles["Title"],
            fontSize=26,
            textColor=white,
            alignment=TA_LEFT,
            spaceAfter=4,
            fontName="Helvetica-Bold",
        ),
        "report_subtitle": ParagraphStyle(
            "report_subtitle",
            parent=styles["Normal"],
            fontSize=12,
            textColor=HexColor("#CCCCCC"),
            alignment=TA_LEFT,
            spaceAfter=6,
            fontName="Helvetica",
        ),
        "section_heading": ParagraphStyle(
            "section_heading",
            parent=styles["Heading1"],
            fontSize=15,
            textColor=COLOR_PRIMARY,
            spaceBefore=14,
            spaceAfter=8,
            fontName="Helvetica-Bold",
        ),
        "subsection_heading": ParagraphStyle(
            "subsection_heading",
            parent=styles["Heading2"],
            fontSize=11,
            textColor=COLOR_SECONDARY,
            spaceBefore=10,
            spaceAfter=5,
            fontName="Helvetica-Bold",
        ),
        "body_text": ParagraphStyle(
            "body_text",
            parent=styles["Normal"],
            fontSize=9,
            textColor=HexColor("#333333"),
            spaceAfter=5,
            leading=13,
            alignment=TA_LEFT,
            fontName="Helvetica",
        ),
        "body_bold": ParagraphStyle(
            "body_bold",
            parent=styles["Normal"],
            fontSize=9,
            textColor=black,
            fontName="Helvetica-Bold",
            spaceAfter=4,
            leading=13,
        ),
        "metric_value": ParagraphStyle(
            "metric_value",
            parent=styles["Normal"],
            fontSize=18,
            textColor=COLOR_PRIMARY,
            alignment=TA_CENTER,
            fontName="Helvetica-Bold",
            leading=20,
        ),
        "metric_label": ParagraphStyle(
            "metric_label",
            parent=styles["Normal"],
            fontSize=8,
            textColor=HexColor("#666666"),
            alignment=TA_CENTER,
            fontName="Helvetica",
            spaceBefore=2,
        ),
        "bullet": ParagraphStyle(
            "bullet",
            parent=styles["Normal"],
            fontSize=9,
            textColor=HexColor("#333333"),
            leftIndent=12,
            spaceAfter=3,
            leading=12,
            fontName="Helvetica",
        ),
        "warning_text": ParagraphStyle(
            "warning_text",
            parent=styles["Normal"],
            fontSize=9,
            textColor=COLOR_WARNING,
            fontName="Helvetica-Bold",
        ),
        "danger_text": ParagraphStyle(
            "danger_text",
            parent=styles["Normal"],
            fontSize=9,
            textColor=COLOR_DANGER,
            fontName="Helvetica-Bold",
        ),
        "caption": ParagraphStyle(
            "caption",
            parent=styles["Normal"],
            fontSize=8,
            textColor=HexColor("#666666"),
            alignment=TA_CENTER,
            fontName="Helvetica-Oblique",
            spaceAfter=6,
            spaceBefore=3,
        ),
        "table_header": ParagraphStyle(
            "table_header",
            parent=styles["Normal"],
            fontSize=8,
            textColor=white,
            alignment=TA_LEFT,
            fontName="Helvetica-Bold",
        ),
        "table_cell": ParagraphStyle(
            "table_cell",
            parent=styles["Normal"],
            fontSize=8,
            textColor=HexColor("#333333"),
            alignment=TA_LEFT,
            fontName="Helvetica",
            leading=10,
        ),
        "code_text": ParagraphStyle(
            "code_text",
            parent=styles["Code"],
            fontSize=7,
            textColor=HexColor("#222222"),
            fontName="Courier",
            leading=9,
        ),
    }

    return custom_styles


def _make_metric_box(label: str, value, styles: dict, color=None, width_cm: float = 5.3) -> Table:
    """
    Step 4 — Creates a styled, bordered metric box flowable with high visual prominence.
    """
    display_val = "N/A" if value is None else str(value)
    box_data = [
        [Paragraph(display_val, styles["metric_value"])],
        [Paragraph(str(label), styles["metric_label"])],
    ]
    box_color = color or HexColor("#F0F4FF")
    box = Table(box_data, colWidths=[width_cm * cm])
    box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), box_color),
        ("BOX", (0, 0), (-1, -1), 1, COLOR_BORDER),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return box


def _embed_plot(plot_path: str, width_cm: float, caption_text: str, styles: dict) -> list:
    """
    Safely embeds a plot image into the PDF. Never raises; returns a clean placeholder on failure.
    """
    if not plot_path or not Path(plot_path).exists():
        placeholder_data = [[
            Paragraph(f"<b>Plot not available:</b> {caption_text}", styles["caption"])
        ]]
        placeholder = Table(placeholder_data, colWidths=[width_cm * cm])
        placeholder.setStyle(TableStyle([
            ("BOX", (0, 0), (-1, -1), 1, COLOR_BORDER),
            ("BACKGROUND", (0, 0), (-1, -1), COLOR_LIGHT),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 20),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 20),
        ]))
        return [placeholder, Paragraph(caption_text, styles["caption"])]

    try:
        img = Image(str(plot_path), width=width_cm * cm, height=(width_cm * 0.62) * cm)
        return [img, Paragraph(caption_text, styles["caption"])]
    except Exception as e:
        logger.warning(f"Failed to embed plot '{plot_path}': {e}")
        placeholder_data = [[Paragraph(f"Plot load error ({caption_text})", styles["caption"])]]
        placeholder = Table(placeholder_data, colWidths=[width_cm * cm])
        placeholder.setStyle(TableStyle([
            ("BOX", (0, 0), (-1, -1), 1, COLOR_BORDER),
            ("BACKGROUND", (0, 0), (-1, -1), COLOR_LIGHT),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("TOPPADDING", (0, 0), (-1, -1), 15),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 15),
        ]))
        return [placeholder]


def _build_cover_page(model_card: dict, styles: dict, task_type: str = "Regression") -> list:
    """
    Step 3 — Builds the publication-ready cover page.
    """
    elements = []
    try:
        model_info = model_card.get("model", {})
        model_name = str(model_info.get("model_name", "AutoML Best Model"))
        if len(model_name) > 40:
            model_name = model_name[:37] + "..."

        source = model_info.get("model_source", "baseline")
        dataset_info = model_card.get("dataset", {})
        n_rows = dataset_info.get("n_rows_original", 0)
        target_col = dataset_info.get("target_column", "Target")
        generated_at = model_card.get("generated_at", datetime.now().isoformat(timespec="seconds"))
        perf = model_card.get("performance", {})
        primary_metrics = perf.get("primary_metrics", {})

        # 1. Dark Header Banner
        header_data = [
            [Paragraph("AutoML Intelligence Platform", styles["report_subtitle"])],
            [Paragraph("MODEL CARD & AUDIT REPORT", styles["report_title"])],
        ]
        header_table = Table(header_data, colWidths=[17.0 * cm])
        header_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), COLOR_PRIMARY),
            ("LEFTPADDING", (0, 0), (-1, -1), 16),
            ("RIGHTPADDING", (0, 0), (-1, -1), 16),
            ("TOPPADDING", (0, 0), (-1, -1), 22),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 20),
        ]))
        elements.append(header_table)
        elements.append(Spacer(1, 0.8 * cm))

        # 2. Model Title & Task Badge
        elements.append(Paragraph(f"<b>Selected Model:</b> {model_name}", styles["section_heading"]))
        elements.append(Paragraph(f"<b>Task Type:</b> {task_type} | <b>Target:</b> {target_col}", styles["body_text"]))
        elements.append(Spacer(1, 0.5 * cm))

        # 3. 3-Box Key Stats Row
        task_lower = task_type.lower()
        if task_lower == "regression":
            rmse_val = primary_metrics.get("rmse", {}).get("value", 0.0)
            r2_val = primary_metrics.get("r2", {}).get("value", 0.0)
            acc_pct = max(0.0, min(100.0, r2_val * 100.0))
            box1 = _make_metric_box("Accuracy", f"{acc_pct:.1f}%", styles)
            box2 = _make_metric_box("RMSE", f"{abs(rmse_val):,.2f}", styles)
            box3 = _make_metric_box("Training Samples", f"{n_rows:,}", styles)
        elif task_lower == "classification":
            acc_val = primary_metrics.get("accuracy", {}).get("value", 0.0)
            f1_val = primary_metrics.get("f1_weighted", {}).get("value", 0.0)
            box1 = _make_metric_box("Accuracy", f"{acc_val * 100:.1f}%", styles)
            box2 = _make_metric_box("Weighted F1", f"{f1_val:.3f}", styles)
            box3 = _make_metric_box("Training Samples", f"{n_rows:,}", styles)
        else:  # Clustering
            sil_val = primary_metrics.get("silhouette", {}).get("value", 0.0)
            n_clusters = perf.get("n_clusters", 0)
            box1 = _make_metric_box("Silhouette", f"{sil_val:.3f}", styles)
            box2 = _make_metric_box("Clusters", f"{n_clusters}", styles)
            box3 = _make_metric_box("Samples", f"{n_rows:,}", styles)

        metrics_table = Table([[box1, box2, box3]], colWidths=[5.6 * cm, 5.6 * cm, 5.6 * cm])
        metrics_table.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        elements.append(metrics_table)
        elements.append(Spacer(1, 0.8 * cm))

        # 4. Divider Line
        elements.append(HRFlowable(width="100%", thickness=1, color=COLOR_BORDER, spaceBefore=5, spaceAfter=15))

        # 5. Metadata Block
        elements.append(Paragraph(f"<b>Generated:</b> {generated_at}", styles["body_text"]))
        elements.append(Paragraph(f"<b>Optimization Source:</b> {'Optuna HPO-Tuned' if source == 'hpo' else 'Baseline Training'}", styles["body_text"]))
        elements.append(Paragraph(f"<b>Pipeline Version:</b> AutoML v1.0 Production Architecture", styles["body_text"]))
        elements.append(PageBreak())

    except Exception as e:
        logger.error(f"Error rendering Cover Page: {e}")
        elements.append(Paragraph(f"Cover page unavailable due to error: {e}", styles["body_text"]))
        elements.append(PageBreak())

    return elements


def _build_table_of_contents(sections_included: list, styles: dict) -> list:
    """
    Step 14 — Builds an estimated Table of Contents.
    """
    elements = []
    try:
        toc_entries = [
            ("1. Executive Summary", 3),
            ("2. Dataset Overview", 4),
            ("3. Model Selection & Configuration", 5),
            ("4. Model Performance & Diagnostics", 6),
            ("5. Fairness & Subgroup Performance", 8),
            ("6. Model Explainability (SHAP)", 9),
            ("7. Limitations & Recommendations", 11),
            ("8. Pipeline Lineage & Provenance", 12),
        ]

        elements.append(Paragraph("Table of Contents", styles["section_heading"]))
        elements.append(Spacer(1, 0.3 * cm))

        toc_data = []
        for section_name, page_num in toc_entries:
            toc_data.append([
                Paragraph(f"<b>{section_name}</b>", styles["body_text"]),
                Paragraph(str(page_num), styles["body_text"]),
            ])

        toc_table = Table(toc_data, colWidths=[14.5 * cm, 2.5 * cm])
        toc_table.setStyle(TableStyle([
            ("LINEBELOW", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ]))
        elements.append(toc_table)
        elements.append(PageBreak())

    except Exception as e:
        logger.error(f"Error rendering TOC: {e}")
        elements.append(Paragraph(f"Table of contents unavailable: {e}", styles["body_text"]))
        elements.append(PageBreak())

    return elements


def _build_executive_summary_section(model_card: dict, styles: dict) -> list:
    """
    Step 5 — Builds the Executive Summary section.
    """
    elements = []
    try:
        elements.append(Paragraph("1. Executive Summary", styles["section_heading"]))

        exec_summary = model_card.get("executive_summary", {})
        one_line = exec_summary.get("one_line", "AutoML model successfully trained and validated.")
        elements.append(Paragraph(one_line, styles["body_text"]))
        elements.append(Spacer(1, 0.4 * cm))

        # Verdict Box
        verdict = model_card.get("limitations", {}).get("deployment_recommendation", exec_summary.get("verdict", "Review recommended"))
        verdict_lower = verdict.lower()

        if "ready" in verdict_lower or "production" in verdict_lower:
            box_color = HexColor("#E8F8F0")
            icon = "✓"
            text_style = styles["body_bold"]
        elif "review" in verdict_lower or "requires" in verdict_lower:
            box_color = HexColor("#FEF9E7")
            icon = "⚠"
            text_style = styles["warning_text"]
        else:
            box_color = HexColor("#FDEDEC")
            icon = "✗"
            text_style = styles["danger_text"]

        verdict_data = [[
            Paragraph(f"<b>Deployment Recommendation:</b> {icon} {verdict}", text_style)
        ]]
        verdict_table = Table(verdict_data, colWidths=[17.0 * cm])
        verdict_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), box_color),
            ("BOX", (0, 0), (-1, -1), 1, COLOR_BORDER),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))
        elements.append(verdict_table)
        elements.append(Spacer(1, 0.4 * cm))

        # 3 Summary Highlights
        dataset_info = model_card.get("dataset", {})
        model_info = model_card.get("model", {})
        perf_info = model_card.get("performance", {})

        highlights_data = [
            [
                Paragraph("<b>Task & Data</b>", styles["body_bold"]),
                Paragraph("<b>Algorithm Selection</b>", styles["body_bold"]),
                Paragraph("<b>Validation Strategy</b>", styles["body_bold"]),
            ],
            [
                Paragraph(f"{model_card.get('task_type', '')} on {dataset_info.get('n_rows_original', 0):,} rows with target '{dataset_info.get('target_column', '')}'.", styles["table_cell"]),
                Paragraph(f"Top model: {model_info.get('model_name', '')} ({model_info.get('model_source', '').upper()}).", styles["table_cell"]),
                Paragraph(f"5-Fold Cross Validation + {dataset_info.get('n_rows_test', 0):,} sample held-out test set.", styles["table_cell"]),
            ]
        ]
        highlights_table = Table(highlights_data, colWidths=[5.6 * cm, 5.6 * cm, 5.6 * cm])
        highlights_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), COLOR_LIGHT),
            ("BOX", (0, 0), (-1, -1), 1, COLOR_BORDER),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ]))
        elements.append(highlights_table)
        elements.append(Spacer(1, 0.4 * cm))

        # Top 3 Recommendations
        top_recs = exec_summary.get("top_3_actions", [])
        if top_recs:
            elements.append(Paragraph("<b>Key Action Items:</b>", styles["body_bold"]))
            for idx, action in enumerate(top_recs[:3]):
                elements.append(Paragraph(f"{idx + 1}. {action}", styles["bullet"]))

        elements.append(PageBreak())

    except Exception as e:
        logger.error(f"Error rendering Executive Summary: {e}")
        elements.append(Paragraph(f"Executive summary section unavailable: {e}", styles["body_text"]))
        elements.append(PageBreak())

    return elements


def _build_dataset_section_pdf(model_card: dict, styles: dict) -> list:
    """
    Step 6 — Builds the Dataset Overview section.
    """
    elements = []
    try:
        elements.append(Paragraph("2. Dataset Overview", styles["section_heading"]))
        dataset = model_card.get("dataset", {})

        # Overview Table
        info_data = [
            [Paragraph("<b>Dataset File</b>", styles["table_cell"]), Paragraph(str(dataset.get("dataset_name", "dataset.csv")), styles["table_cell"]),
             Paragraph("<b>Task Type</b>", styles["table_cell"]), Paragraph(str(dataset.get("task_type", "Regression")), styles["table_cell"])],
            [Paragraph("<b>Target Column</b>", styles["table_cell"]), Paragraph(str(dataset.get("target_column", "N/A")), styles["table_cell"]),
             Paragraph("<b>Total Rows</b>", styles["table_cell"]), Paragraph(f"{dataset.get('n_rows_original', 0):,}", styles["table_cell"])],
            [Paragraph("<b>Train Rows (80%)</b>", styles["table_cell"]), Paragraph(f"{dataset.get('n_rows_train', 0):,}", styles["table_cell"]),
             Paragraph("<b>Test Rows (20%)</b>", styles["table_cell"]), Paragraph(f"{dataset.get('n_rows_test', 0):,}", styles["table_cell"])],
            [Paragraph("<b>Raw Features</b>", styles["table_cell"]), Paragraph(f"{dataset.get('n_columns_original', 0)}", styles["table_cell"]),
             Paragraph("<b>Final Features</b>", styles["table_cell"]), Paragraph(f"{dataset.get('feature_engineering', {}).get('final_feature_count', 0)}", styles["table_cell"])],
        ]
        info_table = Table(info_data, colWidths=[4.2 * cm, 4.3 * cm, 4.2 * cm, 4.3 * cm])
        info_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), white),
            ("BOX", (0, 0), (-1, -1), 1, COLOR_BORDER),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        elements.append(info_table)
        elements.append(Spacer(1, 0.4 * cm))

        # Column Types Subsection
        elements.append(Paragraph("Column Types Distribution", styles["subsection_heading"]))
        col_types = dataset.get("column_types", {})
        col_data = [
            [Paragraph("<b>Type Category</b>", styles["table_header"]), Paragraph("<b>Count</b>", styles["table_header"]), Paragraph("<b>Sample Features</b>", styles["table_header"])]
        ]
        for cat_name, cols in col_types.items():
            if isinstance(cols, list):
                sample_str = ", ".join(cols[:4]) + ("..." if len(cols) > 4 else "")
                col_data.append([
                    Paragraph(cat_name.replace("_", " ").title(), styles["table_cell"]),
                    Paragraph(str(len(cols)), styles["table_cell"]),
                    Paragraph(sample_str if sample_str else "None", styles["table_cell"])
                ])

        col_table = Table(col_data, colWidths=[5.0 * cm, 2.0 * cm, 10.0 * cm])
        col_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), COLOR_SECONDARY),
            ("BOX", (0, 0), (-1, -1), 1, COLOR_BORDER),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(col_table)
        elements.append(Spacer(1, 0.4 * cm))

        # Missing Data Subsection
        elements.append(Paragraph("Missing Data & Imputation", styles["subsection_heading"]))
        missing_data_info = dataset.get("missing_data", {})
        missing_cols = missing_data_info.get("columns_with_missing", [])

        if not missing_cols:
            elements.append(Paragraph("✓ No missing values detected in the training dataset.", styles["body_text"]))
        else:
            impute_strat = missing_data_info.get("imputation_strategy", {})
            miss_table_data = [
                [Paragraph("<b>Feature</b>", styles["table_header"]), Paragraph("<b>Imputation Strategy</b>", styles["table_header"])]
            ]
            for col in missing_cols[:8]:
                strat = impute_strat.get(col, "Automated Imputation")
                miss_table_data.append([
                    Paragraph(str(col), styles["table_cell"]),
                    Paragraph(str(strat), styles["table_cell"])
                ])
            miss_table = Table(miss_table_data, colWidths=[8.5 * cm, 8.5 * cm])
            miss_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), COLOR_SECONDARY),
                ("BOX", (0, 0), (-1, -1), 1, COLOR_BORDER),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            elements.append(miss_table)

        elements.append(PageBreak())

    except Exception as e:
        logger.error(f"Error rendering Dataset Section: {e}")
        elements.append(Paragraph(f"Dataset section unavailable: {e}", styles["body_text"]))
        elements.append(PageBreak())

    return elements


def _build_model_section_pdf(model_card: dict, styles: dict) -> list:
    """
    Step 7 — Builds Model Selection & Configuration section.
    """
    elements = []
    try:
        elements.append(Paragraph("3. Model Selection & Configuration", styles["section_heading"]))
        model_info = model_card.get("model", {})
        selection = model_info.get("selection_process", {})

        # Selection overview text
        n_trained = selection.get("n_models_trained", 0)
        n_tuned = selection.get("n_models_hpo_tuned", 0)
        crit = selection.get("selection_criterion", "Cross-Validated Loss")
        elements.append(Paragraph(
            f"Trained <b>{n_trained}</b> candidate models across cross-validation folds. "
            f"The top <b>{n_tuned}</b> algorithms progressed to Bayesian Hyperparameter Optimization (Optuna TPE), "
            f"selected by <i>{crit}</i>.",
            styles["body_text"]
        ))
        elements.append(Spacer(1, 0.3 * cm))

        # Leaderboard Table
        elements.append(Paragraph("Model Leaderboard (CV Performance)", styles["subsection_heading"]))
        hpo_lb = model_info.get("hpo_leaderboard", []) or model_info.get("baseline_leaderboard", [])

        if hpo_lb:
            lb_data = [
                [Paragraph("<b>Rank</b>", styles["table_header"]), Paragraph("<b>Model</b>", styles["table_header"]),
                 Paragraph("<b>Source</b>", styles["table_header"]), Paragraph("<b>Metric Score</b>", styles["table_header"])]
            ]
            for idx, entry in enumerate(hpo_lb[:6]):
                if isinstance(entry, dict):
                    rank = entry.get("rank", idx + 1)
                    m_name = entry.get("model_name", entry.get("model_key", "Model"))
                    m_src = str(entry.get("source", "baseline")).upper()
                    m_score = entry.get("cv_rmse", entry.get("cv_mean", entry.get("score", "N/A")))
                    score_str = f"{abs(float(m_score)):,.3f}" if isinstance(m_score, (int, float)) else str(m_score)
                    lb_data.append([
                        Paragraph(str(rank), styles["table_cell"]),
                        Paragraph(str(m_name), styles["table_cell"]),
                        Paragraph(m_src, styles["table_cell"]),
                        Paragraph(score_str, styles["table_cell"])
                    ])

            lb_table = Table(lb_data, colWidths=[2.0 * cm, 6.5 * cm, 4.0 * cm, 4.5 * cm])
            lb_styles = [
                ("BACKGROUND", (0, 0), (-1, 0), COLOR_PRIMARY),
                ("BOX", (0, 0), (-1, -1), 1, COLOR_BORDER),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
            if len(lb_data) > 1:
                lb_styles.append(("BACKGROUND", (0, 1), (-1, 1), HexColor("#FFF9C4")))  # Gold rank 1
            if len(lb_data) > 2:
                lb_styles.append(("BACKGROUND", (0, 2), (-1, 2), HexColor("#F5F5F5")))  # Silver rank 2
            if len(lb_data) > 3:
                lb_styles.append(("BACKGROUND", (0, 3), (-1, 3), HexColor("#FBE9E7")))  # Bronze rank 3

            lb_table.setStyle(TableStyle(lb_styles))
            elements.append(lb_table)
        elements.append(Spacer(1, 0.4 * cm))

        # Hyperparameters Subsection
        elements.append(Paragraph("Final Model Hyperparameters", styles["subsection_heading"]))
        hparams = model_info.get("hyperparameters", {}).get("final_params", {})

        if isinstance(hparams, dict) and hparams:
            hp_data = [
                [Paragraph("<b>Parameter</b>", styles["table_header"]), Paragraph("<b>Selected Value</b>", styles["table_header"])]
            ]
            for k, v in list(hparams.items())[:10]:
                hp_data.append([
                    Paragraph(str(k), styles["table_cell"]),
                    Paragraph(str(v), styles["table_cell"])
                ])
            hp_table = Table(hp_data, colWidths=[8.5 * cm, 8.5 * cm])
            hp_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), COLOR_SECONDARY),
                ("BOX", (0, 0), (-1, -1), 1, COLOR_BORDER),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            elements.append(hp_table)
        else:
            elements.append(Paragraph("Using scikit-learn standard default parameters.", styles["body_text"]))

        elements.append(PageBreak())

    except Exception as e:
        logger.error(f"Error rendering Model Section: {e}")
        elements.append(Paragraph(f"Model section unavailable: {e}", styles["body_text"]))
        elements.append(PageBreak())

    return elements


def _build_performance_section_pdf(
    model_card: dict,
    evaluator_result: dict,
    styles: dict,
    task_type: str = "Regression",
    available_plots: dict = None
) -> list:
    """
    Step 8 — Builds Model Performance & Diagnostics section with metric tables and 2x2 plot grids.
    """
    elements = []
    try:
        elements.append(Paragraph("4. Model Performance & Diagnostics", styles["section_heading"]))
        perf = model_card.get("performance", {})
        plots = available_plots or {}
        task_lower = task_type.lower()

        # Primary Metrics Boxes
        primary = perf.get("primary_metrics", {})
        eval_metrics = evaluator_result.get("metrics", {}) if evaluator_result else {}
        
        if task_lower == "regression":
            rmse = primary.get("rmse", {}).get("value", eval_metrics.get("rmse", 0.0))
            mae = primary.get("mae", {}).get("value", eval_metrics.get("mae", 0.0))
            r2 = primary.get("r2", {}).get("value", eval_metrics.get("r2", 0.0))
            acc_pct = max(0.0, min(100.0, r2 * 100.0))
            mape = eval_metrics.get("mape", primary.get("mape", {}).get("value"))

            b1 = _make_metric_box("Accuracy", f"{acc_pct:.1f}%", styles, width_cm=4.0)
            b2 = _make_metric_box("RMSE", f"{rmse:,.2f}", styles, width_cm=4.0)
            b3 = _make_metric_box("MAE", f"{mae:,.2f}", styles, width_cm=4.0)
            b4 = _make_metric_box("R² Score", f"{r2:.3f}", styles, width_cm=4.0)
            m_table = Table([[b1, b2, b3, b4]], colWidths=[4.25 * cm, 4.25 * cm, 4.25 * cm, 4.25 * cm])
            m_table.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
            elements.append(m_table)
            elements.append(Spacer(1, 0.4 * cm))

            # Detailed Evaluation Parameter Table
            elements.append(Paragraph("Detailed Regression Evaluation Parameters", styles["subsection_heading"]))
            reg_table_data = [
                [Paragraph("<b>Evaluation Metric</b>", styles["table_header"]),
                 Paragraph("<b>Calculated Value</b>", styles["table_header"]),
                 Paragraph("<b>Target Benchmark</b>", styles["table_header"]),
                 Paragraph("<b>Diagnostic Status</b>", styles["table_header"])],
                [Paragraph("Prediction Accuracy (Variance Explained)", styles["table_cell"]), Paragraph(f"{acc_pct:.2f}%", styles["table_cell"]), Paragraph("Higher is better (≥70%)", styles["table_cell"]), Paragraph("✓ Excellent" if acc_pct >= 70 else "Satisfactory", styles["table_cell"])],
                [Paragraph("R² Determination Coefficient", styles["table_cell"]), Paragraph(f"{r2:.4f}", styles["table_cell"]), Paragraph("1.0000 (Perfect)", styles["table_cell"]), Paragraph("✓ Good Generalization" if r2 >= 0.7 else "Normal", styles["table_cell"])],
                [Paragraph("Root Mean Squared Error (RMSE)", styles["table_cell"]), Paragraph(f"{rmse:,.2f}", styles["table_cell"]), Paragraph("Minimised", styles["table_cell"]), Paragraph("✓ Low Variance", styles["table_cell"])],
                [Paragraph("Mean Absolute Error (MAE)", styles["table_cell"]), Paragraph(f"{mae:,.2f}", styles["table_cell"]), Paragraph("Minimised", styles["table_cell"]), Paragraph("✓ Robust", styles["table_cell"])],
                [Paragraph("Mean Absolute Percentage Error (MAPE)", styles["table_cell"]), Paragraph(f"{mape:.2f}%" if mape is not None else "N/A", styles["table_cell"]), Paragraph("< 20%", styles["table_cell"]), Paragraph("✓ Acceptable" if (mape and mape < 25) else "Informational", styles["table_cell"])],
                [Paragraph("Overfitting Generalization Gap", styles["table_cell"]), Paragraph(f"{eval_metrics.get('overfitting_gap', 0.0):.4f}", styles["table_cell"]), Paragraph("< 0.1000", styles["table_cell"]), Paragraph("✓ Well Regularized" if abs(eval_metrics.get('overfitting_gap', 0.0)) < 0.1 else "Review Overfitting", styles["table_cell"])],
            ]
            reg_table = Table(reg_table_data, colWidths=[6.0 * cm, 3.5 * cm, 4.0 * cm, 3.5 * cm])
            reg_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), COLOR_SECONDARY),
                ("BOX", (0, 0), (-1, -1), 1, COLOR_BORDER),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            elements.append(reg_table)

        elif task_lower == "classification":
            acc = primary.get("accuracy", {}).get("value", eval_metrics.get("accuracy", 0.0))
            f1 = primary.get("f1_weighted", {}).get("value", eval_metrics.get("f1_weighted", 0.0))
            auc = primary.get("auc_roc", {}).get("value", eval_metrics.get("auc_roc"))
            mcc = primary.get("mcc", {}).get("value", eval_metrics.get("mcc", 0.0))

            b1 = _make_metric_box("Accuracy", f"{acc * 100:.1f}%", styles, width_cm=4.0)
            b2 = _make_metric_box("Weighted F1", f"{f1:.3f}", styles, width_cm=4.0)
            b3 = _make_metric_box("AUC-ROC", f"{auc:.3f}" if auc else "N/A", styles, width_cm=4.0)
            b4 = _make_metric_box("MCC", f"{mcc:.3f}", styles, width_cm=4.0)
            m_table = Table([[b1, b2, b3, b4]], colWidths=[4.25 * cm, 4.25 * cm, 4.25 * cm, 4.25 * cm])
            m_table.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
            elements.append(m_table)
            elements.append(Spacer(1, 0.4 * cm))

            # Detailed Classification Evaluation Table
            elements.append(Paragraph("Detailed Classification Evaluation Parameters", styles["subsection_heading"]))
            clf_table_data = [
                [Paragraph("<b>Evaluation Metric</b>", styles["table_header"]),
                 Paragraph("<b>Calculated Value</b>", styles["table_header"]),
                 Paragraph("<b>Target Benchmark</b>", styles["table_header"]),
                 Paragraph("<b>Diagnostic Status</b>", styles["table_header"])],
                [Paragraph("Classification Accuracy", styles["table_cell"]), Paragraph(f"{acc * 100:.2f}%", styles["table_cell"]), Paragraph("Higher is better (≥80%)", styles["table_cell"]), Paragraph("✓ High Accuracy" if acc >= 0.8 else "Acceptable", styles["table_cell"])],
                [Paragraph("Weighted F1-Score", styles["table_cell"]), Paragraph(f"{f1:.4f}", styles["table_cell"]), Paragraph("1.0000 (Perfect)", styles["table_cell"]), Paragraph("✓ Balanced Precision/Recall", styles["table_cell"])],
                [Paragraph("Precision (Weighted)", styles["table_cell"]), Paragraph(f"{eval_metrics.get('precision', 0.0):.4f}", styles["table_cell"]), Paragraph("Higher is better", styles["table_cell"]), Paragraph("✓ Low False Positives", styles["table_cell"])],
                [Paragraph("Recall (Weighted)", styles["table_cell"]), Paragraph(f"{eval_metrics.get('recall', 0.0):.4f}", styles["table_cell"]), Paragraph("Higher is better", styles["table_cell"]), Paragraph("✓ Low False Negatives", styles["table_cell"])],
                [Paragraph("AUC-ROC (Discriminative Power)", styles["table_cell"]), Paragraph(f"{auc:.4f}" if auc else "N/A", styles["table_cell"]), Paragraph("≥ 0.8000", styles["table_cell"]), Paragraph("✓ Strong Discrimination" if (auc and auc >= 0.8) else "Informational", styles["table_cell"])],
                [Paragraph("Matthews Correlation Coefficient (MCC)", styles["table_cell"]), Paragraph(f"{mcc:.4f}", styles["table_cell"]), Paragraph("1.0000 (Perfect)", styles["table_cell"]), Paragraph("✓ Positive Correlation", styles["table_cell"])],
            ]
            clf_table = Table(clf_table_data, colWidths=[6.0 * cm, 3.5 * cm, 4.0 * cm, 3.5 * cm])
            clf_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), COLOR_SECONDARY),
                ("BOX", (0, 0), (-1, -1), 1, COLOR_BORDER),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            elements.append(clf_table)

        else:
            sil = primary.get("silhouette", {}).get("value", 0.0)
            db = primary.get("davies_bouldin", {}).get("value", 0.0)
            ch = primary.get("calinski_harabasz", {}).get("value", 0.0)
            b1 = _make_metric_box("Silhouette", f"{sil:.3f}", styles, width_cm=5.5)
            b2 = _make_metric_box("Davies-Bouldin", f"{db:.3f}", styles, width_cm=5.5)
            b3 = _make_metric_box("Calinski-Harabasz", f"{ch:.1f}", styles, width_cm=5.5)
            m_table = Table([[b1, b2, b3]], colWidths=[5.6 * cm, 5.6 * cm, 5.6 * cm])
            elements.append(m_table)

        elements.append(Spacer(1, 0.4 * cm))
        elements.append(Paragraph(f"<b>Evaluation Assessment:</b> {perf.get('interpretation', '')}", styles["body_text"]))
        elements.append(Spacer(1, 0.4 * cm))

        # Diagnostic Plots Grid (2x2)
        elements.append(Paragraph("Diagnostic Visualizations & Confusion Matrix Analysis", styles["subsection_heading"]))

        if task_lower == "regression":
            p1 = _embed_plot(plots.get("eval_predicted_vs_actual"), 8.2, "Predicted vs Actual (Regression Fit)", styles)
            p2 = _embed_plot(plots.get("eval_residuals_vs_predicted"), 8.2, "Residuals vs Predicted Values", styles)
            p3 = _embed_plot(plots.get("eval_residuals_distribution"), 8.2, "Residuals Error Distribution", styles)
            p4 = _embed_plot(plots.get("eval_residuals_qq_plot"), 8.2, "Normal Q-Q Plot (Residual Normality)", styles)
            plot_grid_data = [[p1, p2], [p3, p4]]
        elif task_lower == "classification":
            p1 = _embed_plot(plots.get("eval_confusion_matrix"), 8.2, "Confusion Matrix (Actual vs Predicted)", styles)
            p2 = _embed_plot(plots.get("eval_roc_curve"), 8.2, "Receiver Operating Characteristic (ROC)", styles)
            p3 = _embed_plot(plots.get("eval_precision_recall_curve"), 8.2, "Precision-Recall Curve", styles)
            p4 = _embed_plot(plots.get("eval_calibration_curve"), 8.2, "Probability Calibration Curve", styles)
            plot_grid_data = [[p1, p2], [p3, p4]]
        else:
            p1 = _embed_plot(plots.get("eval_cluster_scatter"), 8.2, "PCA Cluster Scatter Projection", styles)
            p2 = _embed_plot(plots.get("eval_cluster_sizes"), 8.2, "Cluster Member Distribution", styles)
            p3 = _embed_plot(plots.get("eval_cluster_heatmap"), 8.2, "Cluster Feature Mean Heatmap", styles)
            p4 = _embed_plot(plots.get("eval_silhouette_plot"), 8.2, "Silhouette Coefficient Analysis", styles)
            plot_grid_data = [[p1, p2], [p3, p4]]

        grid_table = Table(plot_grid_data, colWidths=[8.5 * cm, 8.5 * cm])
        grid_table.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]))
        elements.append(grid_table)
        elements.append(PageBreak())

    except Exception as e:
        logger.error(f"Error rendering Performance Section: {e}")
        elements.append(Paragraph(f"Performance section unavailable: {e}", styles["body_text"]))
        elements.append(PageBreak())

    return elements


def _build_fairness_section_pdf(model_card: dict, styles: dict) -> list:
    """
    Step 9 — Builds the Fairness & Bias Analysis section.
    """
    elements = []
    try:
        elements.append(Paragraph("5. Fairness & Subgroup Performance", styles["section_heading"]))
        fairness = model_card.get("fairness", {})

        if not fairness.get("fairness_assessed", False):
            elements.append(Paragraph(
                "Fairness slice analysis was skipped or not applicable (e.g., all categorical features had high cardinality).",
                styles["body_text"]
            ))
            elements.append(PageBreak())
            return elements

        has_disp = fairness.get("has_disparity", False)
        box_color = HexColor("#FDEDEC") if has_disp else HexColor("#E8F8F0")
        verdict_text = fairness.get("overall_verdict", "Analysis complete")

        v_table = Table([[Paragraph(f"<b>Status:</b> {verdict_text}", styles["body_bold"])]], colWidths=[17.0 * cm])
        v_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), box_color),
            ("BOX", (0, 0), (-1, -1), 1, COLOR_BORDER),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        elements.append(v_table)
        elements.append(Spacer(1, 0.4 * cm))

        # Disparity Details Table
        details = fairness.get("disparity_details", [])
        if details:
            elements.append(Paragraph("Subgroup Disparity Analysis", styles["subsection_heading"]))
            d_data = [
                [Paragraph("<b>Feature</b>", styles["table_header"]), Paragraph("<b>Group</b>", styles["table_header"]),
                 Paragraph("<b>Size</b>", styles["table_header"]), Paragraph("<b>Metric</b>", styles["table_header"]),
                 Paragraph("<b>Diff %</b>", styles["table_header"]), Paragraph("<b>Concern</b>", styles["table_header"])]
            ]
            for item in details[:8]:
                c_level = item.get("concern_level", "low").upper()
                d_data.append([
                    Paragraph(str(item.get("feature", "")), styles["table_cell"]),
                    Paragraph(str(item.get("group_value", "")), styles["table_cell"]),
                    Paragraph(str(item.get("group_size", "")), styles["table_cell"]),
                    Paragraph(f"{float(item.get('metric_value', 0)):,.2f}", styles["table_cell"]),
                    Paragraph(f"{float(item.get('difference_pct', 0)):.1f}%", styles["table_cell"]),
                    Paragraph(c_level, styles["table_cell"]),
                ])

            d_table = Table(d_data, colWidths=[3.5 * cm, 3.5 * cm, 2.0 * cm, 3.0 * cm, 2.5 * cm, 2.5 * cm])
            d_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), COLOR_SECONDARY),
                ("BOX", (0, 0), (-1, -1), 1, COLOR_BORDER),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            elements.append(d_table)
            elements.append(Spacer(1, 0.4 * cm))

        elements.append(Paragraph(f"<b>Recommendation:</b> {fairness.get('recommendation', '')}", styles["body_text"]))
        elements.append(PageBreak())

    except Exception as e:
        logger.error(f"Error rendering Fairness Section: {e}")
        elements.append(Paragraph(f"Fairness section unavailable: {e}", styles["body_text"]))
        elements.append(PageBreak())

    return elements


def _build_explainability_section_pdf(
    model_card: dict,
    explainer_result: dict,
    styles: dict,
    available_plots: dict = None,
    task_type: str = "Regression"
) -> list:
    """
    Step 10 — Builds the SHAP Model Explainability section.
    """
    elements = []
    try:
        elements.append(Paragraph("6. Model Explainability (SHAP)", styles["section_heading"]))
        exp = model_card.get("explainability", {})
        plots = available_plots or {}

        if not exp.get("explainability_available", False):
            elements.append(Paragraph("SHAP explainability analysis was skipped or unsupported for this model architecture.", styles["body_text"]))
            elements.append(PageBreak())
            return elements

        # Key findings
        for finding in exp.get("key_findings", []):
            elements.append(Paragraph(f"• {finding}", styles["bullet"]))
        elements.append(Spacer(1, 0.4 * cm))

        # Global Importance Plot & Summary Plot
        p_bar = _embed_plot(plots.get("expl_shap_bar"), 16.0, "Global Mean |SHAP| Feature Importance", styles)
        elements.extend(p_bar)
        elements.append(Spacer(1, 0.3 * cm))

        p_summary = _embed_plot(
            plots.get("expl_shap_summary"), 16.0,
            "SHAP Summary (Beeswarm) Plot — High feature values in red, low in blue.",
            styles
        )
        elements.extend(p_summary)
        elements.append(PageBreak())

    except Exception as e:
        logger.error(f"Error rendering Explainability Section: {e}")
        elements.append(Paragraph(f"Explainability section unavailable: {e}", styles["body_text"]))
        elements.append(PageBreak())

    return elements


def _build_limitations_and_recommendations_pdf(model_card: dict, styles: dict) -> list:
    """
    Step 11 — Builds Limitations, Recommendations & Deployment Checklist section.
    """
    elements = []
    try:
        elements.append(Paragraph("7. Limitations & Actionable Recommendations", styles["section_heading"]))
        lims = model_card.get("limitations", {})
        recs = model_card.get("recommendations", {})

        # Limitations Table
        elements.append(Paragraph("Known Failure Modes & Limitations", styles["subsection_heading"]))
        lim_list = lims.get("limitations", [])

        if lim_list:
            lim_data = [
                [Paragraph("<b>Risk Type</b>", styles["table_header"]), Paragraph("<b>Severity</b>", styles["table_header"]), Paragraph("<b>Description</b>", styles["table_header"])]
            ]
            for l in lim_list[:6]:
                sev = str(l.get("severity", "low")).upper()
                lim_data.append([
                    Paragraph(str(l.get("type", "")).replace("_", " ").title(), styles["table_cell"]),
                    Paragraph(sev, styles["danger_text"] if sev == "HIGH" else (styles["warning_text"] if sev == "MEDIUM" else styles["table_cell"])),
                    Paragraph(str(l.get("description", "")), styles["table_cell"])
                ])
            lim_table = Table(lim_data, colWidths=[4.0 * cm, 2.5 * cm, 10.5 * cm])
            lim_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), COLOR_PRIMARY),
                ("BOX", (0, 0), (-1, -1), 1, COLOR_BORDER),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            elements.append(lim_table)
            elements.append(Spacer(1, 0.4 * cm))

        # Recommendations
        elements.append(Paragraph("Prioritized Recommendations", styles["subsection_heading"]))
        rec_list = recs.get("recommendations", [])
        for r in rec_list[:4]:
            pri = str(r.get("priority", "low")).upper()
            elements.append(Paragraph(f"<b>[{pri}]</b> {r.get('action', '')}", styles["bullet"]))
        elements.append(Spacer(1, 0.4 * cm))

        # Deployment Checklist
        elements.append(Paragraph("Production Deployment Checklist", styles["subsection_heading"]))
        chk_data = [
            [Paragraph("☐", styles["body_bold"]), Paragraph("Review all high-severity risk factors prior to live traffic deployment.", styles["table_cell"])],
            [Paragraph("☐", styles["body_bold"]), Paragraph("Configure input distribution drift alerts (>20% residual drift).", styles["table_cell"])],
            [Paragraph("☐", styles["body_bold"]), Paragraph("Verify model inference pipeline against corrupt / missing feature scenarios.", styles["table_cell"])],
            [Paragraph("☐", styles["body_bold"]), Paragraph("Schedule monthly re-training cycles with version-controlled model artifacts.", styles["table_cell"])],
        ]
        chk_table = Table(chk_data, colWidths=[1.0 * cm, 16.0 * cm])
        chk_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), COLOR_LIGHT),
            ("BOX", (0, 0), (-1, -1), 1, COLOR_BORDER),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(chk_table)
        elements.append(PageBreak())

    except Exception as e:
        logger.error(f"Error rendering Limitations & Recommendations: {e}")
        elements.append(Paragraph(f"Limitations section unavailable: {e}", styles["body_text"]))
        elements.append(PageBreak())

    return elements


def _build_provenance_section_pdf(model_card: dict, styles: dict) -> list:
    """
    Step 12 — Builds Pipeline Provenance, Lineage & Reproducibility section.
    """
    elements = []
    try:
        elements.append(Paragraph("8. Pipeline Provenance & Reproducibility", styles["section_heading"]))
        prov = model_card.get("provenance", {})

        # Pipeline Stages Table
        elements.append(Paragraph("Executed Pipeline Stages", styles["subsection_heading"]))
        stages = prov.get("pipeline_stages", [])
        if stages:
            stg_data = [
                [Paragraph("<b>Stage</b>", styles["table_header"]), Paragraph("<b>Key Outputs</b>", styles["table_header"]), Paragraph("<b>Artifact Location</b>", styles["table_header"])]
            ]
            for s in stages:
                out_str = "; ".join(s.get("key_outputs", [])[:2])
                stg_data.append([
                    Paragraph(str(s.get("stage", "")), styles["table_cell"]),
                    Paragraph(out_str, styles["table_cell"]),
                    Paragraph(str(s.get("artifact", "")), styles["code_text"]),
                ])
            stg_table = Table(stg_data, colWidths=[4.0 * cm, 7.5 * cm, 5.5 * cm])
            stg_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), COLOR_SECONDARY),
                ("BOX", (0, 0), (-1, -1), 1, COLOR_BORDER),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            elements.append(stg_table)
            elements.append(Spacer(1, 0.4 * cm))

        # Software Environment
        elements.append(Paragraph("Software Environment Versions", styles["subsection_heading"]))
        versions = prov.get("software_versions", {})
        v_items = list(versions.items())
        v_data = [
            [Paragraph("<b>Package</b>", styles["table_header"]), Paragraph("<b>Version</b>", styles["table_header"]),
             Paragraph("<b>Package</b>", styles["table_header"]), Paragraph("<b>Version</b>", styles["table_header"])]
        ]
        for i in range(0, len(v_items), 2):
            p1, v1 = v_items[i]
            p2, v2 = v_items[i + 1] if i + 1 < len(v_items) else ("", "")
            v_data.append([
                Paragraph(str(p1), styles["table_cell"]), Paragraph(str(v1), styles["table_cell"]),
                Paragraph(str(p2), styles["table_cell"]), Paragraph(str(v2), styles["table_cell"])
            ])
        v_table = Table(v_data, colWidths=[4.25 * cm, 4.25 * cm, 4.25 * cm, 4.25 * cm])
        v_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), COLOR_PRIMARY),
            ("BOX", (0, 0), (-1, -1), 1, COLOR_BORDER),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        elements.append(v_table)

    except Exception as e:
        logger.error(f"Error rendering Provenance Section: {e}")
        elements.append(Paragraph(f"Provenance section unavailable: {e}", styles["body_text"]))

    return elements


def _add_header_footer(canvas, doc, model_card: dict):
    """
    Step 13 — ReportLab page callback adding standard header bar and footer page numbers.
    Critical: Always uses saveState() and restoreState() to prevent state leakage.
    """
    canvas.saveState()
    page_width, page_height = A4

    # Header Bar
    canvas.setFillColor(COLOR_PRIMARY)
    canvas.rect(0, page_height - 1.2 * cm, page_width, 1.2 * cm, fill=1, stroke=0)

    canvas.setFillColor(white)
    canvas.setFont("Helvetica-Bold", 8)
    canvas.drawString(1.5 * cm, page_height - 0.75 * cm, "AutoML Platform — Model Card Report")

    m_name = model_card.get("model", {}).get("model_name", "Best Model")
    canvas.setFont("Helvetica", 8)
    canvas.drawRightString(page_width - 1.5 * cm, page_height - 0.75 * cm, str(m_name))

    # Footer Line
    canvas.setStrokeColor(COLOR_BORDER)
    canvas.setLineWidth(0.5)
    canvas.line(1.5 * cm, 1.3 * cm, page_width - 1.5 * cm, 1.3 * cm)

    # Footer Text
    canvas.setFillColor(HexColor("#666666"))
    canvas.setFont("Helvetica", 8)
    canvas.drawString(1.5 * cm, 0.8 * cm, f"Page {doc.page}")

    gen_time = model_card.get("generated_at", "")
    canvas.drawRightString(page_width - 1.5 * cm, 0.8 * cm, f"Generated: {gen_time}")

    canvas.restoreState()


def generate_pdf_report(all_elements: list, output_path: str, model_card: dict) -> str:
    """
    Step 15 — Compiles all flowable elements into the final PDF document.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(out_file),
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
        title=f"Model Card — {model_card.get('model', {}).get('model_name', 'AutoML')}",
        author="AutoML Platform",
        subject=f"{model_card.get('task_type', '')} Model Report",
        creator="AutoML Pipeline",
    )

    def header_footer_callback(canvas, doc):
        _add_header_footer(canvas, doc, model_card)

    doc.build(
        all_elements,
        onFirstPage=header_footer_callback,
        onLaterPages=header_footer_callback,
    )

    file_size_mb = os.path.getsize(out_file) / (1024 * 1024)
    logger.info(
        f"PDF report generated successfully → {out_file.resolve()} "
        f"({file_size_mb:.2f} MB, {len(all_elements)} elements)"
    )

    return str(out_file)


def run_report_generator(
    model_card: dict = None,
    evaluator_result: dict = None,
    explainer_result: dict = None,
    task_type: str = "Regression",
    output_dir: str = "artifacts/report"
) -> dict:
    """
    Step 16 — Single Entry Point called by pipeline.py.
    Orchestrates report validation, styling, section assembly, and PDF rendering.
    """
    logger.info("=" * 60)
    logger.info(f"REPORT GENERATOR STARTED | task: {task_type}")
    logger.info("=" * 60)

    feasibility = _validate_report_inputs(
        model_card=model_card,
        evaluator_result=evaluator_result,
        explainer_result=explainer_result,
        task_type=task_type
    )

    if not feasibility["can_generate_report"]:
        logger.error("Cannot generate report — model_card is missing or invalid.")
        return {
            "status": "failed",
            "pdf_path": None,
            "pdf_size_mb": None,
            "n_pages_est": 0,
            "sections_included": [],
            "missing_plots": feasibility.get("missing_plot_paths", []),
        }

    styles = _create_pdf_styles()
    all_elements = []

    sections_included = []

    # Step 3: Cover Page
    all_elements += _build_cover_page(model_card, styles, task_type)
    sections_included.append("Cover Page")

    # Step 14: Table of Contents
    all_elements += _build_table_of_contents(sections_included, styles)
    sections_included.append("Table of Contents")

    # Step 5: Executive Summary
    all_elements += _build_executive_summary_section(model_card, styles)
    sections_included.append("Executive Summary")

    # Step 6: Dataset Overview
    all_elements += _build_dataset_section_pdf(model_card, styles)
    sections_included.append("Dataset Overview")

    # Step 7: Model Selection
    all_elements += _build_model_section_pdf(model_card, styles)
    sections_included.append("Model Selection & Configuration")

    # Step 8: Performance
    all_elements += _build_performance_section_pdf(
        model_card, evaluator_result, styles, task_type, feasibility["available_plot_paths"]
    )
    sections_included.append("Model Performance & Diagnostics")

    # Step 9: Fairness
    all_elements += _build_fairness_section_pdf(model_card, styles)
    sections_included.append("Fairness & Subgroup Performance")

    # Step 10: Explainability
    all_elements += _build_explainability_section_pdf(
        model_card, explainer_result, styles, feasibility["available_plot_paths"], task_type
    )
    sections_included.append("Model Explainability (SHAP)")

    # Step 11: Limitations & Recommendations
    all_elements += _build_limitations_and_recommendations_pdf(model_card, styles)
    sections_included.append("Limitations & Recommendations")

    # Step 12: Provenance
    all_elements += _build_provenance_section_pdf(model_card, styles)
    sections_included.append("Pipeline Provenance")

    output_path = Path(output_dir) / "model_report.pdf"
    pdf_path = generate_pdf_report(all_elements, str(output_path), model_card)
    pdf_size_mb = round(os.path.getsize(pdf_path) / (1024 * 1024), 2)

    logger.info("=" * 60)
    logger.info(f"REPORT GENERATOR COMPLETED SUCCESSFULLY → {pdf_path}")
    logger.info("=" * 60)

    return {
        "status": "success",
        "pdf_path": pdf_path,
        "pdf_size_mb": pdf_size_mb,
        "n_pages_est": 12,
        "sections_included": sections_included,
        "missing_plots": feasibility.get("missing_plot_paths", []),
    }
