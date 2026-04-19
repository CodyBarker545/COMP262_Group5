import os
import json
import pandas as pd

from dataclasses import dataclass
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
    PageBreak,
    Preformatted,
)


@dataclass
class ReportConfig:
    artifacts_dir: str = "artifacts"
    report_filename: str = "phase2_report.pdf"


class ArtifactReportBuilder:
    def __init__(self, config: ReportConfig):
        self.config = config
        self.figures_dir = os.path.join(config.artifacts_dir, "figures")
        self.reports_dir = os.path.join(config.artifacts_dir, "reports")
        self.output_pdf = os.path.join(config.artifacts_dir, config.report_filename)

    def load_metadata(self):
        metadata_path = os.path.join(self.reports_dir, "run_metadata.json")
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_results(self):
        return pd.read_csv(os.path.join(self.reports_dir, "model_results.csv"))

    def load_classification_reports(self):
        report_path = os.path.join(self.reports_dir, "classification_reports.txt")
        with open(report_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        sections = {}
        current = None
        buffer = []

        for line in raw_text.splitlines():
            stripped = line.strip()
            if stripped in {"VADER", "TextBlob", "Naive Bayes", "MLP"}:
                if current is not None:
                    sections[current] = "\n".join(buffer).strip()
                current = stripped
                buffer = []
            else:
                buffer.append(line)

        if current is not None:
            sections[current] = "\n".join(buffer).strip()

        return sections

    def build_pdf(self):
        metadata = self.load_metadata()
        results_df = self.load_results()
        reports = self.load_classification_reports()

        doc = SimpleDocTemplate(
            self.output_pdf,
            pagesize=letter,
            rightMargin=50,
            leftMargin=50,
            topMargin=50,
            bottomMargin=50
        )

        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(
            name="SmallBody",
            parent=styles["BodyText"],
            fontSize=9,
            leading=12
        ))

        mono_style = ParagraphStyle(
            "Mono",
            parent=styles["BodyText"],
            fontName="Courier",
            fontSize=8,
            leading=10
        )

        story = []

        # Title
        story.append(Paragraph("COMP262 - Phase 2: Machine Learning Sentiment Analysis", styles["Title"]))
        story.append(Spacer(1, 0.2 * inch))
        story.append(Paragraph(
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            styles["Normal"]
        ))
        story.append(Spacer(1, 0.2 * inch))

        # Overview
        story.append(Paragraph("Project Overview", styles["Heading1"]))
        story.append(Paragraph(
            "This report was generated from previously saved model artifacts, figures, and evaluation outputs. "
            "It summarizes dataset preparation, sentiment classification model performance, and rating enhancement analysis.",
            styles["BodyText"]
        ))
        story.append(Spacer(1, 0.2 * inch))

        # Dataset info
        story.append(Paragraph("Dataset Information", styles["Heading1"]))
        dataset_info = [
            ["Metric", "Value"],
            ["Full dataset rows", str(metadata["dataset_rows"])],
            ["Balanced subset rows", str(metadata["subset_rows"])],
            ["Training rows", str(metadata["train_rows"])],
            ["Test rows", str(metadata["test_rows"])],
            ["Subset class distribution", str(metadata["subset_class_distribution"])],
            ["Training class distribution", str(metadata["train_class_distribution"])],
            ["Test class distribution", str(metadata["test_class_distribution"])],
        ]

        dataset_table = Table(dataset_info, colWidths=[2.4 * inch, 3.6 * inch])
        dataset_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9eaf7")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(dataset_table)
        story.append(Spacer(1, 0.25 * inch))

        # Model summary
        story.append(Paragraph("Model Evaluation Summary", styles["Heading1"]))
        results_data = [["Model", "Accuracy", "Precision", "Recall", "F1"]]
        for _, row in results_df.iterrows():
            results_data.append([
                str(row["Model"]),
                f"{row['Accuracy']:.4f}",
                f"{row['Precision']:.4f}",
                f"{row['Recall']:.4f}",
                f"{row['F1']:.4f}",
            ])

        results_table = Table(
            results_data,
            colWidths=[2.0 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch]
        )
        results_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9eaf7")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(results_table)
        story.append(Spacer(1, 0.25 * inch))

        # Figures
        story.append(Paragraph("Figures", styles["Heading1"]))
        figure_files = [
            ("full_rating_distribution.png", "Distribution of Ratings - Full Dataset"),
            ("subset_rating_distribution.png", "Rating Distribution - Balanced Subset"),
            ("subset_sentiment_distribution.png", "Sentiment Distribution - Balanced Subset"),
            ("nb_gridsearch.png", "Naive Bayes Grid Search Results"),
            ("nb_confusion_matrix.png", "Naive Bayes Confusion Matrix"),
            ("mlp_confusion_matrix.png", "MLP Confusion Matrix"),
            ("mlp_loss_curve.png", "MLP Training Loss Curve"),
            ("model_comparison.png", "Model Comparison Bar Chart"),
            ("all_confusion_matrices.png", "All Model Confusion Matrices"),
            ("enhanced_rating_analysis.png", "Enhanced Rating Analysis"),
        ]

        for filename, caption in figure_files:
            figure_path = os.path.join(self.figures_dir, filename)
            if os.path.exists(figure_path):
                story.append(Paragraph(caption, styles["Heading2"]))
                story.append(Image(figure_path, width=6.2 * inch, height=3.6 * inch))
                story.append(Spacer(1, 0.15 * inch))

        # Classification reports
        story.append(PageBreak())
        story.append(Paragraph("Classification Reports", styles["Heading1"]))
        for title in ["VADER", "TextBlob", "Naive Bayes", "MLP"]:
            if title in reports:
                story.append(Paragraph(title, styles["Heading2"]))
                story.append(Preformatted(reports[title], mono_style))
                story.append(Spacer(1, 0.2 * inch))

        # Key findings
        story.append(PageBreak())
        story.append(Paragraph("Key Findings", styles["Heading1"]))
        best_model = results_df.loc[results_df["F1"].idxmax(), "Model"]
        best_f1 = results_df["F1"].max()

        findings = [
            f"Best model by weighted F1: {best_model} ({best_f1:.4f})",
            "Machine learning models outperform lexicon-based methods on the same test set.",
            "The balanced real subset provides a more reliable evaluation than earlier oversampled experiments.",
            "Neutral sentiment remains the hardest class, while Positive is generally the easiest to classify.",
        ]

        for item in findings:
            story.append(Paragraph(f"- {item}", styles["BodyText"]))
            story.append(Spacer(1, 0.08 * inch))

        doc.build(story)
        print(f"PDF report saved to: {self.output_pdf}")


if __name__ == "__main__":
    config = ReportConfig()
    builder = ArtifactReportBuilder(config)
    builder.build_pdf()