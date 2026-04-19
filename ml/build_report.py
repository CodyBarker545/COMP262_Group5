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
    artifacts_dir: str = str((__file__ and os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")))
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

    def load_optional_llm_outputs(self):
        llm_path = os.path.join(self.reports_dir, "llm_outputs.json")
        if not os.path.exists(llm_path):
            return None
        with open(llm_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def add_bullet_lines(self, story, styles, lines):
        for line in lines:
            story.append(Paragraph(f"- {line}", styles["BodyText"]))
            story.append(Spacer(1, 0.08 * inch))

    def add_preformatted_block(self, story, title, content, styles, mono_style):
        story.append(Paragraph(title, styles["Heading2"]))
        story.append(Preformatted(content.strip(), mono_style))
        story.append(Spacer(1, 0.18 * inch))

    def build_pdf(self):
        metadata = self.load_metadata()
        results_df = self.load_results()
        reports = self.load_classification_reports()
        llm_outputs = self.load_optional_llm_outputs()

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

        # Methodology and preprocessing justification
        story.append(Paragraph("Methodology and Preprocessing", styles["Heading1"]))
        self.add_bullet_lines(story, styles, [
            "A balanced real subset was selected from the Amazon Gift Cards review dataset so that the model evaluation was not dominated by the positive class.",
            "The review text was lowercased to normalize vocabulary and reduce duplicate tokens that differ only by casing.",
            "Contractions such as n't were expanded so negation could be captured more reliably during preprocessing.",
            "Negation handling was applied so short windows after words like not and no were marked with a NOT_ prefix, helping the models distinguish phrases such as not good from good.",
            "Non-alphabetic characters were removed while preserving exclamation marks because they can signal stronger sentiment intensity.",
            "Stopwords were removed to reduce noise, but key negation words were retained so sentiment polarity was not lost.",
            "Lemmatization was used to reduce inflected forms to a common base word and improve generalization.",
        ])
        story.append(Spacer(1, 0.12 * inch))

        story.append(Paragraph("Text Representation Choice", styles["Heading2"]))
        story.append(Paragraph(
            "TF-IDF was selected because it provides a strong and interpretable baseline for review classification, works well with Naive Bayes and MLP, "
            "and highlights informative unigrams and bigrams without requiring a large neural language model for feature extraction.",
            styles["BodyText"]
        ))
        story.append(Spacer(1, 0.18 * inch))

        # Dataset info
        story.append(Paragraph("Dataset Information", styles["Heading1"]))
        dataset_info = [
            ["Metric", "Value"],
            ["Full dataset rows", str(metadata["dataset_rows"])],
            ["Balanced subset rows", str(metadata["subset_rows"])],
            ["Training rows", str(metadata["train_rows"])],
            ["Test rows", str(metadata["test_rows"])],
            ["Subset class distribution", str(metadata["subset_class_distribution"])],
            ["Subset rating distribution", str(metadata.get("subset_rating_distribution", "Not saved in this run"))],
            ["Training class distribution", str(metadata["train_class_distribution"])],
            ["Test class distribution", str(metadata["test_class_distribution"])],
            ["Training rating distribution", str(metadata.get("train_rating_distribution", "Not saved in this run"))],
            ["Test rating distribution", str(metadata.get("test_rating_distribution", "Not saved in this run"))],
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

        story.append(Paragraph("Training Process Results", styles["Heading1"]))
        self.add_bullet_lines(story, styles, [
            f"Naive Bayes best parameters: {metadata['nb_best_params']}",
            f"Naive Bayes best cross-validation weighted F1: {metadata['nb_best_cv_f1']:.4f}",
            f"MLP best parameters: {metadata['mlp_best_params']}",
            f"MLP best cross-validation weighted F1: {metadata['mlp_best_cv_f1']:.4f}",
            "The training process used GridSearchCV with stratified folds to tune each model before final evaluation on the held-out 30% test set.",
        ])
        story.append(Spacer(1, 0.18 * inch))

        story.append(Paragraph("Lexicon Versus Machine Learning Experiment Design", styles["Heading1"]))
        story.append(Paragraph(
            "To ensure an apples-to-apples comparison, the lexicon-based methods and the machine learning models were all evaluated on the exact same held-out test split. "
            "The gold labels came from the rating-derived sentiment field, and VADER, TextBlob, Naive Bayes, and MLP were then scored with the same accuracy, precision, recall, and weighted F1 metrics.",
            styles["BodyText"]
        ))
        story.append(Spacer(1, 0.18 * inch))

        story.append(Paragraph("Rating Enhancement Using Review Text", styles["Heading1"]))
        story.append(Paragraph(
            "One option suggested by the recommender-systems literature is to refine a user rating by blending the original star value with a sentiment score estimated from the review text. "
            "In this project, the VADER compound score is normalized to the same range as the original star rating and then combined with the original score using a weighted average.",
            styles["BodyText"]
        ))
        story.append(Spacer(1, 0.1 * inch))

        diagram = """
Original review text -> VADER sentiment score -> normalize to [1, 5]
                                 |
                                 v
Original star rating --------> weighted blend --------> enhanced rating
"""
        pseudocode = """
for each review:
    compound = vader(review_text).compound
    normalized_sentiment = ((compound + 1) / 2) * 4 + 1
    normalized_star = original_star
    enhanced_rating = alpha * normalized_star + (1 - alpha) * normalized_sentiment
    save enhanced_rating
"""
        self.add_preformatted_block(story, "Diagram", diagram, styles, mono_style)
        self.add_preformatted_block(story, "Pseudocode", pseudocode, styles, mono_style)
        story.append(Paragraph(
            "The saved enhanced-rating outputs show how textual sentiment can reduce or reinforce the original star value. "
            "Large disagreements indicate reviews where the text tone and rating are not well aligned.",
            styles["BodyText"]
        ))
        story.append(Spacer(1, 0.18 * inch))

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

        # LLM tasks
        story.append(PageBreak())
        story.append(Paragraph("LLM Tasks", styles["Heading1"]))
        if llm_outputs is None:
            story.append(Paragraph(
                "LLM results file not found. Run run_phase2_llm_tasks.py to generate the 10 summaries and the service representative response, "
                "then rebuild this report to include the outputs for requirements 16 and 17.",
                styles["BodyText"]
            ))
            story.append(Spacer(1, 0.18 * inch))
        else:
            story.append(Paragraph("Requirement 16 - Summaries", styles["Heading2"]))
            story.append(Paragraph(
                f"The summarization model used was {llm_outputs['model']}. Ten reviews longer than 100 words were selected, and the first two are shown below.",
                styles["BodyText"]
            ))
            story.append(Spacer(1, 0.1 * inch))

            for index, item in enumerate(llm_outputs["summary_task"]["first_two_for_report"], start=1):
                story.append(Paragraph(f"Review {index} Original Text", styles["Heading3"]))
                story.append(Paragraph(item["original_text"], styles["SmallBody"]))
                story.append(Spacer(1, 0.08 * inch))
                story.append(Paragraph(f"Review {index} 50-Word Summary", styles["Heading3"]))
                story.append(Paragraph(item["summary"], styles["BodyText"]))
                story.append(Spacer(1, 0.16 * inch))

            story.append(Paragraph("Requirement 17 - Service Representative Response", styles["Heading2"]))
            story.append(Paragraph("Selected customer review:", styles["Heading3"]))
            story.append(Paragraph(
                llm_outputs["service_response_task"]["selected_review"]["reviewText"],
                styles["SmallBody"]
            ))
            story.append(Spacer(1, 0.08 * inch))
            story.append(Paragraph("Generated response:", styles["Heading3"]))
            story.append(Paragraph(llm_outputs["service_response_task"]["response"], styles["BodyText"]))
            story.append(Spacer(1, 0.18 * inch))

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
