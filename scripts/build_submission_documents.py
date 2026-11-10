"""Synchronize Markdown reports from the current reviewed LaTeX sources.

Run from the repository root: python -m scripts.build_submission_documents.
No experiments are rerun and older prose templates are not substituted.
"""
from scripts.document_sync import main


if __name__ == '__main__':
    main()
