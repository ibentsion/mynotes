<!-- GSD:project-start source:PROJECT.md -->
## Project

**Hebrew Handwriting OCR Pipeline**

A local Python MVP pipeline for personal Hebrew handwritten OCR, built around a human-in-the-loop workflow. It converts scanned PDF pages into region crops, flags suspicious segmentations for manual review, trains a CRNN+CTC baseline model on minimal labeled data, and logs all data versions, metrics, and artifacts to ClearML.

**Core Value:** A reviewable, labeled dataset of personal Hebrew handwriting that a baseline OCR model can train on — getting the data pipeline and annotation workflow right matters more than model accuracy at MVP.

### Constraints

- **Runtime**: Python 3.13, CPU-only for MVP — model must train without CUDA
- **Stack**: pdf2image + Poppler, OpenCV, PyTorch, Streamlit, ClearML — no additional heavy dependencies
- **Data**: Personal Hebrew notes only; privacy-sensitive — stays local
- **Reproducibility**: Git commit, package versions, and all configs stored in ClearML per run
- **Modularity**: Scripts are independent CLI tools, not a monolithic app — easy to extend or replace individual steps
- **Poppler**: Required system dependency for pdf2image on Linux
<!-- GSD:project-end -->

<!-- GSD:stack-start source:STACK.md -->
## Technology Stack

Technology stack not yet documented. Will populate after codebase mapping or first phase.
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

Conventions not yet established. Will populate as patterns emerge during development.
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

Architecture not yet mapped. Follow existing patterns found in the codebase.
<!-- GSD:architecture-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd:quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd:debug` for investigation and bug fixing
- `/gsd:execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->



<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd:profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
