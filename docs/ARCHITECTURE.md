# Aave V2 Wallet Credit Scoring Architecture

## Purpose

Scores Aave wallet behavior from DeFi transaction signals.

## Stack

Python, Pandas, NumPy, scikit-learn

## System Context

```mermaid
flowchart LR
    User["Aave wallet transaction JSON"] --> App["Feature engineering and scoring model"]
    App --> Data["Wallet transactions and scoring weights"]
    App --> Output["Wallet credit score CSV/report"]
    Data --> Output
```
## Runtime Workflow

```mermaid
flowchart TD
    S1["Load wallet transactions"] --> S2["Extract behavior features"]
    S2["Extract behavior features"] --> S3["Normalize feature values"]
    S3["Normalize feature values"] --> S4["Calculate wallet credit scores"]
    S4["Calculate wallet credit scores"] --> S5["Export scores and analysis"]
```
## Production Readiness Notes

- Keep secrets in environment variables and commit only .env.example templates.
- Keep generated files, dependency folders, caches, and local databases out of version control.
- Run the GitHub Actions workflow before presenting or deploying changes.
- Update this document when the source layout, dependencies, or deployment model changes.

## Review Checklist

- Architecture diagram matches current source files.
- Workflow diagram matches the main user or data path.
- README links to this architecture document.
- CI workflow validates the project on every push and pull request.

