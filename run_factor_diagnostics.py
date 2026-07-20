from pathlib import Path

from factor_diagnostics import render_report, run


def main():
    root = Path(__file__).resolve().parent
    panel, summary = run(root)
    reports = root / "reports"
    reports.mkdir(exist_ok=True)
    panel.to_csv(reports / "factor_panel.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(reports / "factor_summary.csv", index=False, encoding="utf-8-sig")
    print(render_report(summary))


if __name__ == "__main__":
    main()
