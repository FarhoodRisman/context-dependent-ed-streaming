"""Number audit (number-integrity rule; workflow Gate 7): list every digit typed in the
results, discussion and abstract text that is not inside a macro, a table/figure
reference, a citation, a label or a section of model inputs. Prints the findings and
exits 1 if any result-like number is found."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FILES = ["paper/sections/results.tex", "paper/sections/results_alpha_sens.tex", "paper/sections/results_sens.tex",
         "paper/sections/discussion.tex", "paper/sections/abstract.tex", "paper/sections/results_opt.tex",
         "paper/sections/optimization.tex", "paper/sections/appendix_results.tex"]
# allowed literal numbers: CTAS levels, section/equation words, thresholds that are definitions
ALLOWED = {"1", "2", "3", "4", "5", "10", "95", "0.95", "0", "90", "1--3", "4--5", "3--5", "2026", "2025", "2025--2026"}
STRIP = [r"\\begin\{tabular\}.*", r"\bE\d+b?\b", r"\\setlength\{[^}]*\}\{[^}]*\}", r"\\(ref|label|cite|input|includegraphics|begin|end|caption)\{[^}]*\}", r"\\[A-Za-z]+\{\}",
         r"\\[A-Za-z]+", r"(?<!\\)%.*", r"\$[^$]*\$", r"\[[^\]]*\]"]


def main():
    bad = []
    for f in FILES:
        for i, line in enumerate((ROOT / f).read_text().splitlines(), 1):
            t = line
            for pat in STRIP:
                t = re.sub(pat, " ", t)
            for num in re.findall(r"\d+(?:[.,]\d+)*(?:--\d+)?", t):
                if num not in ALLOWED:
                    bad.append((f, i, num, line.strip()[:100]))
    for b in bad:
        print("NUMBER?", *b)
    print(f"audit: {len(bad)} typed numbers outside macros")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
