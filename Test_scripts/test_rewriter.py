"""Script to test text neutralizers.
Evaluates the fine-tuned T5 model and reports rewriting quality.
"""

from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

BASE_DIR = Path(__file__).resolve().parent.parent
T5_DIR = BASE_DIR / "models" / "t5_neutralizer"


def test_t5_rewriter():
    print("=" * 60)
    print("Testing Fine-Tuned T5 Neutralizer")
    print("=" * 60)

    if not T5_DIR.exists():
        print(f"T5 directory not found at: {T5_DIR}")
        return

    device = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    tok = AutoTokenizer.from_pretrained(str(T5_DIR))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(T5_DIR))
    model.to(device)
    model.eval()

    prefix = "Rewrite neutrally: "
    sample_texts = [
        "I hate you and your stupid behavior.",
        "Shut up you idiot, you know nothing about this.",
        "This is the most garbage code I have ever seen in my life.",
        "Your proposal is ridiculous and only a moron would suggest it.",
        "The project deadlines were missed due to poor coordination."
    ]

    for s in sample_texts:
        inputs = tok(prefix + s, return_tensors="pt", truncation=True, max_length=64).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                num_beams=5,
                no_repeat_ngram_size=3,
                max_new_tokens=64,
                early_stopping=True
            )
        neutralized = tok.decode(out[0], skip_special_tokens=True).strip()
        print(f"\nINPUT : {s}")
        print(f"OUTPUT: {neutralized}")


if __name__ == "__main__":
    test_t5_rewriter()
