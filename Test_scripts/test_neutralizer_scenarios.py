"""Test script to evaluate end-to-end rewriting scenarios."""

from src.classification.detector import ToxicityDetector
from src.rewriting.neutralizer import TextNeutralizer

def run_tests():
    print("Loading detector and neutralizer...")
    detector = ToxicityDetector(model_key="bigru")
    detector.load()

    neutralizer = TextNeutralizer()
    neutralizer.load()

    test_cases = [
        ("You are such a shit, improve your work.", "Abusive feedback with request"),
        ("You are a complete idiot and nobody cares what you think.", "Pure personal attack"),
        ("This code is trash, fix it.", "Code critique with request"),
        ("Shut up and listen to me.", "Hostile demand"),
        ("Your explanation is stupid and makes no sense.", "Critique of reasoning"),
        ("I disagree with your proposal because latency is too high.", "Legitimate safe disagreement"),
        ("You are a pathetic loser who knows nothing.", "Unseen personal attack"),
        ("Stop writing terrible documentation, clean this up.", "Unseen feedback request"),
        ("I will find you and hurt you.", "Explicit violent threat"),
    ]

    print("\n" + "="*80)
    print("EVALUATION RESULTS:")
    print("="*80)

    for text, category in test_cases:
        det_res = detector.predict(text)
        is_toxic = det_res["is_toxic"]
        orig_score = det_res["probability"]

        if not is_toxic:
            print(f"\n[SAFE INPUT - NO REWRITE NEEDED]")
            print(f"Category:     {category}")
            print(f"Input Text:   {text}")
            print(f"Toxicity:     {orig_score:.4f} (Safe < {det_res['threshold']:.2f})")
            continue

        rewrite_res = neutralizer.rewrite(
            text,
            detector_instance=detector,
            original_prob=orig_score,
            threshold=det_res["threshold"]
        )

        print(f"\n[{rewrite_res['status']}] ({rewrite_res['status_label']})")
        print(f"Category:     {category}")
        print(f"Input:        {text}")
        print(f"Rewrite:      {rewrite_res['display_text']}")
        print(f"Scores:       {orig_score:.4f} -> {rewrite_res['post_toxicity'] if rewrite_res['post_toxicity'] is not None else 'N/A'}")
        if rewrite_res.get("intent_warning"):
            print(f"Warning:      {rewrite_res['intent_warning']}")
        print(f"Message:      {rewrite_res['message']}")

if __name__ == "__main__":
    run_tests()

