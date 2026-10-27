"""Neural rewriting module using fine-tuned T5 for toxic-to-neutral text conversion."""

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.config import (
    REWRITER_INSTRUCTION,
    REWRITER_MAX_INPUT_LEN,
    REWRITER_MAX_OUTPUT_LEN,
    REWRITER_MODEL_DIR,
    get_torch_device,
)
from src.preprocessing.cleaner import clean_for_rewriter, validate_input_text
from src.rewriting.postprocessor import analyze_rewrite_quality


class TextNeutralizer:
    """Wrapper for fine-tuned T5 text neutralizer with beam search decoding and quality verification."""

    def __init__(
        self,
        model_dir: Optional[Union[str, Path]] = None,
        prefix: str = REWRITER_INSTRUCTION,
        max_input_length: int = REWRITER_MAX_INPUT_LEN,
        max_output_length: int = REWRITER_MAX_OUTPUT_LEN,
        num_beams: int = 4,
    ):
        self.model_dir = Path(model_dir or REWRITER_MODEL_DIR)
        self.prefix = prefix
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.num_beams = num_beams
        self.tokenizer = None
        self.model = None
        self.device = get_torch_device()

    def load(self) -> None:
        """Load T5 model and tokenizer."""
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Rewriter model directory not found at: {self.model_dir}. "
                "Ensure Git LFS pull has been run."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.model = AutoModelForSeq2SeqLM.from_pretrained(str(self.model_dir))
        self.model.to(self.device)
        self.model.eval()

    def _format_prompt(self, text: str) -> str:
        """Format the input text with the instruction prompt."""
        cleaned = text.strip()
        # If the prefix already ends with 'Rewritten:', use directly
        if "Message:" in self.prefix:
            return f"{self.prefix} \"{cleaned}\"\nRewritten:"
        return f"{self.prefix}\n\nMessage: \"{cleaned}\"\nRewritten:"

    def _generate(
        self,
        text: str,
        num_beams: int = 4,
        no_repeat_ngram_size: int = 3,
        temperature: float = 1.0,
        do_sample: bool = False,
        length_penalty: float = 1.0,
    ) -> str:
        """Internal generation call."""
        prompt = self._format_prompt(text)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_length,
        ).to(self.device)

        gen_kwargs = {
            "max_new_tokens": self.max_output_length,
            "early_stopping": True,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "length_penalty": length_penalty,
        }

        if do_sample:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.90
        else:
            gen_kwargs["num_beams"] = num_beams

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        decoded = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        # Clean output formatting
        if decoded.startswith('"') and decoded.endswith('"') and len(decoded) > 2:
            decoded = decoded[1:-1].strip()
        if decoded.startswith("'") and decoded.endswith("'") and len(decoded) > 2:
            decoded = decoded[1:-1].strip()

        if decoded:
            decoded = decoded[0].upper() + decoded[1:] if len(decoded) > 1 else decoded.upper()

        return decoded

    def rewrite(
        self,
        text: str,
        detector_instance: Optional[Any] = None,
        original_prob: Optional[float] = None,
        threshold: float = 0.40,
    ) -> Dict[str, Any]:
        """Neutralize toxic input text with validation, bounded retry, and quality verification.
        
        Args:
            text: Input text string.
            detector_instance: Optional ToxicityDetector to re-score the generated text.
            original_prob: Toxicity probability of input text.
            threshold: Toxicity threshold for classifying as toxic.
            
        Returns:
            Dict containing rewritten text, quality status, toxicity metrics, and warnings.
        """
        if self.model is None or self.tokenizer is None:
            self.load()

        is_valid, err_msg = validate_input_text(text)
        if not is_valid:
            return {
                "original_text": text,
                "rewritten_text": None,
                "display_text": "Could not generate a reliable rewrite",
                "status": "INVALID_INPUT",
                "is_acceptable": False,
                "message": err_msg,
                "original_toxicity": original_prob,
                "post_toxicity": None,
            }

        cleaned_input = clean_for_rewriter(text)

        # Primary generation with standard beam search
        rewritten = self._generate(cleaned_input, num_beams=self.num_beams)

        # Compute original probability if not provided
        if original_prob is None and detector_instance is not None:
            orig_eval = detector_instance.predict(text, threshold=threshold)
            original_prob = orig_eval["probability"]
        elif original_prob is None:
            original_prob = 1.0

        # Secondary check: re-score generated text with detector
        post_prob = None
        if detector_instance is not None and rewritten:
            post_eval = detector_instance.predict(rewritten, threshold=threshold)
            post_prob = post_eval["probability"]

        # Run quality analysis (checks grammar, incomplete phrases, repetition, message loss)
        quality = analyze_rewrite_quality(
            original_text=text,
            rewritten_text=rewritten,
            original_toxicity_prob=original_prob,
            post_toxicity_prob=post_prob,
            threshold=threshold,
        )

        # Bounded retry: if quality check failed, try one alternative generation attempt
        if not quality["is_acceptable"]:
            retry_rewritten = self._generate(
                cleaned_input,
                num_beams=3,
                no_repeat_ngram_size=2,
                length_penalty=1.2,
            )
            if retry_rewritten and retry_rewritten != rewritten:
                retry_post_prob = None
                if detector_instance is not None:
                    retry_eval = detector_instance.predict(retry_rewritten, threshold=threshold)
                    retry_post_prob = retry_eval["probability"]

                retry_quality = analyze_rewrite_quality(
                    original_text=text,
                    rewritten_text=retry_rewritten,
                    original_toxicity_prob=original_prob,
                    post_toxicity_prob=retry_post_prob,
                    threshold=threshold,
                )
                if retry_quality["is_acceptable"]:
                    rewritten = retry_rewritten
                    post_prob = retry_post_prob
                    quality = retry_quality

        # If after retry the output remains unacceptable, enforce reliable failure state
        final_rewrite_text = rewritten if quality["is_acceptable"] else None
        display_output = quality["display_text"]

        # Re-score strictly on final_rewrite_text to guarantee identical text is scored
        if quality["is_acceptable"] and detector_instance is not None and final_rewrite_text:
            final_eval = detector_instance.predict(final_rewrite_text, threshold=threshold)
            post_prob = final_eval["probability"]
        elif not quality["is_acceptable"]:
            post_prob = None

        score_decrease = (original_prob - post_prob) if (original_prob is not None and post_prob is not None) else None

        return {
            "original_text": text,
            "rewritten_text": final_rewrite_text,
            "display_text": display_output,
            "status": quality["status"],
            "status_label": quality.get("status_label", "Suggested rewrite" if quality["is_acceptable"] else "Rewrite unavailable"),
            "is_acceptable": quality["is_acceptable"],
            "message": quality["message"],
            "intent_warning": quality.get("intent_warning", ""),
            "suggestion": quality.get("suggestion", ""),
            "original_toxicity": original_prob,
            "post_toxicity": post_prob,
            "score_decrease": score_decrease,
            "toxicity_reduction": score_decrease,
        }
