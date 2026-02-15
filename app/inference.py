from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer


@dataclass
class Span:
    start_char: int
    end_char: int
    text: str


class IdiomSpanDetector:
    def __init__(self, model_dir: str = "model", device: Optional[str] = None):
        self.model_dir = model_dir

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
        self.model.eval()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)

        # Prefer model config label maps if present
        cfg = self.model.config
        self.id2label = getattr(cfg, "id2label", None) or {}
        self.label2id = getattr(cfg, "label2id", None) or {}

        # If model was saved with generic labels, override with BIO meaning
        if set(self.id2label.values()) == {"LABEL_0", "LABEL_1", "LABEL_2"}:
            self.id2label = {0: "O", 1: "B-IDIOM", 2: "I-IDIOM"}

    @torch.inference_mode()
    def predict(self, text: str, max_length: int = 256) -> Dict[str, Any]:
        if not text or not text.strip():
            return {"tokens": [], "labels": [], "spans": []}

        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,  # requires fast tokenizer (you have tokenizer.json ✅)
        )

        offsets = enc.pop("offset_mapping")[0].tolist()  # (seq_len, 2)
        enc = {k: v.to(self.device) for k, v in enc.items()}

        logits = self.model(**enc).logits[0]            # (seq_len, num_labels)
        pred_ids = torch.argmax(logits, dim=-1).tolist()

        input_ids = enc["input_ids"][0].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        labels = [self.id2label.get(i, str(i)) for i in pred_ids]

        spans = self._bio_to_spans(text, labels, offsets)
        idioms = [s.text for s in spans]
        return {
            "tokens": tokens,
            "labels": labels,
            "spans": [s.__dict__ for s in spans],
            "device": str(self.device),
            "max_length": max_length,
        }

    def _bio_to_spans(self, text: str, labels: List[str], offsets: List[List[int]]) -> List[Span]:
        """
        Convert token-level BIO labels + char offsets into contiguous spans.
        Assumes labels like: B-IDIOM, I-IDIOM, O (or similar).
        """
        spans: List[Span] = []

        active_start: Optional[int] = None
        active_end: Optional[int] = None

        for lab, (start, end) in zip(labels, offsets):
            # Special tokens often have (0,0)
            if start == 0 and end == 0:
                continue

            is_b = lab.startswith("B-")
            is_i = lab.startswith("I-")

            if is_b:
                # close previous
                if active_start is not None and active_end is not None and active_end > active_start:
                    spans.append(Span(active_start, active_end, text[active_start:active_end]))

                active_start, active_end = start, end

            elif is_i and active_start is not None:
                # extend current span
                active_end = end

            else:
                # O or other label: close span if any
                if active_start is not None and active_end is not None and active_end > active_start:
                    spans.append(Span(active_start, active_end, text[active_start:active_end]))
                active_start, active_end = None, None

        # close at end
        if active_start is not None and active_end is not None and active_end > active_start:
            spans.append(Span(active_start, active_end, text[active_start:active_end]))

        return spans
