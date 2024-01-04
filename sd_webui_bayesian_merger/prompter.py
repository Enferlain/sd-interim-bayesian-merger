import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from omegaconf import DictConfig, ListConfig, OmegaConf

PathT = os.PathLike

class CardDealer:
    def __init__(self, wildcards_dir: str):
        self.wildcards_dir = Path(wildcards_dir)
        self.wildcards = {}
        self.load_wildcards()

    def load_wildcards(self):
        wildcard_files = list(self.wildcards_dir.rglob("*.txt"))
        for file in wildcard_files:
            # Use relative path from wildcards directory
            relative_path = file.relative_to(self.wildcards_dir)
            # Replace slashes with underscores
            wildcard_name = str(relative_path).replace("/", "_").replace(".txt", "")
            with open(file, "r", encoding="utf-8") as f:
                 lines = f.readlines()
                 self.wildcards[wildcard_name] = [line.strip() for line in lines]

    def sample_wildcard(self, wildcard_name: str) -> str:
        if wildcard_name in self.wildcards:
            content = self.wildcards[wildcard_name]
            if content:
                # Randomly choose a line from the content
                return random.choice(content)
        return f"__{wildcard_name}__"  # Return the original wildcard if not found

    def replace_wildcards(self, prompt: str) -> str:
        chunks = re.split("(__\w+__)", prompt)
        replacements = []
        for chunk in chunks:
            if chunk.startswith("__") and chunk.endswith("__"):
                # This is a wildcard
                replacement = self.sample_wildcard(chunk[2:-2])
                replacements.append(replacement)
            else:
                # This is not a wildcard
                replacements.append(chunk)
        return "".join(replacements)

def assemble_payload(defaults: Dict, payload: Dict) -> Dict:
    for k, v in defaults.items():
        if k not in payload.keys():
            payload[k] = v
    return payload

def unpack_cargo(cargo: DictConfig) -> Tuple[Dict, Dict]:
    defaults = {}
    payloads = {}
    for k, v in cargo.items():
        if k == "cargo":
            for p_name, p in v.items():
                payloads[p_name] = OmegaConf.to_container(p)
        elif isinstance(v, (DictConfig, ListConfig)):
            defaults[k] = OmegaConf.to_container(v)
        else:
            defaults[k] = v
    return defaults, payloads

@dataclass
class Prompter:
    cfg: DictConfig

    def __post_init__(self):
        self.load_payloads()
        self.dealer = CardDealer(self.cfg.wildcards_dir)

    def load_payloads(self) -> None:
        self.raw_payloads = {}
        defaults, payloads = unpack_cargo(self.cfg.payloads)
        for payload_name, payload in payloads.items():
            self.raw_payloads[payload_name] = assemble_payload(defaults, payload)

    def render_payloads(self, batch_size: int = 0) -> Tuple[List[Dict], List[str]]:
        payloads = []
        paths = []
        for p_name, p in self.raw_payloads.items():
            for _ in range(batch_size):
                rendered_payload = p.copy()
                processed_prompt = self.dealer.replace_wildcards(p["prompt"])
                rendered_payload["prompt"] = processed_prompt
                paths.append(p_name)
                payloads.append(rendered_payload)
        return payloads, paths
