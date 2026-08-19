# Copyright 2026 DirecTune contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FixCodeGen — incremental code repair via Search/Replace.

Ported from AKG's diff_utils.py. Provides:
- CodeMatcher: 4-level match fallback (exact → trimmed → whitespace-normalized → fuzzy)
- DiffApplier: sequential modification application with conflict detection
- parse_modifications: JSON extraction from LLM output
- fix_code_with_llm: all-in-one async function (render → LLM → parse → apply)
"""

import difflib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Modification:
    """A single search/replace edit."""
    old_string: str
    new_string: str
    reason: str = ""
    replace_all: bool = False
    anchor: str = ""


@dataclass
class DiffResult:
    """Complete result of applying modifications."""
    success: bool
    modified_code: str
    original_code: str
    diff_text: str = ""
    applied_count: int = 0
    errors: List[str] = field(default_factory=list)
    raw_llm_output: str = ""
    match_levels: Dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# CodeMatcher — 4-level match fallback chain
# ---------------------------------------------------------------------------


class CodeMatcher:
    """Multi-level code matcher.

    Fallback chain:
      L1 — exact (str.find)
      L2 — trimmed line (line-by-line strip + sliding window)
      L3 — whitespace-normalized (collapse consecutive ws → single space)
      L4 — fuzzy (SequenceMatcher, with confidence gap check + window ±1 tolerance)
    """

    FUZZY_THRESHOLD = 0.8
    FUZZY_CONFIDENCE_GAP = 0.1

    @classmethod
    def find_match(cls, content: str, search: str) -> Tuple[Optional[str], str]:
        """Try each level in order. Returns (matched_text, level_name)."""
        for method, level in [
            (cls.exact_match, "exact"),
            (cls.trimmed_line_match, "trimmed"),
            (cls.whitespace_normalized_match, "whitespace_normalized"),
            (lambda c, s: cls.fuzzy_match(c, s, threshold=cls.FUZZY_THRESHOLD), "fuzzy"),
        ]:
            result = method(content, search)
            if result is not None:
                return result, level
        return None, "none"

    @classmethod
    def find_match_with_anchor(cls, content: str, search: str, anchor: str) -> Tuple[Optional[str], str]:
        """Anchor-disambiguated match: locate anchor first, then search nearby."""
        if not anchor:
            return cls.find_match(content, search)

        anchor_pos = content.find(anchor)
        if anchor_pos == -1:
            logger.warning(f"Anchor not found: '{anchor[:60]}'")
            return None, "none"

        anchor_in_search = search.find(anchor)
        if anchor_in_search > 0:
            search_start = max(0, anchor_pos - anchor_in_search)
        else:
            search_start = anchor_pos

        sub_content = content[search_start:]
        matched, level = cls.find_match(sub_content, search)
        if matched is None:
            return None, "none"
        return matched, level

    @classmethod
    def exact_match(cls, content: str, search: str) -> Optional[str]:
        if not search:
            return None
        pos = content.find(search)
        if pos == -1:
            return None
        return content[pos:pos + len(search)]

    @classmethod
    def trimmed_line_match(cls, content: str, search: str) -> Optional[str]:
        if not search:
            return None
        content_lines = content.splitlines()
        search_lines = search.splitlines()
        if not search_lines:
            return None

        stripped_search = [line.strip() for line in search_lines]
        if all(s == "" for s in stripped_search):
            return None

        window_size = len(search_lines)
        if window_size > len(content_lines):
            return None

        for i in range(len(content_lines) - window_size + 1):
            window = content_lines[i:i + window_size]
            stripped_window = [line.strip() for line in window]
            if stripped_window == stripped_search:
                return "\n".join(window)
        return None

    @classmethod
    def whitespace_normalized_match(cls, content: str, search: str) -> Optional[str]:
        if not search:
            return None

        def normalize(s: str) -> str:
            return re.sub(r'\s+', ' ', s).strip()

        norm_search = normalize(search)
        if not norm_search:
            return None

        content_lines = content.splitlines()
        search_lines = search.splitlines()
        window_size = len(search_lines)
        if window_size == 0 or window_size > len(content_lines):
            return None

        for i in range(len(content_lines) - window_size + 1):
            window = "\n".join(content_lines[i:i + window_size])
            if normalize(window) == norm_search:
                return window
        return None

    @classmethod
    def fuzzy_match(cls, content: str, search: str, threshold: float = 0.8) -> Optional[str]:
        if not search or not content:
            return None

        content_lines = content.splitlines()
        search_lines = search.splitlines()
        search_line_count = len(search_lines)
        if search_line_count == 0:
            return None

        best_match: Optional[str] = None
        best_ratio = 0.0
        second_best_ratio = 0.0

        for delta in [0, -1, 1]:
            window_size = search_line_count + delta
            if window_size <= 0 or window_size > len(content_lines):
                continue
            for i in range(len(content_lines) - window_size + 1):
                window = "\n".join(content_lines[i:i + window_size])
                ratio = difflib.SequenceMatcher(None, search, window).ratio()
                if ratio > best_ratio:
                    second_best_ratio = best_ratio
                    best_ratio = ratio
                    best_match = window
                elif ratio > second_best_ratio:
                    second_best_ratio = ratio

        if best_ratio < threshold:
            return None
        if best_ratio - second_best_ratio < cls.FUZZY_CONFIDENCE_GAP:
            logger.warning(
                f"Fuzzy match rejected: confidence gap too small "
                f"(best={best_ratio:.3f}, second={second_best_ratio:.3f})"
            )
            return None
        return best_match


# ---------------------------------------------------------------------------
# DiffApplier
# ---------------------------------------------------------------------------


class DiffApplier:
    """Applies a list of Modifications to code sequentially.

    Supports replace_all, anchor disambiguation, conflict pre-detection,
    and match-level tracking.
    """

    @classmethod
    def apply_modifications(
        cls, code: str, modifications: List[Modification], raw_llm_output: str = "",
    ) -> DiffResult:
        original_code = code
        current_code = code
        applied_count = 0
        errors: List[str] = []
        match_levels: Dict[str, int] = {}

        conflict_warnings = cls._detect_conflicts(modifications)
        for w in conflict_warnings:
            logger.warning(w)

        for idx, mod in enumerate(modifications):
            if mod.old_string == mod.new_string:
                errors.append(f"Mod {idx + 1}: old_string == new_string, skipped")
                continue

            matched_text, level = CodeMatcher.find_match_with_anchor(
                current_code, mod.old_string, mod.anchor,
            )
            match_levels[level] = match_levels.get(level, 0) + 1

            if matched_text is None:
                if mod.anchor and current_code.find(mod.anchor) == -1:
                    errors.append(f"Mod {idx + 1}: anchor not found (anchor: '{mod.anchor[:60]}')")
                else:
                    errors.append(
                        f"Mod {idx + 1}: no match in code "
                        f"(old_string head 60: '{mod.old_string[:60]}...')"
                    )
                continue

            if mod.replace_all:
                count = current_code.count(matched_text)
                current_code = current_code.replace(matched_text, mod.new_string)
                applied_count += count
            elif mod.anchor:
                anchor_pos = current_code.find(mod.anchor)
                anchor_in_old = mod.old_string.find(mod.anchor)
                search_start = max(0, anchor_pos - anchor_in_old) if anchor_in_old > 0 else anchor_pos
                sub_content = current_code[search_start:]
                replaced_sub = sub_content.replace(matched_text, mod.new_string, 1)
                current_code = current_code[:search_start] + replaced_sub
                applied_count += 1
            else:
                current_code = current_code.replace(matched_text, mod.new_string, 1)
                applied_count += 1

        diff_text = cls._generate_diff(original_code, current_code)
        success = applied_count > 0
        return DiffResult(
            success=success,
            modified_code=current_code,
            original_code=original_code,
            diff_text=diff_text,
            applied_count=applied_count,
            errors=errors,
            raw_llm_output=raw_llm_output,
            match_levels=match_levels,
        )

    @staticmethod
    def _detect_conflicts(modifications: List[Modification]) -> List[str]:
        warnings: List[str] = []
        for i, mod_a in enumerate(modifications):
            for j, mod_b in enumerate(modifications):
                if i >= j:
                    continue
                if mod_a.old_string in mod_b.old_string or mod_b.old_string in mod_a.old_string:
                    warnings.append(
                        f"Mod {i + 1} and {j + 1} may conflict (overlapping old_string)"
                    )
        return warnings

    @staticmethod
    def _generate_diff(original: str, modified: str) -> str:
        original_lines = original.splitlines(keepends=True)
        modified_lines = modified.splitlines(keepends=True)
        diff = difflib.unified_diff(
            original_lines, modified_lines,
            fromfile="original", tofile="modified",
        )
        return "".join(diff)


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


def _extract_json_object(text: str) -> str | None:
    """Extract the outermost balanced JSON object/array from a string that may
    be wrapped in prose. Model-robustness helper: ds4-flash sometimes prefixes
    explanations before the JSON payload (GLM-5.2 does not)."""
    start = -1
    depth = 0
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if start == -1:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                return text[start : i + 1]
        elif ch == "[":
            if start == -1:
                start = i
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0 and start != -1:
                return text[start : i + 1]
    return None


def parse_modifications(llm_output: str) -> List[Modification]:
    """Extract Modification list from LLM JSON output.

    Supports:
    1. Full JSON: {"analysis": "...", "modifications": [...], "summary": "..."}
    2. Bare array: [{"old_string": "...", "new_string": "..."}]

    Tolerates markdown ```json``` wrapping.
    """
    text = llm_output.strip()

    # Strip markdown code fence
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        # 模型差异鲁棒性（ds4-flash 实测高频）：LLM 在 JSON 前加了解释性文字（"好的，分析如下："
        # 或 ```json 围栏前有 prose）。退回从文本中提取最外层 {...} 再解析一次。
        # 只做一次子串尝试，失败则放弃（不无限重试）。
        if not (text.startswith("{") or text.startswith("[")):
            braced = _extract_json_object(text)
            if braced is not None:
                try:
                    data = json.loads(braced)
                except json.JSONDecodeError:
                    logger.warning(f"JSON parse failed: {e}")
                    return []
            else:
                logger.warning(f"JSON parse failed: {e}")
                return []
        else:
            logger.warning(f"JSON parse failed: {e}")
            return []

    if isinstance(data, dict):
        mods_raw = data.get("modifications", [])
    elif isinstance(data, list):
        mods_raw = data
    else:
        logger.warning(f"Unexpected JSON top-level type: {type(data)}")
        return []

    modifications = []
    for item in mods_raw:
        if not isinstance(item, dict):
            continue
        old = item.get("old_string")
        new = item.get("new_string")
        if old is None or new is None:
            logger.warning(f"Skipping mod item missing old_string/new_string: {item}")
            continue
        modifications.append(Modification(
            old_string=old,
            new_string=new,
            reason=item.get("reason", ""),
            replace_all=bool(item.get("replace_all", False)),
            anchor=str(item.get("anchor", "")),
        ))
    return modifications


def truncate_error_log(error_log: str, max_len: int = 5000) -> str:
    """Truncate long error logs, keeping head 1/3 + tail 2/3.

    Traceback key info (actual error type + recent frames) is at the tail,
    so the tail gets more space.
    """
    if len(error_log) <= max_len:
        return error_log
    head_len = max_len // 3
    tail_len = max_len - head_len - 50
    return (
        error_log[:head_len]
        + f"\n\n... ({len(error_log) - head_len - tail_len} chars truncated) ...\n\n"
        + error_log[-tail_len:]
    )


# ---------------------------------------------------------------------------
# All-in-one async fix function
# ---------------------------------------------------------------------------


async def fix_code_with_llm(
    client,
    model: str,
    fix_code_gen_template: str,
    original_code: str,
    error_log: str,
    conductor_suggestion: str = "",
    temperature: float = 0.1,
) -> Optional[DiffResult]:
    """Render FixCodeGen prompt, call LLM, parse JSON, apply modifications.

    Args:
        client: AsyncOpenAI client
        model: model name to use
        fix_code_gen_template: prompt text with {original_code}, {error_log},
                               {conductor_suggestion} placeholders
        original_code: the code to repair
        error_log: error message / traceback
        conductor_suggestion: Conductor's analysis and fix suggestion
        temperature: LLM sampling temperature (default 0.1 for precise edits)

    Returns:
        DiffResult on success (with modified_code), None on LLM failure.
    """
    # Render prompt
    user_prompt = fix_code_gen_template.replace("{original_code}", original_code)
    user_prompt = user_prompt.replace("{error_log}", truncate_error_log(error_log))
    user_prompt = user_prompt.replace("{conductor_suggestion}", conductor_suggestion or "(无)")

    messages = [{"role": "user", "content": user_prompt}]

    # Call LLM with retry
    content = None
    for attempt in range(3):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=4096,
                timeout=120,
            )
            content = resp.choices[0].message.content
            break
        except Exception as e:
            logger.warning(f"FixCodeGen LLM call attempt {attempt + 1} failed: {e}")
            if attempt == 2:
                return None
            import asyncio
            await asyncio.sleep(2 ** attempt)

    if content is None:
        return None

    # Parse modifications from LLM output
    modifications = parse_modifications(content)
    if not modifications:
        logger.warning("FixCodeGen: no valid modifications parsed from LLM output")
        return DiffResult(
            success=False, modified_code=original_code, original_code=original_code,
            errors=["No valid modifications parsed"],
            raw_llm_output=content,
        )

    # Apply modifications
    result = DiffApplier.apply_modifications(original_code, modifications, raw_llm_output=content)
    return result
