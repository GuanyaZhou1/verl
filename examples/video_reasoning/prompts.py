#!/usr/bin/env python3
"""
Shared prompt templates for video reasoning data preprocessing.

Usage in preprocessing scripts:
    from prompts import MULTITURN_SYSTEM_PROMPT, OUTPUT_TEMPLATE, OUTPUT_TEMPLATE_OPENENDED

Or load from file:
    python script.py --prompt_file /path/to/custom_prompt.txt
"""

# Default system prompt for multi-turn video reasoning
MULTITURN_SYSTEM_PROMPT = """You should reason step by step and, in EACH step, FIRST analyze and THEN focus on specific video segments. Place the grounded time segments at the END of the step.

Each reasoning step must be enclosed within '<think>' tags and reference specific time segments.

<think>
{Single reasoning step — analyze the question; summarize relevant findings from the currently available sampled input and any previously inspected segments; brainstorm hypotheses; verify whether current evidence is sufficient; refine errors; revisit prior steps if needed; if insufficient to answer, decide the NEXT most informative segments to inspect based on question intent and previously seen content}
</think>

When identifying relevant segments, use '<segment>' tags with time ranges in seconds:

<segment>
[(start1, end1), (start2, end2), ...]
</segment>

Your reasoning should be grounded in visual spatiotemporal evidence from the video. When mentioning any objects related to the evidence, strictly follow this format:
<obj>object_name</obj><box>[x1,y1,x2,y2]</box>at<t>time_in_seconds</t>

When ready to provide the final answer, enclose it within '<answer>' tags:

<answer> {final answer} </answer>
"""

# Single-turn system prompt (no multi-turn, no segment/bbox grounding)
SINGLETURN_SYSTEM_PROMPT = """You are a helpful assistant. The user asks a question, and then you solves it.

Please first think deeply about the question based on the given video, and then provide the final answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."""

# Output template for multiple-choice questions
OUTPUT_TEMPLATE = "Please provide only the single option (e.g., A, B, C, D, etc.) within the <answer> </answer> tags."

# Output template for open-ended questions
OUTPUT_TEMPLATE_OPENENDED = "Please provide your answer within the <answer> </answer> tags."

# Prompt version mapping
PROMPT_VERSIONS = {
    "default": MULTITURN_SYSTEM_PROMPT,
    "multiturn": MULTITURN_SYSTEM_PROMPT,
    "singleturn": SINGLETURN_SYSTEM_PROMPT,
}


def load_prompt_from_file(filepath: str) -> str:
    """Load prompt from a text file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read().strip()


def get_prompts(prompt_file: str = None, prompt_version: str = "default", output_template_file: str = None, output_template_openended_file: str = None):
    """
    Get prompts, optionally loading from files or using preset versions.

    Args:
        prompt_file: Path to custom system prompt file (optional, takes precedence)
        prompt_version: Preset prompt version: "default"/"multiturn" or "singleturn"
        output_template_file: Path to custom multiple-choice output template (optional)
        output_template_openended_file: Path to custom open-ended output template (optional)

    Returns:
        tuple: (system_prompt, output_template, output_template_openended)
    """
    if prompt_file:
        system_prompt = load_prompt_from_file(prompt_file)
    else:
        system_prompt = PROMPT_VERSIONS.get(prompt_version, MULTITURN_SYSTEM_PROMPT)

    output_tmpl = load_prompt_from_file(output_template_file) if output_template_file else OUTPUT_TEMPLATE
    output_tmpl_open = load_prompt_from_file(output_template_openended_file) if output_template_openended_file else OUTPUT_TEMPLATE_OPENENDED

    return system_prompt, output_tmpl, output_tmpl_open
