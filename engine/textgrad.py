"""
engine/textgrad.py
==================
TextGrad: Automatic "Differentiation" via Text.

A minimal from-scratch implementation inspired by the TextGrad paper
(Yuksekgonul et al., Nature 2025). Provides PyTorch-like primitives for
optimizing text via LLM-generated textual gradients.

Core classes:
    TextVariable -- a mutable text string that tracks gradients
    TextLoss     -- evaluates text quality using LLM as judge
    TGD          -- Textual Gradient Descent optimizer

How it works (vs PyTorch):
    PyTorch:  tensor -> forward pass -> loss.backward() -> optimizer.step()
    TextGrad: text   -> LLM generates -> loss.backward() -> optimizer.step()

    The key difference: in PyTorch, gradients are numerical derivatives
    computed via chain rule. In TextGrad, gradients are *structured text
    critiques* produced by an LLM that analyzes what went wrong.

    The LLM plays three roles:
    1. Forward pass: generates code/text (done by execute_node)
    2. Loss function: evaluates quality (TextLoss.__call__)
    3. Update rule: applies gradient to fix the text (TGD._apply_gradient)
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable

from openai import OpenAI

from engine.interfaces import Mutation, TextGradient

logger = logging.getLogger("recurseforge.engine.textgrad")


# ---------------------------------------------------------------------------
# TextVariable
# ---------------------------------------------------------------------------

class TextVariable:
    """
    A mutable text value that can accumulate gradients.

    Analogous to torch.Tensor with requires_grad=True.

    Attributes:
        value: The current text content (code or prompt).
        role_description: What this variable represents.
        requires_grad: If True, this variable can be optimized.
        grad: The accumulated gradient (a TextGradient object).
        history: List of previous values (for debugging/logging).
    """

    def __init__(
        self,
        value: str,
        role_description: str = "text",
        requires_grad: bool = False,
    ):
        self.value = value
        self.role_description = role_description
        self.requires_grad = requires_grad
        self.grad: TextGradient | None = None
        self.history: list[str] = [value]

    def set_value(self, new_value: str) -> None:
        """Update the text value and record in history."""
        self.value = new_value
        self.history.append(new_value)

    def backward(self, gradient: TextGradient) -> None:
        """
        Store a gradient for later application by the optimizer.

        In PyTorch, backward() computes gradients via chain rule.
        Here (single-variable mode), we simply store the gradient
        that was computed externally by TextLoss.
        """
        if not self.requires_grad:
            logger.warning("backward() called on variable with requires_grad=False")
            return
        self.grad = gradient
        logger.debug("[TextGrad] Gradient stored for '%s' (severity=%.2f, "
                     "%d mutations)",
                     self.role_description[:30],
                     gradient.severity,
                     len(gradient.mutations))

    def __repr__(self) -> str:
        grad_info = " (has grad)" if self.grad else ""
        return "TextVariable(role='{}', len={}, requires_grad={}{})".format(
            self.role_description[:30], len(self.value),
            self.requires_grad, grad_info)


# ---------------------------------------------------------------------------
# Evaluation / Loss
# ---------------------------------------------------------------------------

# Prompt template for code evaluation
CODE_EVAL_PROMPT = """\
You are an expert code reviewer. Analyze this code and its execution output.
Identify specific errors and suggest concrete fixes.

Role: {role_description}

Code:
```python
{code}
```

Execution output (stdout):
{stdout}

Execution errors (stderr):
{stderr}

Original task this code was meant to solve:
{task}

Analyze the errors and suggest fixes. Use this EXACT format for each issue:
LINE: <line number where the issue is>
CAUSE: <what went wrong on that line>
FIX: <specific code change to fix it>

If there are multiple issues, list them one after another.
Be precise and concise. Do not suggest style changes, only functional fixes.
"""

# Generic evaluation prompt for non-code text
GENERIC_EVAL_PROMPT = """\
You are a critical evaluator. Analyze this text and identify problems.

Role: {role_description}

Text:
{text}

Context:
{context}

Identify specific issues and suggest fixes. Use this EXACT format:
LINE: <line or section>
CAUSE: <what is wrong>
FIX: <specific change>

Be precise and concise.
"""


class TextLoss:
    """
    Evaluates text quality using the LLM as a judge.

    Analogous to a PyTorch loss function (e.g., nn.CrossEntropyLoss).
    Instead of computing a numerical loss, it produces a textual gradient
    -- a structured critique of what's wrong and how to fix it.

    Usage:
        loss_fn = TextLoss(client, model, CODE_EVAL_PROMPT)
        gradient = loss_fn(code_variable, context={...})
        code_variable.backward(gradient)
    """

    def __init__(
        self,
        client: OpenAI,
        model: str,
        eval_prompt: str = CODE_EVAL_PROMPT,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ):
        """
        Args:
            client: OpenAI client pointed at llama.cpp.
            model: Model name string.
            eval_prompt: Template for the evaluation prompt.
            max_tokens: Max tokens for the critique response.
            temperature: Low temp for precise critique (0.1 recommended).
        """
        self.client = client
        self.model = model
        self.eval_prompt = eval_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature

    def __call__(
        self,
        variable: TextVariable,
        context: dict[str, Any] | None = None,
    ) -> TextGradient:
        """
        Evaluate the variable and return a textual gradient.

        Args:
            variable: The text to evaluate.
            context: Extra context dict. For code evaluation, should include:
                - task: original task description
                - stdout: sandbox stdout
                - stderr: sandbox stderr

        Returns:
            A TextGradient with structured mutations.
        """
        ctx = context or {}

        # Build the evaluation prompt
        if "code" in self.eval_prompt or "{code}" in self.eval_prompt:
            prompt = self.eval_prompt.format(
                role_description=variable.role_description,
                code=variable.value,
                stdout=ctx.get("stdout", "(none)"),
                stderr=ctx.get("stderr", "(none)"),
                task=ctx.get("task", "(not specified)"),
            )
        else:
            prompt = self.eval_prompt.format(
                role_description=variable.role_description,
                text=variable.value,
                context=str(ctx),
            )

        # Call LLM for critique
        logger.info("[TextGrad] Computing gradient for '%s'...",
                    variable.role_description[:30])

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a precise code reviewer. "
                     "Only output in the LINE/CAUSE/FIX format requested."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            critique = (response.choices[0].message.content or "").strip()
        except Exception as e:
            logger.error("[TextGrad] Evaluation LLM call failed: %s", e)
            critique = "EVALUATION FAILED: {}".format(e)

        # Parse critique into structured TextGradient
        gradient = self._parse_gradient(
            critique, variable, ctx.get("task", ""))

        logger.info("[TextGrad] Gradient computed: severity=%.2f, %d mutations",
                    gradient.severity, len(gradient.mutations))
        return gradient

    def _parse_gradient(
        self,
        critique: str,
        variable: TextVariable,
        task: str,
    ) -> TextGradient:
        """
        Parse the LLM's critique into a structured TextGradient.

        Looks for LINE/CAUSE/FIX patterns in the critique text.
        Falls back to treating the whole critique as a single mutation.
        """
        mutations = []

        # Parse LINE/CAUSE/FIX blocks
        line_pattern = re.compile(
            r"LINE:\s*(\d+)\s*\nCAUSE:\s*(.+?)\s*\nFIX:\s*(.+?)(?=\nLINE:|\Z)",
            re.DOTALL | re.IGNORECASE,
        )
        for match in line_pattern.finditer(critique):
            mutations.append(Mutation(
                line=int(match.group(1)),
                cause=match.group(2).strip(),
                suggestion=match.group(3).strip(),
            ))

        # Fallback: if no structured mutations found, create one from the whole critique
        if not mutations and critique:
            mutations.append(Mutation(
                line=0,
                cause="General issue detected",
                suggestion=critique[:500],
            ))

        # Compute severity based on number of mutations
        severity = min(1.0, len(mutations) * 0.25)
        if not mutations:
            severity = 0.0

        return TextGradient(
            node_id="textgrad",
            loss_description=critique[:300],
            mutations=mutations,
            target_variable="code_output",
            severity=severity,
        )


# ---------------------------------------------------------------------------
# TGD Optimizer
# ---------------------------------------------------------------------------

UPDATE_PROMPT = """\
You are improving the following {role_description}.

Current version:
```
{current_value}
```

Feedback (apply ALL of these changes):
{feedback}

Produce the IMPROVED version. Rules:
- Apply every fix listed above.
- Keep everything else exactly the same.
- Return ONLY the updated code/text, wrapped in ```python ... ``` if it's code.
- Do NOT add explanations, comments about the changes, or preamble.
"""


class TGD:
    """
    Textual Gradient Descent optimizer.

    Analogous to torch.optim.SGD. Instead of updating numerical weights
    via w = w - lr * gradient, it updates text via:
        new_text = LLM(current_text, gradient_feedback)

    The LLM acts as the "update rule" -- it reads the current text and
    the structured critique, then produces an improved version.

    Usage:
        optimizer = TGD(client, model, parameters=[code_var])
        optimizer.step()  # applies gradients to all parameters
    """

    def __init__(
        self,
        client: OpenAI,
        model: str,
        parameters: list[TextVariable],
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ):
        """
        Args:
            client: OpenAI client pointed at llama.cpp.
            model: Model name string.
            parameters: List of TextVariables to optimize.
            max_tokens: Max tokens for the update response.
            temperature: Low temp for faithful gradient application.
        """
        self.client = client
        self.model = model
        self.parameters = parameters
        self.max_tokens = max_tokens
        self.temperature = temperature

    def step(self) -> dict[str, str]:
        """
        Apply accumulated gradients to update all parameters.

        Returns:
            Dict mapping role_description -> new_value for each updated variable.
        """
        updates = {}
        for param in self.parameters:
            if param.grad is None:
                continue
            if not param.requires_grad:
                continue

            new_value = self._apply_gradient(param)
            if new_value and new_value != param.value:
                old_value = param.value
                param.set_value(new_value)
                updates[param.role_description] = new_value
                logger.info("[TextGrad] Updated '%s' (%d -> %d chars)",
                            param.role_description[:30],
                            len(old_value), len(new_value))
            else:
                logger.warning("[TextGrad] No update produced for '%s'",
                               param.role_description[:30])

            # Clear gradient after step
            param.grad = None

        return updates

    def zero_grad(self) -> None:
        """Clear all accumulated gradients."""
        for param in self.parameters:
            param.grad = None

    def _apply_gradient(self, variable: TextVariable) -> str:
        """
        Use LLM to produce updated text based on the gradient.

        This is the core "update rule" -- analogous to w = w - lr * grad
        in numerical optimization, but using the LLM as the update function.
        """
        if variable.grad is None:
            return variable.value

        # Format the gradient as structured feedback
        feedback = _format_gradient(variable.grad)

        prompt = UPDATE_PROMPT.format(
            role_description=variable.role_description,
            current_value=variable.value,
            feedback=feedback,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a code improvement agent. "
                     "Apply the feedback precisely and return only the updated code."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            result = (response.choices[0].message.content or "").strip()

            # Extract code block if present
            code = _extract_code(result)
            return code if code else result

        except Exception as e:
            logger.error("[TextGrad] Update LLM call failed: %s", e)
            return variable.value


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_gradient(gradient: TextGradient) -> str:
    """Format a TextGradient into human-readable feedback text."""
    if hasattr(gradient, 'to_formatted_string'):
        return gradient.to_formatted_string()

    lines = []
    for m in gradient.mutations:
        if m.line > 0:
            lines.append("[L{}] CAUSE: {} -> FIX: {}".format(
                m.line, m.cause, m.suggestion))
        else:
            lines.append("CAUSE: {} -> FIX: {}".format(m.cause, m.suggestion))
    return "\n".join(lines) if lines else gradient.loss_description


def _extract_code(text: str) -> str | None:
    """Extract code from a markdown code block."""
    pattern = r"```(?:python)?\s*\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


# ---------------------------------------------------------------------------
# High-level API: one-shot gradient fix
# ---------------------------------------------------------------------------

def gradient_fix(
    client: OpenAI,
    model: str,
    code: str,
    task: str,
    stdout: str,
    stderr: str,
    max_iterations: int = 1,
    eval_temperature: float = 0.1,
    update_temperature: float = 0.2,
    max_tokens: int = 2048,
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
) -> tuple[str, list[dict]]:
    """
    One-shot gradient fix: evaluate code, compute gradient, apply it.

    This is the main entry point for TextGrad integration.

    Args:
        client: OpenAI client.
        model: Model name.
        code: The code to fix.
        task: Original task description.
        stdout: Sandbox stdout from failed execution.
        stderr: Sandbox stderr from failed execution.
        max_iterations: How many gradient steps to perform.
        eval_temperature: Temperature for the evaluator.
        update_temperature: Temperature for the updater.
        max_tokens: Max tokens per LLM call.

    Returns:
        (fixed_code, gradient_log) where gradient_log is a list of
        dicts with gradient info for each iteration.
    """
    code_var = TextVariable(
        value=code,
        role_description="Python function",
        requires_grad=True,
    )

    loss_fn = TextLoss(
        client=client,
        model=model,
        eval_prompt=CODE_EVAL_PROMPT,
        max_tokens=max_tokens,
        temperature=eval_temperature,
    )

    optimizer = TGD(
        client=client,
        model=model,
        parameters=[code_var],
        max_tokens=max_tokens,
        temperature=update_temperature,
    )

    gradient_log = []

    def report(phase: str, **details: Any) -> None:
        if progress_callback is not None:
            try:
                progress_callback(phase, details)
            except Exception as exc:
                logger.warning("[TextGrad] Progress callback failed: %s", exc)

    for iteration in range(max_iterations):
        # Forward: compute loss (evaluate current code)
        report("evaluating_loss", iteration=iteration + 1)
        gradient = loss_fn(
            code_var,
            context={
                "task": task,
                "stdout": stdout,
                "stderr": stderr,
            },
        )

        # Backward: store gradient
        code_var.backward(gradient)

        # Log this iteration
        gradient_log.append({
            "iteration": iteration + 1,
            "severity": gradient.severity,
            "num_mutations": len(gradient.mutations),
            "mutations": [
                {"line": m.line, "cause": m.cause, "suggestion": m.suggestion}
                for m in gradient.mutations
            ],
        })
        report(
            "gradient_ready",
            iteration=iteration + 1,
            severity=gradient.severity,
            num_mutations=len(gradient.mutations),
            mutations=gradient_log[-1]["mutations"],
        )

        logger.info("[TextGrad] Iteration %d/%d: severity=%.2f, %d mutations",
                    iteration + 1, max_iterations,
                    gradient.severity, len(gradient.mutations))

        # Step: apply gradient
        report("applying_update", iteration=iteration + 1)
        optimizer.step()
        report("iteration_complete", iteration=iteration + 1)

        # If gradient was low severity, stop early
        if gradient.severity < 0.2:
            logger.info("[TextGrad] Low severity gradient (%.2f), stopping early.",
                        gradient.severity)
            break

    return code_var.value, gradient_log
