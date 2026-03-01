# ========= Copyright 2025 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2025 @ CAMEL-AI.org. All Rights Reserved. =========

"""
Minimax M2.5 Interleaved Thinking Patch for CAMEL AI (CAMEL 0.2.82)

CRITICAL: This patch does NOT rely on DeepSeek patch infrastructure.

For CAMEL 0.2.82, Minimax returns reasoning in <think> tags (not reasoning_details).
This patch:
1. Extracts reasoning from <think> tags in responses
2. Stores reasoning in msg.reasoning_content for results.jsonl
3. At SERIALIZATION time, detects Minimax and injects actual content/reasoning
4. Converts Minimax context window errors to CAMEL-compatible format for auto-summarization
5. Retries summarization on timeout/failure to prevent context loss
6. Retries model API calls on timeout (not entire step)

Strategy - NO attribute transfer needed:
- Store all Minimax responses in a global registry (thread-safe)
- When serializing FunctionCallingMessage, check registry to see if this is Minimax
- If yes, inject actual content and reasoning (overwriting any template)

Context Window Error Handling:
- Minimax returns "context window exceeds limit" errors which CAMEL doesn't recognize
- CAMEL's auto-summarization looks for "context limit" or "context length" in error messages
- We convert Minimax errors to "context limit exceeded" to trigger CAMEL's auto-summarization

Summarization Retry Logic:
- When summarization times out (180s), CAMEL clears ALL context with empty summary
- This leaves the agent with no memory of what it was doing
- We retry summarization up to 10 times with linear backoff (5s, 10s, 15s, ..., 45s)
- Total max time: 10 attempts × 180s + (5+10+15+...+45)s = ~1800s + 225s = ~34 minutes
- This prevents catastrophic context loss from transient timeout errors

Model API Call Retry Logic:
- CAMEL's step_timeout (500s) wraps the ENTIRE step including tool execution
- If any tool takes too long, the whole step fails even if model API is fast
- We add a 180s timeout specifically for model API calls (not tools)
- We retry the model API call 10 times on timeout with linear backoff (10s, 20s, ..., 90s)
- Total max time: 10 attempts × 180s + (10+20+30+...+90)s = 1800s + 450s = ~37 minutes
- This allows tool execution to take as long as needed without failing the step

Usage:
======
    from eigent_search.minimax_m25_patch import apply_minimax_m25_patches
    apply_minimax_m25_patches()
"""

import re
import threading
from typing import Dict, Optional
from camel.logger import get_logger

logger = get_logger(__name__)

_patches_applied = False
_patch_lock = threading.Lock()

# Global registry of Minimax tool calls: tool_call_id -> (full_content, reasoning_content)
_minimax_tool_calls: Dict[str, tuple[str, str]] = {}
_registry_lock = threading.Lock()


def extract_think_tags(content: str) -> str:
    """Extract content from <think> tags."""
    if not content:
        return ""

    pattern = r'<think>(.*?)</think>'
    matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)

    if matches:
        return '\n'.join(match.strip() for match in matches)

    return ""


def register_minimax_tool_call(tool_call_id: str, full_content: str, reasoning: str):
    """Register a Minimax tool call response for later retrieval."""
    with _registry_lock:
        _minimax_tool_calls[tool_call_id] = (full_content, reasoning)


def get_minimax_content(tool_call_id: str) -> Optional[tuple[str, str]]:
    """Get Minimax content for a tool call ID, if it exists."""
    with _registry_lock:
        return _minimax_tool_calls.get(tool_call_id)


def apply_minimax_m25_patches(verbose: bool = False):
    """Apply Minimax M2.5 interleaved thinking patches for CAMEL 0.2.82."""
    global _patches_applied

    import os
    debug = os.getenv("DEBUG_MINIMAX_PATCH") == "true"

    with _patch_lock:
        if _patches_applied:
            if verbose or debug:
                logger.info("Minimax M2.5 patches already applied")
            return

        if verbose or debug:
            logger.info("=" * 80)
            logger.info("Applying Minimax M2.5 Patches...")
            logger.info("=" * 80)

        from camel.agents import chat_agent as chat_agent_module
        from camel.messages import func_message as func_message_module

        # ================================================================
        # PATCH #1: Extract reasoning and register tool calls
        # ================================================================
        original_handle_batch = chat_agent_module.ChatAgent._handle_batch_response

        def patched_handle_batch(self, response):
            """Extract <think> tags from Minimax responses and register them."""
            model_response = original_handle_batch(self, response)

            # Only process Minimax models
            if hasattr(response, 'model') and 'MiniMax' in str(response.model):
                if debug:
                    logger.info(f"🎯 Minimax response detected")

                # Check if response has tool_calls to get the tool_call_id
                if hasattr(response, 'choices') and response.choices:
                    for choice in response.choices:
                        if hasattr(choice, 'message') and hasattr(choice.message, 'tool_calls'):
                            if choice.message.tool_calls:
                                # Get the tool_call_id and content
                                for tool_call in choice.message.tool_calls:
                                    tool_call_id = tool_call.id
                                    content = choice.message.content or ""

                                    if content:
                                        reasoning = extract_think_tags(content)
                                        if reasoning:
                                            # Register this tool call with its content
                                            register_minimax_tool_call(tool_call_id, content, reasoning)

                                            if debug:
                                                logger.info(f"✅ Registered tool_call_id={tool_call_id}")
                                                logger.info(f"   Content: {len(content)} chars, Reasoning: {len(reasoning)} chars")

                # CRITICAL: Set reasoning_content on output_messages BEFORE they're returned
                # This ensures the messages in ChatAgentResponse.msgs have reasoning_content set
                if model_response.output_messages:
                    from camel.messages import FunctionCallingMessage

                    # Build a map of tool_call_id -> reasoning from our registry
                    # (We need to do this because output_messages might not be in the same order)
                    for msg in model_response.output_messages:
                        if isinstance(msg, FunctionCallingMessage) and hasattr(msg, 'tool_call_id'):
                            # Look up reasoning for this specific tool_call_id
                            minimax_data = get_minimax_content(msg.tool_call_id)
                            if minimax_data:
                                _, reasoning = minimax_data
                                msg.reasoning_content = str(reasoning)

                                if debug:
                                    logger.info(f"✅ Set reasoning_content on output_message")
                                    logger.info(f"   tool_call_id: {msg.tool_call_id}, reasoning: {len(reasoning)} chars")

            return model_response

        chat_agent_module.ChatAgent._handle_batch_response = patched_handle_batch

        if verbose or debug:
            logger.info("✅ Patch #1: Register Minimax tool calls")

        # ================================================================
        # PATCH #2: Add retry logic for Minimax choices=None errors
        #           AND convert Minimax context window errors to CAMEL-compatible format
        # ================================================================
        # Minimax API frequently returns choices=None which causes 'NoneType' is not iterable
        # We need to retry at the API call level (not orchestrator level) to preserve tool history
        #
        # IMPORTANT: Minimax returns "context window exceeds limit" errors which CAMEL doesn't
        # recognize. We need to convert these to a format that triggers CAMEL's auto-summarization.

        import asyncio
        import random
        from openai import BadRequestError

        original_aget_model_response = chat_agent_module.ChatAgent._aget_model_response

        async def patched_aget_model_response(self, openai_messages, num_tokens, current_iteration=0,
                                               response_format=None, tool_schemas=None, prev_num_openai_messages=0):
            """Retry on Minimax choices=None errors and handle context window errors."""
            max_retries = 10  # High retry count - we want to succeed!
            last_error = None

            for attempt in range(max_retries):
                try:
                    # Call original method
                    return await original_aget_model_response(
                        self, openai_messages, num_tokens, current_iteration,
                        response_format, tool_schemas, prev_num_openai_messages
                    )
                except BadRequestError as e:
                    # Check if this is a Minimax context window error
                    error_msg = str(e).lower()
                    if "context window exceeds limit" in error_msg:
                        # Convert to CAMEL-compatible error message for auto-summarization
                        # CAMEL looks for "context limit" or "context length" in the error message
                        if debug:
                            logger.warning(f"Minimax context window error detected, converting for CAMEL auto-summarization")

                        # Create a new exception with a message CAMEL will recognize
                        raise BadRequestError(
                            f"context limit exceeded: {str(e)}",
                            response=e.response,
                            body=e.body
                        ) from e
                    else:
                        # Not a context window error, re-raise as-is
                        raise
                except (TypeError, AttributeError) as e:
                    # Check if this is the Minimax choices=None error
                    if "'NoneType' object is not iterable" in str(e) or "choices" in str(e):
                        last_error = e
                        if attempt < max_retries - 1:
                            # Cap delay at 10 seconds to avoid excessive waiting
                            delay = min(2.0 * (attempt + 1), 10.0)  # 2s, 4s, 6s, 8s, 10s, 10s...
                            import sys
                            print(f"⚠️  Minimax API returned invalid response (attempt {attempt + 1}/{max_retries}), retrying in {delay}s...", file=sys.stderr)
                            if debug:
                                logger.warning(f"Minimax choices=None error, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                            await asyncio.sleep(delay)
                        else:
                            print(f"❌ Minimax API failed after {max_retries} attempts", file=sys.stderr)
                            raise
                    else:
                        # Not a Minimax error, re-raise immediately
                        raise

            # Should not reach here
            raise last_error

        chat_agent_module.ChatAgent._aget_model_response = patched_aget_model_response

        if verbose or debug:
            logger.info("✅ Patch #2: Add Minimax API retry logic + context window error conversion")

        # ================================================================
        # PATCH #3: Transfer reasoning_content to FunctionCallingMessage
        # ================================================================
        # When _record_tool_calling creates FunctionCallingMessage, it needs reasoning_content
        # for results.jsonl. We retrieve it from the registry.

        original_record_tool_calling = chat_agent_module.ChatAgent._record_tool_calling

        def patched_record_tool_calling(self, func_name, args, result, tool_call_id, mask_output=False, extra_content=None):
            """Transfer reasoning_content from registry to FunctionCallingMessage."""
            # Call original to create the record
            record = original_record_tool_calling(self, func_name, args, result, tool_call_id, mask_output, extra_content)

            # Check if this tool_call_id has Minimax reasoning in the registry
            minimax_data = get_minimax_content(tool_call_id)
            if minimax_data:
                full_content, reasoning = minimax_data

                # Find the FunctionCallingMessage that was just created and set reasoning_content
                try:
                    memory_records = self.memory.retrieve()
                    if memory_records:
                        from camel.messages import FunctionCallingMessage

                        for mem_record in reversed(memory_records):
                            msg = mem_record.memory_record.message
                            if isinstance(msg, FunctionCallingMessage) and hasattr(msg, 'tool_call_id'):
                                if msg.tool_call_id == tool_call_id and msg.func_name == func_name:
                                    # Set reasoning_content (always overwrite to ensure correctness)
                                    # Store as a new string object to prevent reference sharing
                                    msg.reasoning_content = str(reasoning)

                                    if debug:
                                        logger.info(f"✅ Set reasoning_content on FunctionCallingMessage")
                                        logger.info(f"   tool_call_id: {tool_call_id}, func_name: {func_name}")
                                        logger.info(f"   msg object id: {id(msg)}, reasoning: {len(reasoning)} chars")
                                        logger.info(f"   First 100 chars of reasoning: {reasoning[:100]}")
                                    break
                except Exception as e:
                    if debug:
                        logger.warning(f"⚠️  Failed to set reasoning_content: {e}")

            return record

        chat_agent_module.ChatAgent._record_tool_calling = patched_record_tool_calling

        if verbose or debug:
            logger.info("✅ Patch #3: Transfer reasoning_content to FunctionCallingMessage")

        # ================================================================
        # PATCH #4: Override FunctionCallingMessage serialization
        # ================================================================
        original_func_to_openai = func_message_module.FunctionCallingMessage.to_openai_assistant_message

        def patched_minimax_func_to_openai_assistant_message(self):
            """Check registry and inject Minimax content if available."""
            # Call the original (or DeepSeek-patched) version
            message_dict = original_func_to_openai(self)

            # Check if this tool_call_id is in the Minimax registry
            if hasattr(self, 'tool_call_id') and self.tool_call_id:
                minimax_data = get_minimax_content(self.tool_call_id)
                if minimax_data:
                    full_content, reasoning = minimax_data

                    # Inject the actual content with <think> tags
                    message_dict["content"] = full_content

                    # IMPORTANT: Remove reasoning_content field for Minimax
                    # Minimax only needs <think> tags in content, not a separate reasoning_content field
                    # The DeepSeek patch may have added this, but Minimax doesn't require it
                    if "reasoning_content" in message_dict:
                        del message_dict["reasoning_content"]

                    if debug:
                        logger.info(f"✅ Injected Minimax content for tool_call_id={self.tool_call_id}")
                        logger.info(f"   Content: {len(full_content)} chars (includes <think> tags)")
                        logger.info(f"   Removed reasoning_content field (not needed for Minimax)")

            return message_dict

        func_message_module.FunctionCallingMessage.to_openai_assistant_message = patched_minimax_func_to_openai_assistant_message

        if verbose or debug:
            logger.info("✅ Patch #4: Override serialization with registry lookup")

        # ================================================================
        # PATCH #5: Retry summarization on timeout/failure
        # ================================================================
        # When summarization times out (180s default), CAMEL logs error but continues
        # with empty summary, which CLEARS ALL CONTEXT and leaves agent with no memory.
        # We add retry logic to attempt summarization multiple times before giving up.

        original_asummarize = chat_agent_module.ChatAgent.asummarize

        async def patched_asummarize(self, include_summaries: bool = False):
            """Retry summarization on timeout or failure."""
            max_retries = 10  # Try 10 times total
            base_delay = 5.0  # Start with 5s, increase linearly

            for attempt in range(max_retries):
                try:
                    result = await original_asummarize(self, include_summaries)

                    # Check if summarization succeeded (has "summary" field with non-empty content)
                    if result.get("summary"):
                        if attempt > 0:
                            logger.info(f"✅ Summarization succeeded on attempt {attempt + 1}/{max_retries}")
                        return result
                    else:
                        # Empty summary or failure status
                        error_status = result.get("status", "Unknown error")

                        if attempt < max_retries - 1:
                            # Linear backoff: 5s, 10s, 15s, 20s, 25s, 30s, 35s, 40s, 45s
                            retry_delay = base_delay * (attempt + 1)
                            logger.warning(
                                f"⚠️  Summarization failed (attempt {attempt + 1}/{max_retries}): {error_status}"
                            )
                            logger.warning(f"   Retrying in {retry_delay}s...")
                            await asyncio.sleep(retry_delay)
                        else:
                            logger.error(
                                f"❌ Summarization failed after {max_retries} attempts: {error_status}"
                            )
                            logger.error(
                                "   Context will be cleared without proper summary - agent may lose memory!"
                            )
                            return result

                except Exception as e:
                    if attempt < max_retries - 1:
                        # Linear backoff: 5s, 10s, 15s, 20s, 25s, 30s, 35s, 40s, 45s
                        retry_delay = base_delay * (attempt + 1)
                        logger.warning(
                            f"⚠️  Summarization exception (attempt {attempt + 1}/{max_retries}): {e}"
                        )
                        logger.warning(f"   Retrying in {retry_delay}s...")
                        await asyncio.sleep(retry_delay)
                    else:
                        logger.error(
                            f"❌ Summarization exception after {max_retries} attempts: {e}"
                        )
                        # Re-raise to let CAMEL handle it
                        raise

            # Should not reach here
            return {"status": "Failed after all retries"}

        chat_agent_module.ChatAgent.asummarize = patched_asummarize

        if verbose or debug:
            logger.info("✅ Patch #5: Add retry logic for summarization failures")

        # ================================================================
        # PATCH #6: Retry model API call on timeout (not entire step)
        # ================================================================
        # Currently CAMEL wraps the entire step (including tool execution) with step_timeout
        # This means if ANY tool takes too long, the whole step fails
        # We want to ONLY retry the model API call on timeout, not fail the entire step
        #
        # Strategy:
        # 1. Wrap model_backend.arun with a 180s timeout
        # 2. Retry the API call up to 3 times on timeout
        # 3. Tool execution is NOT subject to this timeout

        from camel.models import BaseModelBackend

        # Store original arun method
        original_model_arun = BaseModelBackend.arun

        async def patched_model_arun_with_retry(self, messages, response_format=None, tools=None):
            """Retry model API call on timeout."""
            max_retries = 10  # Try 10 times total
            timeout_seconds = 180.0  # 180s timeout per API call attempt
            base_delay = 10.0  # Start with 10s, increase linearly

            for attempt in range(max_retries):
                try:
                    # Wrap the API call with timeout
                    result = await asyncio.wait_for(
                        original_model_arun(self, messages, response_format, tools),
                        timeout=timeout_seconds
                    )

                    if attempt > 0:
                        logger.info(f"✅ Model API call succeeded on attempt {attempt + 1}/{max_retries}")

                    return result

                except asyncio.TimeoutError:
                    if attempt < max_retries - 1:
                        # Linear backoff: 10s, 20s, 30s, 40s, 50s, 60s, 70s, 80s, 90s
                        retry_delay = base_delay * (attempt + 1)
                        logger.warning(
                            f"⚠️  Model API call timed out after {timeout_seconds}s "
                            f"(attempt {attempt + 1}/{max_retries})"
                        )
                        logger.warning(f"   Retrying in {retry_delay}s (attempt {attempt + 2}/{max_retries})...")
                        await asyncio.sleep(retry_delay)
                    else:
                        logger.error(
                            f"❌ Model API call timed out after {max_retries} attempts "
                            f"({timeout_seconds}s each)"
                        )
                        raise asyncio.TimeoutError(
                            f"Model API call timed out after {max_retries} attempts "
                            f"of {timeout_seconds}s each"
                        )

        BaseModelBackend.arun = patched_model_arun_with_retry

        if verbose or debug:
            logger.info("✅ Patch #6: Retry model API call on timeout (180s × 3 attempts)")

        _patches_applied = True

        if verbose or debug:
            logger.info("=" * 80)
            logger.info("✅ Minimax M2.5 patches applied successfully! (6 patches)")
            logger.info("   - Patch #1: Register Minimax tool calls with reasoning")
            logger.info("   - Patch #2: Retry on choices=None + convert context errors for auto-summarization")
            logger.info("   - Patch #3: Transfer reasoning_content to FunctionCallingMessage")
            logger.info("   - Patch #4: Inject content and remove reasoning_content field")
            logger.info("   - Patch #5: Retry summarization on timeout/failure (10 attempts, linear backoff 5s-45s)")
            logger.info("   - Patch #6: Retry model API call on timeout (10 attempts × 180s, linear backoff 10s-90s)")
            logger.info("=" * 80)


# Auto-apply on import
apply_minimax_m25_patches(verbose=False)
