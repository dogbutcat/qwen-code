/**
 * @license
 * Copyright 2025 Qwen Team
 * SPDX-License-Identifier: Apache-2.0
 */

import type {
  Config,
  ServerGeminiStreamEvent,
  ToolCallRequestInfo,
  ToolCallResponseInfo,
} from '@qwen-code/qwen-code-core';
import { GeminiEventType, OutputFormat } from '@qwen-code/qwen-code-core';
import type { CLIAssistantMessage, CLIMessage } from '../types.js';
import {
  BaseJsonOutputAdapter,
  toolResultContent,
  type JsonOutputAdapterInterface,
  type ResultOptions,
} from './BaseJsonOutputAdapter.js';

/**
 * Maximum length of tool-call argument JSON rendered in the codex-style
 * stderr trace. Beyond this we truncate with an ellipsis to keep the
 * progress stream readable.
 */
const STDERR_TRACE_ARGS_LIMIT = 200;

/**
 * Maximum length of a tool result preview rendered in the codex-style
 * stderr trace. Beyond this we truncate with an ellipsis.
 */
const STDERR_TRACE_RESULT_LIMIT = 500;

/**
 * Common single-string envelope keys returned by tools. Matches MCP
 * (`{response: "..."}`), Qwen shell (`{output: "..."}`) and a few other
 * conventions. We unwrap recursively (max depth 4) so a doubly-encoded
 * payload like `{"response":"{\"output\":\"...\"}"}` collapses to the
 * inner string.
 */
const TOOL_RESULT_TEXT_KEYS = [
  'response',
  'output',
  'text',
  'content',
  'result',
  'message',
  'stdout',
];

/**
 * Recursively unwrap JSON envelopes around tool results so the codex-style
 * stderr trace shows the actual payload (matching `succeeded in Xms:\n<text>`)
 * instead of a giant escaped JSON blob.
 *
 * Strategy:
 * 1. If the input parses as a JSON object whose only meaningful field is one
 *    of {@link TOOL_RESULT_TEXT_KEYS}, recurse into that field.
 * 2. If it parses as a JSON string, recurse into the unescaped value.
 * 3. Otherwise return the input verbatim.
 */
export function unwrapToolResultText(raw: string, depth = 0): string {
  if (depth >= 4) return raw;
  const trimmed = raw.trim();
  if (trimmed.length === 0) return raw;
  // Cheap reject: only attempt JSON parse if it looks like JSON.
  const first = trimmed[0];
  if (first !== '{' && first !== '[' && first !== '"') return raw;
  let parsed: unknown;
  try {
    parsed = JSON.parse(trimmed);
  } catch {
    return raw;
  }
  if (typeof parsed === 'string') {
    return unwrapToolResultText(parsed, depth + 1);
  }
  if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
    const obj = parsed as Record<string, unknown>;
    const keys = Object.keys(obj);
    // Single-field envelope: unwrap directly regardless of key name when
    // value is a string (matches MCP results that wrap text in {response: …}).
    if (keys.length === 1 && typeof obj[keys[0]] === 'string') {
      return unwrapToolResultText(obj[keys[0]] as string, depth + 1);
    }
    // Multi-field object: pick the first conventional text field if present.
    for (const key of TOOL_RESULT_TEXT_KEYS) {
      if (typeof obj[key] === 'string') {
        return unwrapToolResultText(obj[key] as string, depth + 1);
      }
    }
  }
  return raw;
}

/**
 * JSON output adapter that collects all messages and emits them
 * as a single JSON array at the end of the turn.
 * Supports both main agent and subagent messages through distinct APIs.
 */
export class JsonOutputAdapter
  extends BaseJsonOutputAdapter
  implements JsonOutputAdapterInterface
{
  private readonly messages: CLIMessage[] = [];
  /**
   * Cumulative model prose captured from `Content` stream events. Used by
   * the codex-style TEXT mode so that stdout receives the complete answer
   * even when the final assistant turn only contains tool_use blocks
   * (in which case `extractTextFromBlocks(lastAssistantMessage)` returns "").
   */
  private accumulatedContentText = '';
  /**
   * Thought-token streaming buffer.
   * `GeminiEventType.Thought` fires once per streamed token, so we accumulate
   * `description` fragments and flush them as a single line only when the
   * thought stream ends (i.e. a non-Thought event arrives or the message is
   * finalised). This prevents the "one word per line" mess.
   */
  private thoughtDescBuf = '';
  private thoughtSubjBuf = '';
  /**
   * True while we are streaming Content tokens to stderr without a trailing
   * newline. Used to ensure the next non-Content write starts on a new line.
   */
  private stderrMidLine = false;

  constructor(config: Config) {
    super(config);
  }

  private isTextMode(): boolean {
    return this.config.getOutputFormat() === OutputFormat.TEXT;
  }

  /**
   * Flush the accumulated thought buffer as a single stderr line.
   * No-op when buffer is empty.
   */
  private flushThoughtBuf(): void {
    if (this.thoughtDescBuf.length === 0) return;
    const prefix = this.stderrMidLine ? '\n' : '';
    const sep = this.thoughtSubjBuf ? ': ' : '';
    process.stderr.write(
      `${prefix}[reasoning] ${this.thoughtSubjBuf}${sep}${this.thoughtDescBuf}\n`,
    );
    this.thoughtDescBuf = '';
    this.thoughtSubjBuf = '';
    this.stderrMidLine = false;
  }

  /**
   * Write a labelled stderr trace line, respecting mid-line state so that
   * a leading newline is inserted when content was being streamed inline.
   */
  private writeTrace(line: string): void {
    const prefix = this.stderrMidLine ? '\n' : '';
    process.stderr.write(`${prefix}${line}\n`);
    this.stderrMidLine = false;
  }

  /**
   * Codex-style TEXT mode: forward every meaningful turn event to stderr in
   * real time so users see progress, while keeping stdout reserved for the
   * final answer (emitted in {@link emitResult}). All other output formats
   * (JSON, STREAM_JSON) are unaffected — we just delegate to `super`.
   */
  override processEvent(event: ServerGeminiStreamEvent): void {
    super.processEvent(event);

    if (!this.isTextMode()) {
      return;
    }

    // Flush buffered thought whenever a non-Thought event arrives so that
    // the complete thought appears as one line before the next trace entry.
    if (event.type !== GeminiEventType.Thought) {
      this.flushThoughtBuf();
    }

    switch (event.type) {
      case GeminiEventType.Content: {
        if (typeof event.value === 'string' && event.value.length > 0) {
          process.stderr.write(event.value);
          this.accumulatedContentText += event.value;
          this.stderrMidLine = true;
        }
        break;
      }
      case GeminiEventType.Citation: {
        if (typeof event.value === 'string' && event.value.length > 0) {
          this.writeTrace(`[citation] ${event.value}`);
          // Mirror BaseJsonOutputAdapter.processEvent which folds citations
          // into the assistant text block — keep stdout consistent.
          this.accumulatedContentText += `\n${event.value}`;
        }
        break;
      }
      case GeminiEventType.Thought: {
        // Accumulate streaming thought tokens — do NOT write per-token.
        const subj = event.value?.subject ? String(event.value.subject) : '';
        const desc = event.value?.description
          ? String(event.value.description)
          : '';
        // If the subject changes mid-stream a new distinct thought has begun.
        if (
          subj &&
          subj !== this.thoughtSubjBuf &&
          this.thoughtDescBuf.length > 0
        ) {
          this.flushThoughtBuf();
        }
        if (subj) this.thoughtSubjBuf = subj;
        this.thoughtDescBuf += desc;
        break;
      }
      case GeminiEventType.ToolCallRequest: {
        const name = event.value?.name ?? 'unknown_tool';
        let argsStr: string;
        try {
          argsStr = JSON.stringify(event.value?.args ?? {});
        } catch {
          argsStr = '[unserializable args]';
        }
        if (argsStr.length > STDERR_TRACE_ARGS_LIMIT) {
          argsStr = `${argsStr.slice(0, STDERR_TRACE_ARGS_LIMIT)}…`;
        }
        this.writeTrace(`[tool] ${name} ${argsStr}`);
        break;
      }
      case GeminiEventType.Error: {
        const msg = event.value?.error?.message ?? 'unknown error';
        this.writeTrace(`[error] ${msg}`);
        break;
      }
      default:
        break;
    }
  }

  /**
   * Codex-style trace for tool execution results. JSON / STREAM_JSON paths
   * remain untouched — we only mirror to stderr in TEXT mode.
   */
  override emitToolResult(
    request: ToolCallRequestInfo,
    response: ToolCallResponseInfo,
    parentToolUseId: string | null = null,
  ): void {
    super.emitToolResult(request, response, parentToolUseId);

    if (!this.isTextMode()) {
      return;
    }

    // A tool result always ends the thought stream for the current turn.
    this.flushThoughtBuf();

    const raw = toolResultContent(response) ?? '';
    // Unwrap JSON envelopes (MCP tools and many shell tools return their
    // payload wrapped in {"response":"..."}, {"output":"..."}, {"text":"..."},
    // etc.). Codex prints the raw text after `succeeded in Xms:` — match that.
    const unwrapped = unwrapToolResultText(raw);
    let preview = unwrapped.replace(/\s+$/, '');
    const truncated = preview.length > STDERR_TRACE_RESULT_LIMIT;
    if (truncated) {
      preview = `${preview.slice(0, STDERR_TRACE_RESULT_LIMIT)}…`;
    }
    const status = response.error ? 'error' : 'ok';
    // Codex-style multi-line layout: header line, then payload on its own
    // line(s) so the user can scan it like a real terminal transcript instead
    // of reading a giant escaped JSON blob.
    if (preview.length === 0) {
      this.writeTrace(`[tool-result:${status}] ${request.name} (empty)`);
    } else if (preview.includes('\n')) {
      this.writeTrace(`[tool-result:${status}] ${request.name}:\n${preview}`);
    } else {
      this.writeTrace(`[tool-result:${status}] ${request.name}: ${preview}`);
    }
  }

  /**
   * Emits message to the messages array (batch mode).
   * Tracks the last assistant message for efficient result text extraction.
   */
  protected emitMessageImpl(message: CLIMessage): void {
    this.messages.push(message);
    // Track assistant messages for result generation
    if (
      typeof message === 'object' &&
      message !== null &&
      'type' in message &&
      message.type === 'assistant'
    ) {
      this.updateLastAssistantMessage(message as CLIAssistantMessage);
    }
  }

  /**
   * JSON mode does not emit stream events.
   */
  protected shouldEmitStreamEvents(): boolean {
    return false;
  }

  finalizeAssistantMessage(): CLIAssistantMessage {
    return this.finalizeAssistantMessageInternal(
      this.mainAgentMessageState,
      null,
    );
  }

  emitResult(options: ResultOptions): void {
    const resultMessage = this.buildResultMessage(
      options,
      this.lastAssistantMessage,
    );
    this.messages.push(resultMessage);

    if (this.config.getOutputFormat() === OutputFormat.TEXT) {
      // Flush any buffered thought that was still accumulating at turn end.
      this.flushThoughtBuf();
      // If Content was being streamed inline, move to a new line before the
      // final answer so stdout starts cleanly.
      if (this.stderrMidLine) {
        process.stderr.write('\n');
        this.stderrMidLine = false;
      }

      if (resultMessage.is_error) {
        process.stderr.write(`${resultMessage.error?.message || ''}\n`);
      } else {
        // Codex-style: stdout carries the *complete* model prose accumulated
        // from Content stream events. Falls back to the legacy
        // `extractTextFromBlocks(lastAssistantMessage)` payload only when no
        // streaming Content was captured (e.g. structured_result path or
        // synthetic summaries).
        const stdoutPayload =
          this.accumulatedContentText.length > 0
            ? this.accumulatedContentText
            : (resultMessage.result ?? '');
        process.stdout.write(
          stdoutPayload.endsWith('\n') ? stdoutPayload : `${stdoutPayload}\n`,
        );
      }
    } else {
      // Emit the entire messages array as JSON (includes all main agent + subagent messages)
      const json = JSON.stringify(this.messages);
      process.stdout.write(`${json}\n`);
    }
  }

  emitMessage(message: CLIMessage): void {
    // In JSON mode, messages are collected in the messages array
    // This is called by the base class's finalizeAssistantMessageInternal
    // but can also be called directly for user/tool/system messages
    this.messages.push(message);
  }
}
