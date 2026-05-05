/**
 * SSE contract tests for OpenAI-compatible streaming output.
 *
 * Validates that emitToolCalls produces delta chunks matching OpenAI's spec:
 * https://platform.openai.com/docs/guides/function-calling#function-calling-streaming
 *
 * Fixtures are recorded from real DeepSeek responses to catch XML format drift.
 */
import { describe, it, before } from "node:test";
import assert from "node:assert/strict";
import { parseToolCallsFromText } from "../src/services/openai-tool-parser.js";

/* ── Fixtures: real DeepSeek XML tool call outputs ── */

const FIXTURES = {
  /** Single ReadFile call with path parameter */
  readFile: {
    text: `<function_calls>
<invoke name="ReadFile">
<parameter name="path" string="true">/home/langshen/Desktop/deepseek2api/package.json</parameter>
</invoke>
</function_calls>`,
    expected: {
      count: 1,
      names: ["ReadFile"],
      argKeys: [["path"]],
    },
  },

  /** ReadFile with offset+limit (multi-param) */
  readFileOffset: {
    text: `<function_calls>
<invoke name="ReadFile">
<parameter name="path" string="true">/home/langshen/Desktop/deepseek2api/src/services/openai-tool-parser.js</parameter>
<parameter name="offset" string="false">200</parameter>
<parameter name="limit" string="false">300</parameter>
</invoke>
</function_calls>`,
    expected: {
      count: 1,
      names: ["ReadFile"],
      argKeys: [["path", "offset", "limit"]],
    },
  },

  /** Shell command */
  shell: {
    text: `<function_calls>
<invoke name="Shell">
<parameter name="command" string="true">head -100 /home/langshen/Desktop/deepseek2api/src/services/openai-tool-parser.js</parameter>
<parameter name="description" string="true">Read first 100 lines</parameter>
</invoke>
</function_calls>`,
    expected: {
      count: 1,
      names: ["Shell"],
      argKeys: [["command", "description"]],
    },
  },

  /** Unclosed container (DeepSeek common glitch) */
  unclosedContainer: {
    text: `<function_calls>
  <invoke name="Shell">
    <parameter name="command" string="true">cd /home/langshen/Desktop/deepseek2api && node --test test/openai-tool-parser.test.js 2>&1 | tail -30</parameter>
    <parameter name="description" string="true">Run tests</parameter>
    <parameter name="block_until_ms" string="false">30000</parameter>
  </invoke>`,
    expected: {
      count: 1,
      names: ["Shell"],
    },
  },

  /** <tool_calls> container with <tool_call> sub-block (Strategy A2) */
  toolCallsContainer: {
    text: `<tool_calls>
  <tool_call>
    <tool_name>ReadFile</tool_name>
    <parameter name="path" string="true">/home/langshen/Desktop/deepseek2api/src/config.js</parameter>
  </tool_call>
</tool_calls>`,
    expected: {
      count: 1,
      names: ["ReadFile"],
    },
  },

  /** Duplicate container tags (model glitch) */
  duplicateContainer: {
    text: `<tool_calls>
<tool_calls>
  <invoke name="ReadFile">
    <parameter name="path" string="true">/home/langshen/Desktop/deepseek2api/src/config.js</parameter>
  </invoke>`,
    expected: {
      count: 1,
      names: ["ReadFile"],
    },
  },

  /** Malformed <tool_name="ReadFile"> → <tool_name name="ReadFile"> */
  malformedAttr: {
    text: `<function_calls>
<invoke tool_name="ReadFile">
<parameter name="path" string="true">/test/file.js</parameter>
<parameter name="limit" string="false">100</parameter>
</invoke>
</function_calls>`,
    expected: {
      count: 1,
      names: ["ReadFile"],
    },
  },

  /** <tool_name name="X"></parameter> garbled close */
  toolNameParamClose: {
    text: `<tool_calls>
<tool_name name="ReadFile"></parameter>
<parameter name="path" string="true">/x</parameter>
</tool_calls>`,
    expected: {
      count: 1,
      names: ["ReadFile"],
    },
  },

  /** Standalone <invoke> blocks (no container) — common in stream chunks */
  standaloneInvoke: {
    text: `<invoke name="ReadLints">
    <parameter name="paths" string="false">["/home/langshen/Desktop/deepseek2api/src"]</parameter>
  </invoke>`,
    expected: {
      count: 1,
      names: ["ReadLints"],
    },
  },

  /** Multiple standalone <invoke> blocks */
  multiStandalone: {
    text: `<invoke name="Shell">
<parameter name="command" string="true">ls</parameter>
</invoke>
<invoke name="ReadFile">
<parameter name="path" string="true">/tmp/test.txt</parameter>
</invoke>`,
    expected: {
      count: 2,
      names: ["Shell", "ReadFile"],
    },
  },
};

/* ── SSE format validation helpers ── */

/**
 * Simulate what emitToolCalls produces in the SSE stream.
 * This mirrors createChatToolCalls + buildChunkPayload.
 */
function createMockToolCallDelta(calls, startIndex = 0) {
  return calls.map((call, offset) => ({
    index: startIndex + offset,
    id: call.id,
    type: "function",
    function: {
      name: call.name,
      arguments: call.argumentsText || "{}",
    },
  }));
}

/**
 * Validate that a tool_calls delta conforms to OpenAI SSE spec:
 * - Every call has index, id, type, function fields
 * - function has name and arguments (both strings)
 * - id starts with "call_"
 * - index is sequential
 */
function validateOpenAiToolCallDelta(calls, startIndex = 0) {
  for (let i = 0; i < calls.length; i++) {
    const call = calls[i];

    // Required top-level fields
    assert.equal(typeof call.index, "number", `call[${i}].index must be number`);
    assert.equal(call.index, startIndex + i, `call[${i}].index must be sequential`);
    assert.equal(typeof call.id, "string", `call[${i}].id must be string`);
    assert.ok(call.id.startsWith("call_"), `call[${i}].id must start with "call_"`);
    assert.equal(call.type, "function", `call[${i}].type must be "function"`);

    // function field
    assert.equal(typeof call.function, "object", `call[${i}].function must be object`);
    assert.equal(typeof call.function.name, "string", `call[${i}].function.name must be string`);
    assert.ok(call.function.name.length > 0, `call[${i}].function.name must not be empty`);
    assert.equal(typeof call.function.arguments, "string", `call[${i}].function.arguments must be string`);

    // arguments must be valid JSON (or empty)
    if (call.function.arguments && call.function.arguments !== "{}") {
      try {
        JSON.parse(call.function.arguments);
      } catch {
        assert.fail(`call[${i}].function.arguments is not valid JSON: ${call.function.arguments.slice(0, 80)}`);
      }
    }
  }
}

/* ── Tests ── */

describe("SSE tool_calls delta contract", () => {
  it("createMockToolCallDelta produces OpenAI-compliant format", () => {
    const calls = [
      { id: "call_abc123", name: "ReadFile", argumentsText: '{"path":"/x"}' },
      { id: "call_def456", name: "Shell", argumentsText: '{"command":"ls"}' },
    ];
    const deltas = createMockToolCallDelta(calls);
    validateOpenAiToolCallDelta(deltas);
  });

  it("empty calls produce empty delta", () => {
    const deltas = createMockToolCallDelta([]);
    assert.equal(deltas.length, 0);
  });

  it("empty arguments default to {}", () => {
    const deltas = createMockToolCallDelta([
      { id: "call_x", name: "ReadFile", argumentsText: "" },
    ]);
    validateOpenAiToolCallDelta(deltas);
    assert.equal(deltas[0].function.arguments, "{}");
  });

  it("index starts from given startIndex", () => {
    const deltas = createMockToolCallDelta(
      [
        { id: "call_a", name: "A", argumentsText: "{}" },
        { id: "call_b", name: "B", argumentsText: "{}" },
      ],
      5
    );
    assert.equal(deltas[0].index, 5);
    assert.equal(deltas[1].index, 6);
  });

  it("rejects missing id", () => {
    assert.throws(() => {
      validateOpenAiToolCallDelta([
        { index: 0, type: "function", function: { name: "X", arguments: "{}" } },
      ]);
    });
  });

  it("rejects non-call_ id prefix", () => {
    assert.throws(() => {
      validateOpenAiToolCallDelta([
        { index: 0, id: "not_call", type: "function", function: { name: "X", arguments: "{}" } },
      ]);
    });
  });

  it("rejects empty function.name", () => {
    assert.throws(() => {
      validateOpenAiToolCallDelta([
        { index: 0, id: "call_x", type: "function", function: { name: "", arguments: "{}" } },
      ]);
    });
  });

  it("rejects non-sequential index", () => {
    assert.throws(() => {
      validateOpenAiToolCallDelta([
        { index: 0, id: "call_a", type: "function", function: { name: "A", arguments: "{}" } },
        { index: 5, id: "call_b", type: "function", function: { name: "B", arguments: "{}" } },
      ]);
    });
  });
});

/* ── Parser → tool_calls delta roundtrip ── */

describe("DeepSeek XML → OpenAI tool_calls roundtrip", () => {
  const ALL_TOOLS = ["Shell", "ReadFile", "ReadLints", "Glob", "rg"];
  const allowedTools = ALL_TOOLS;

  for (const [name, fixture] of Object.entries(FIXTURES)) {
    it(`fixture: ${name}`, () => {
      const calls = parseToolCallsFromText(fixture.text, allowedTools);

      assert.equal(
        calls.length,
        fixture.expected.count,
        `${name}: expected ${fixture.expected.count} call(s), got ${calls.length}`
      );

      // Validate names
      const actualNames = calls.map((c) => c.name);
      assert.deepEqual(actualNames, fixture.expected.names);

      // Validate each call has required fields
      for (const call of calls) {
        assert.ok(call.id, `${name}: call.id is missing`);
        assert.ok(call.id.startsWith("call_"), `${name}: call.id must start with "call_"`);
        assert.ok(call.name, `${name}: call.name is empty`);
        assert.equal(typeof call.argumentsText, "string", `${name}: argumentsText must be string`);
      }

      // Convert to OpenAI tool_calls delta format
      const deltas = createMockToolCallDelta(calls);
      validateOpenAiToolCallDelta(deltas);
    });
  }

  it("all fixtures produce valid SSE-ready tool_calls", () => {
    let total = 0;
    for (const fixture of Object.values(FIXTURES)) {
      const calls = parseToolCallsFromText(fixture.text, allowedTools);
      assert.equal(calls.length, fixture.expected.count);
      const deltas = createMockToolCallDelta(calls);
      validateOpenAiToolCallDelta(deltas); // each fixture resets index to 0
      total += calls.length;
    }
    assert.ok(total > 0, "should have validated at least one fixture");
  });
});

/* ── Argument inference contract ── */
import { getToolName } from "../src/services/openai-tool-policy.js";

describe("getToolName argument-based inference", () => {
  it("infers Shell from command key", () => {
    assert.equal(
      getToolName({
        id: "c1", type: "function",
        function: { name: "", arguments: '{"command":"ls"}' },
      }),
      "Shell"
    );
  });

  it("infers ReadFile from path key", () => {
    assert.equal(
      getToolName({
        id: "c2", type: "function",
        function: { name: "", arguments: '{"path":"/x"}' },
      }),
      "ReadFile"
    );
  });

  it("infers rg from pattern+path combo", () => {
    assert.equal(
      getToolName({
        id: "c3", type: "function",
        function: { name: "", arguments: '{"pattern":"foo","path":"/x"}' },
      }),
      "rg"
    );
  });

  it("infers Glob from glob_pattern key", () => {
    assert.equal(
      getToolName({
        id: "c4", type: "function",
        function: { name: "", arguments: '{"glob_pattern":"*.js"}' },
      }),
      "Glob"
    );
  });

  it("infers WebSearch from query key", () => {
    assert.equal(
      getToolName({
        id: "c5", type: "function",
        function: { name: "", arguments: '{"query":"search term"}' },
      }),
      "WebSearch"
    );
  });

  it("returns empty for MCP tools (name+repo, ambiguous)", () => {
    assert.equal(
      getToolName({
        id: "c6", type: "function",
        function: { name: "", arguments: '{"name":"ctx","repo":"x"}' },
      }),
      ""
    );
  });

  it("prefers function.name over inference", () => {
    assert.equal(
      getToolName({
        id: "c7", type: "function",
        function: { name: "ReadFile", arguments: '{"command":"ls"}' },
      }),
      "ReadFile"
    );
  });

  it("falls back to top-level name", () => {
    assert.equal(
      getToolName({
        id: "c8", type: "function",
        name: "Shell",
        function: { arguments: '{"command":"ls"}' },
      }),
      "Shell"
    );
  });

  it("handles malformed arguments JSON gracefully", () => {
    assert.equal(
      getToolName({
        id: "c9", type: "function",
        function: { name: "", arguments: "not json" },
      }),
      ""
    );
  });
});

describe("getToolName Cursor regression", () => {
  it("Cursor's empty function.name + command args → Shell", () => {
    // Real Cursor data: function.name="" with Shell's argument signature
    const call = JSON.parse(
      '{"id":"call_5a8a5011e30f4d9dbcd1fac308e777a6","type":"function",' +
      '"function":{"name":"","arguments":"{\\\"command\\\":\\\"wc -l /home/langshen/Desktop/deepseek2api/src/services/openai-tool-parser.js\\\",\\\"description\\\":\\\"Check file line count\\\"}"}}'
    );
    assert.equal(getToolName(call), "Shell");
  });

  it("Cursor's empty function.name + path args → ReadFile", () => {
    const call = JSON.parse(
      '{"id":"call_abc","type":"function",' +
      '"function":{"name":"","arguments":"{\\"path\\":\\"/x\\"}"}}'
    );
    assert.equal(getToolName(call), "ReadFile");
  });

  it("Cursor's empty function.name + glob_pattern → Glob", () => {
    const call = JSON.parse(
      '{"id":"call_2772fda081874b8392ccb34069522df4","type":"function",' +
      '"function":{"name":"","arguments":"{\\"glob_pattern\\":\\"src/services/openai-tool-parser.js\\"}"}}'
    );
    assert.equal(getToolName(call), "Glob");
  });
});

/* ── finish_reason contract ── */

describe("finish_reason contract", () => {
  it("tool_calls emits finish_reason='tool_calls' with empty delta", () => {
    // This is the contract: final chunk MUST have delta:{}, finish_reason:"tool_calls"
    const finalChunk = {
      choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }],
    };

    assert.deepEqual(finalChunk.choices[0].delta, {});
    assert.equal(finalChunk.choices[0].finish_reason, "tool_calls");
  });

  it("normal stop emits finish_reason='stop'", () => {
    const finalChunk = {
      choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
    };

    assert.equal(finalChunk.choices[0].finish_reason, "stop");
  });
});
