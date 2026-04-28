/**
 * Comprehensive test suite for openai-tool-parser.js
 * Covers all known DeepSeek model output formats including garbled/malformed XML.
 */
import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { parseToolCallsFromText } from "../src/services/openai-tool-parser.js";
import { createToolSieve } from "../src/services/openai-tool-sieve.js";

const ALL_TOOLS = [
  "Shell", "Glob", "rg", "Await", "ReadFile", "Delete", "EditNotebook",
  "TodoWrite", "ReadLints", "WebSearch", "WebFetch", "GenerateImage",
  "AskQuestion", "Subagent", "ListMcpResources", "FetchMcpResource",
  "SwitchMode", "ApplyPatch"
];

function parseNames(text, tools = ALL_TOOLS) {
  return parseToolCallsFromText(text, tools).map(c => c.name);
}

function parseOne(text, tools = ALL_TOOLS) {
  const calls = parseToolCallsFromText(text, tools);
  assert.equal(calls.length, 1, `Expected 1 call, got ${calls.length}: ${JSON.stringify(calls.map(c => c.name))}`);
  return calls[0];
}

function parseArgs(text, tools = ALL_TOOLS) {
  return parseOne(text, tools).input;
}

/* ═══════════════════════════════════════════════════
   Strategy A: <tool_call name="...">...</tool_call_name>
   ═══════════════════════════════════════════════════ */

describe("Strategy A: <tool_call name='...'> format", () => {
  it("parses standard <tool_call name='Shell'> with <parameter> sub-tags", () => {
    const text = `<tool_call name="Shell">
    <parameter name="command" string="true">ls -la</parameter>
</tool_call_name>`;
    const call = parseOne(text);
    assert.equal(call.name, "Shell");
    assert.equal(call.input.command, "ls -la");
  });

  it("parses multiple <tool_call name='ReadFile'> calls", () => {
    const text = `<tool_call name="ReadFile">
    <parameter name="path" string="true">./douyin.py</parameter>
</tool_call_name>
<tool_call name="ReadFile">
    <parameter name="path" string="true">./main.py</parameter>
</tool_call_name>`;
    const calls = parseToolCallsFromText(text, ALL_TOOLS);
    assert.equal(calls.length, 2);
    assert.equal(calls[0].name, "ReadFile");
    assert.equal(calls[0].input.path, "./douyin.py");
    assert.equal(calls[1].name, "ReadFile");
    assert.equal(calls[1].input.path, "./main.py");
  });
});

/* ═══════════════════════════════════════════════════
   Strategy B: <tool_name>+<parameters> inside <tool_calls>
   ═══════════════════════════════════════════════════ */

describe("Strategy B: <tool_name>+<parameters> inside container", () => {
  it("parses standard <tool_name>Shell</tool_name> + <parameters>", () => {
    const text = `<tool_calls>
   
    <tool_name>Shell</tool_name>
    <parameters><command>find . -maxdepth 2 | head -80</command>
<description>Explore project structure</description></parameters>
   
</tool_calls>`;
    const call = parseOne(text);
    assert.equal(call.name, "Shell");
    assert.equal(call.input.command, "find . -maxdepth 2 | head -80");
  });

  it("parses <tool_name>Glob</tool_name> + <parameters> with CDATA", () => {
    const text = `<tool_calls>
   
    <tool_name>Glob</tool_name>
    <parameters><![CDATA[{"target_directory":"/", "glob_pattern":"*"}]]></parameters>
   
</tool_calls>`;
    const call = parseOne(text);
    assert.equal(call.name, "Glob");
    assert.equal(call.input.target_directory, "/");
    assert.equal(call.input.glob_pattern, "*");
  });

  it("parses multiple tool calls in one <tool_calls> container", () => {
    const text = `<tool_calls>
   
    <tool_name>Glob</tool_name>
    <parameters><![CDATA[{"pattern":"**/*.py","target_directory":"."}]]></parameters>
   
    <tool_name>Glob</tool_name>
    <parameters><![CDATA[{"pattern":"**/*.js","target_directory":"."}]]></parameters>
   
</tool_calls>`;
    const calls = parseToolCallsFromText(text, ALL_TOOLS);
    assert.equal(calls.length, 2);
    assert.equal(calls[0].name, "Glob");
    assert.equal(calls[0].input.pattern, "**/*.py");
    assert.equal(calls[1].name, "Glob");
    assert.equal(calls[1].input.pattern, "**/*.js");
  });

  it("parses <tool_name>ReadFile</tool_name> + <parameters> with sub-element params", () => {
    const text = `<tool_calls>
<tool_name>Glob</tool_name>
<parameters><target_directory>.</target_directory>
<glob_pattern>"*.py"</glob_pattern></parameters>
</tool_calls>`;
    const call = parseOne(text);
    assert.equal(call.name, "Glob");
  });
});

/* ═══════════════════════════════════════════════════
   Malformed tag handling
   ═══════════════════════════════════════════════════ */

describe("Malformed tag preprocessing", () => {
  it("fixes <tool_name='ReadFile'> → <tool_name name='ReadFile'>", () => {
    const text = `<tool_calls>
   
    <tool_name="ReadFile"></tool_name>
    <parameters>
      <parameter name="path">./douyin.py</parameter>
    </parameters>
   
</tool_calls>`;
    const call = parseOne(text);
    assert.equal(call.name, "ReadFile");
    assert.equal(call.input.path, "./douyin.py");
  });

  it("handles <tool_name name='ReadFile'></tool_name> attribute format", () => {
    const text = `<tool_calls>
   
    <tool_name name="ReadFile"></tool_name>
    <parameters>
      <parameter name="path">./main.py</parameter>
    </parameters>
   
</tool_calls>`;
    const call = parseOne(text);
    assert.equal(call.name, "ReadFile");
    assert.equal(call.input.path, "./main.py");
  });

  it("handles <tool_name name='WebSearch</tool_name> (closing tag inside attribute)", () => {
    const text = `<tool_calls>
<tool_name name="WebSearch</tool_name>
<parameters>{"search_term":"test"}</parameters>
</tool_calls>`;
    const call = parseOne(text);
    assert.equal(call.name, "WebSearch");
  });
});

/* ═══════════════════════════════════════════════════
   Garbled/alternative tag names
   ═══════════════════════════════════════════════════ */

describe("Garbled and alternative tag names", () => {
  it("handles <tool_call_name>Shell</tool_call_name> (alternative name tag)", () => {
    const text = `<tool_calls>
<tool_call_name>Shell</tool_call_name>
<tool_call_parameters>{"command":"ls"}</tool_call_parameters>
</tool_calls>`;
    const call = parseOne(text);
    assert.equal(call.name, "Shell");
    assert.equal(call.input.command, "ls");
  });

  it("handles garbled opening tag <tool_cool_name>ReadFile</tool_name>", () => {
    const text = `<tool_calls>
<tool_cool_name>ReadFile</tool_name>
<parameters>{"path":"./test.py"}</parameters>
</tool_calls>`;
    const call = parseOne(text);
    assert.equal(call.name, "ReadFile");
    assert.equal(call.input.path, "./test.py");
  });
});

/* ═══════════════════════════════════════════════════
   Mismatched closing tags
   ═══════════════════════════════════════════════════ */

describe("Mismatched wrapper closing tags", () => {
  it("fixes <tool_call name='Shell'>...</tool_calls> mismatch", () => {
    const text = `<tool_call name="Shell">
<parameter name="command">ls -la</parameter>
</tool_calls>`;
    const call = parseOne(text);
    assert.equal(call.name, "Shell");
    assert.equal(call.input.command, "ls -la");
  });

  it("fixes multiple </tool_calls> → </tool_call_name> for nested calls", () => {
    const text = `<tool_call name="ReadFile">
<parameter name="path">./a.py</parameter>
</tool_calls>
<tool_call name="ReadFile">
<parameter name="path">./b.py</parameter>
</tool_calls>`;
    const calls = parseToolCallsFromText(text, ALL_TOOLS);
    assert.equal(calls.length, 2);
    assert.equal(calls[0].input.path, "./a.py");
    assert.equal(calls[1].input.path, "./b.py");
  });
});

/* ═══════════════════════════════════════════════════
   Unclosed tags
   ═══════════════════════════════════════════════════ */

describe("Unclosed tags", () => {
  it("handles missing </parameters> closing tag", () => {
    const text = `<tool_calls>
<tool_name>Shell</tool_name>
<parameters><command>ls</command>`;
    const call = parseOne(text);
    assert.equal(call.name, "Shell");
    assert.equal(call.input.command, "ls");
  });

  it("handles </parameter> instead of </parameters> (singular/plural mismatch)", () => {
    const text = `<tool_calls>
<tool_name>Shell</tool_name>
<parameters><command>ls</command></parameter>
</tool_calls>`;
    const call = parseOne(text);
    assert.equal(call.name, "Shell");
    assert.equal(call.input.command, "ls");
  });
});

/* ═══════════════════════════════════════════════════
   CDATA handling
   ═══════════════════════════════════════════════════ */

describe("CDATA handling", () => {
  it("parses CDATA-wrapped JSON in <parameters>", () => {
    const text = `<tool_calls>
<tool_name>WebSearch</tool_name>
<parameters><![CDATA[{"search_term":"deepseek"}]]></parameters>
</tool_calls>`;
    const call = parseOne(text);
    assert.equal(call.name, "WebSearch");
    assert.equal(call.input.search_term, "deepseek");
  });

  it("strips garbage after CDATA closing (e.g. ]]>}}}</parameter>)", () => {
    const text = `<tool_calls>
<tool_name>Shell</tool_name>
<parameters><![CDATA[{"command":"ls"}]]></parameter>}}}
</tool_calls>`;
    const call = parseOne(text);
    assert.equal(call.name, "Shell");
    assert.equal(call.input.command, "ls");
  });

  it("handles literal control chars in CDATA JSON (ApplyPatch patch param)", () => {
    const patchContent = "*** Begin Patch\n*** Modify File\n- old line\n+ new line";
    const text = `<tool_calls>
<tool_name>ApplyPatch</tool_name>
<parameters><![CDATA[{"patch":"${patchContent.replace(/\n/g, "\\n")}"}]]></parameters>
</tool_calls>`;
    const call = parseOne(text);
    assert.equal(call.name, "ApplyPatch");
    assert.ok(call.input.patch);
  });
});

/* ═══════════════════════════════════════════════════
   Strategy C: Tag-name-as-tool-name (e.g. <Glob>)
   ═══════════════════════════════════════════════════ */

describe("Strategy C: Tag-name-as-tool-name", () => {
  it("parses <Shell><parameters>...</parameters></Shell>", () => {
    const text = `<tool_calls>
<Shell>
<parameters>{"command":"ls -la"}</parameters>
</Shell>
</tool_calls>`;
    const call = parseOne(text);
    assert.equal(call.name, "Shell");
    assert.equal(call.input.command, "ls -la");
  });
});

/* ═══════════════════════════════════════════════════
   Strategy F: Plain-text Key:Value format
   ═══════════════════════════════════════════════════ */

describe("Strategy F: Plain-text Key:Value format", () => {
  it("parses 'Tool: Shell\\ncommand: ls'", () => {
    const text = `<tool_calls>
Tool: Shell
command: ls -la
</tool_calls>`;
    const call = parseOne(text);
    assert.equal(call.name, "Shell");
    assert.equal(call.input.command, "ls -la");
  });

  it("parses bare tool name 'Shell'", () => {
    const text = `<tool_calls>
Shell
</tool_calls>`;
    const call = parseOne(text);
    assert.equal(call.name, "Shell");
  });
});

/* ═══════════════════════════════════════════════════
   Strategy H: Infer tool name from parameter names
   ═══════════════════════════════════════════════════ */

describe("Strategy H: Infer tool name from params", () => {
  it("infers Shell from 'command' parameter", () => {
    const text = `<tool_calls>
<parameter name="command">ls -la</parameter>
</tool_calls>`;
    const call = parseOne(text);
    assert.equal(call.name, "Shell");
    assert.equal(call.input.command, "ls -la");
  });

  it("infers Glob from 'pattern' parameter", () => {
    const text = `<tool_calls>
<parameter name="pattern">**/*.py</parameter>
</tool_calls>`;
    const call = parseOne(text);
    assert.equal(call.name, "Glob");
  });

  it("infers ApplyPatch from 'patch' parameter", () => {
    const text = `<tool_calls>
<parameter name="patch">*** Begin Patch</parameter>
</tool_calls>`;
    const call = parseOne(text);
    assert.equal(call.name, "ApplyPatch");
  });
});

/* ═══════════════════════════════════════════════════
   Duplicate container tags
   ═══════════════════════════════════════════════════ */

describe("Duplicate container tags", () => {
  it("strips repeated <tool_calls> opening tags", () => {
    const text = `<tool_calls>
<tool_calls>
<tool_calls>
<tool_name>Shell</tool_name>
<parameters>{"command":"ls"}</parameters>
</tool_calls>`;
    const call = parseOne(text);
    assert.equal(call.name, "Shell");
    assert.equal(call.input.command, "ls");
  });
});

/* ═══════════════════════════════════════════════════
   Edge cases
   ═══════════════════════════════════════════════════ */

describe("Edge cases", () => {
  it("returns empty array for text without tool XML", () => {
    const calls = parseToolCallsFromText("Just a normal text response", ALL_TOOLS);
    assert.equal(calls.length, 0);
  });

  it("returns empty array for empty input", () => {
    const calls = parseToolCallsFromText("", ALL_TOOLS);
    assert.equal(calls.length, 0);
  });

  it("filters out tool calls not in allowed list", () => {
    const text = `<tool_calls>
<tool_name>Shell</tool_name>
<parameters>{"command":"ls"}</parameters>
</tool_calls>`;
    // Only allow Glob, not Shell
    const calls = parseToolCallsFromText(text, ["Glob"]);
    assert.equal(calls.length, 0);
  });

  it("handles <tool name='Shell'> format", () => {
    const text = `<tool name="Shell">
<parameter name="command">ls</parameter>
</tool>`;
    const call = parseOne(text);
    assert.equal(call.name, "Shell");
    assert.equal(call.input.command, "ls");
  });

  it("handles <invoke name='Shell'> format", () => {
    const text = `<invoke name="Shell">
<parameter name="command">ls</parameter>
</invoke>`;
    const call = parseOne(text);
    assert.equal(call.name, "Shell");
    assert.equal(call.input.command, "ls");
  });

  it("handles null bytes in input", () => {
    const text = `<tool_calls>\x00<tool_name>Shell</tool_name>
<parameters>{"command":"ls"}</parameters>
</tool_calls>`;
    const call = parseOne(text);
    assert.equal(call.name, "Shell");
    assert.equal(call.input.command, "ls");
  });

  it("handles tool calls in markdown code fences (ignores them)", () => {
    const text = "Here's the code:\n```\n<tool_name>Shell</tool_name>\n```\n<tool_calls>\n<tool_name>Glob</tool_name>\n<parameters>{\"pattern\":\"*.py\"}</parameters>\n</tool_calls>";
    const calls = parseToolCallsFromText(text, ALL_TOOLS);
    assert.equal(calls.length, 1);
    assert.equal(calls[0].name, "Glob");
  });
});

/* ═══════════════════════════════════════════════════
   Real-world debug log test cases
   ═══════════════════════════════════════════════════ */

describe("Real-world debug log test cases", () => {
  it("parses Shell with sub-element params from debug log", () => {
    const text = `<tool_calls>
   
    <tool_name>Shell</tool_name>
    <parameters><command>find . -maxdepth 2 -not -path './.git/*' -not -path './node_modules/*' | head -80</command>
<description>Explore project structure</description></parameters>
   
</tool_calls>`;
    const call = parseOne(text);
    assert.equal(call.name, "Shell");
    assert.equal(call.input.command, "find . -maxdepth 2 -not -path './.git/*' -not -path './node_modules/*' | head -80");
  });

  it("parses 6 ReadFile calls with <tool_call name='ReadFile'> format", () => {
    const text = `<tool_calls>
  <tool_call name="ReadFile">
    <parameter name="path" string="true">./douyin.py</parameter>
  </tool_call_name>
  <tool_call name="ReadFile">
    <parameter name="path" string="true">./main.py</parameter>
  </tool_call_name>
  <tool_call name="ReadFile">
    <parameter name="path" string="true">./jiazai.js</parameter>
  </tool_call_name>
  <tool_call name="ReadFile">
    <parameter name="path" string="true">./env.js</parameter>
  </tool_call_name>
  <tool_call name="ReadFile">
    <parameter name="path" string="true">./pyproject.toml</parameter>
  </tool_call_name>
  <tool_call name="ReadFile">
    <parameter name="path" string="true">./README.md</parameter>
  </tool_call_name>
</tool_calls>`;
    const calls = parseToolCallsFromText(text, ALL_TOOLS);
    assert.equal(calls.length, 6);
    assert.deepEqual(calls.map(c => c.name), ["ReadFile", "ReadFile", "ReadFile", "ReadFile", "ReadFile", "ReadFile"]);
    assert.equal(calls[0].input.path, "./douyin.py");
    assert.equal(calls[5].input.path, "./README.md");
  });

  it("parses <tool_name='ReadFile'> malformed from debug log", () => {
    const text = `<tool_calls>
   
    <tool_name="ReadFile"></tool_name>
    <parameters>
      <parameter name="path">./douyin.py</parameter>
    </parameters>
   
    <tool_name="ReadFile"></tool_name>
    <parameters>
      <parameter name="path">./main.py</parameter>
    </parameters>
   
    <tool_name="ReadFile"></tool_name>
    <parameters>
      <parameter name="path">./jiazai.js</parameter>
    </parameters>
  <tool_name="ReadFile"></tool_name>
    <parameters>
      <parameter name="path">./env.js</parameter>
    </parameters>
    <tool_name="ReadFile"></tool_name>
    <parameters>
      <parameter name="path">./pyproject.toml</parameter>
    </parameters>
    <tool_name="ReadFile"></tool_name>
    <parameters>
      <parameter name="path">./README.md</parameter>
    </parameters>
</tool_calls>`;
    const calls = parseToolCallsFromText(text, ALL_TOOLS);
    assert.equal(calls.length, 6);
    assert.equal(calls[0].input.path, "./douyin.py");
    assert.equal(calls[5].input.path, "./README.md");
  });

  it("parses 5 ReadFile calls with CDATA from debug log", () => {
    const text = `<tool_calls>
   
    <tool_name>ReadFile</tool_name>
    <parameters><![CDATA[{"path":"./douyin.py"}]]></parameters>
   
    <tool_name>ReadFile</tool_name>
    <parameters><![CDATA[{"path":"./main.py"}]]></parameters>
   
    <tool_name>ReadFile</tool_name>
    <parameters><![CDATA[{"path":"./jiazai.js"}]]></parameters>
   
    <tool_name>ReadFile</tool_name>
    <parameters><![CDATA[{"path":"./README.md"}]]></parameters>
   
    <tool_name>ReadFile</tool_name>
    <parameters><![CDATA[{"path":"./env.js"}]]></parameters>
   
</tool_calls>`;
    const calls = parseToolCallsFromText(text, ALL_TOOLS);
    assert.equal(calls.length, 5);
    assert.equal(calls[0].input.path, "./douyin.py");
    assert.equal(calls[4].input.path, "./env.js");
  });
});

/* ═══════════════════════════════════════════════════
   Tool-name-as-tag: standalone <ApplyPatch>...</ApplyPatch>
   (The root cause of the "ApplyPatch not working" bug)
   ═══════════════════════════════════════════════════ */

describe("Tool-name-as-tag: standalone <ApplyPatch>", () => {
  it("parses <ApplyPatch>...</ApplyPatch> without <tool_calls> wrapper", () => {
    const text = `<ApplyPatch>
<parameters>{"patch":"*** Begin Patch\\n*** Modify File\\n- old\\n+ new"}</parameters>
</ApplyPatch>`;
    const call = parseOne(text);
    assert.equal(call.name, "ApplyPatch");
    assert.ok(call.input.patch);
  });

  it("parses <Shell>...</Shell> without <tool_calls> wrapper", () => {
    const text = `<Shell>
<parameters>{"command":"ls -la"}</parameters>
</Shell>`;
    const call = parseOne(text);
    assert.equal(call.name, "Shell");
    assert.equal(call.input.command, "ls -la");
  });

  it("parses <ReadFile> with <parameter> sub-tags, no wrapper", () => {
    const text = `<ReadFile>
<parameter name="path">./test.py</parameter>
</ReadFile>`;
    const call = parseOne(text);
    assert.equal(call.name, "ReadFile");
    assert.equal(call.input.path, "./test.py");
  });
});

/* ═══════════════════════════════════════════════════
   Sieve: tool-name-specific capture pairs
   ═══════════════════════════════════════════════════ */

describe("Sieve: tool-name-specific capture", () => {
  it("detects <ApplyPatch> tag in streaming sieve", () => {
    const sieve = createToolSieve(ALL_TOOLS);
    const events1 = sieve.push("Here is the patch:\n");
    const events2 = sieve.push("<ApplyPatch>\n");
    const events3 = sieve.push("<parameters>{\"patch\":\"*** Begin Patch\"}</parameters>\n");
    const events4 = sieve.push("</ApplyPatch>");
    const flushEvents = sieve.flush();
    
    const allEvents = [...events1, ...events2, ...events3, ...events4, ...flushEvents];
    const toolCallEvents = allEvents.filter(e => e.type === "tool_calls");
    assert.equal(toolCallEvents.length, 1, `Expected 1 tool_call event, got ${toolCallEvents.length}`);
    assert.equal(toolCallEvents[0].calls[0].name, "ApplyPatch");
  });

  it("detects <Shell> tag in streaming sieve", () => {
    const sieve = createToolSieve(ALL_TOOLS);
    const events1 = sieve.push("Running command:\n");
    const events2 = sieve.push("<Shell><parameters>{\"command\":\"ls\"}</parameters></Shell>");
    const flushEvents = sieve.flush();
    
    const allEvents = [...events1, ...events2, ...flushEvents];
    const toolCallEvents = allEvents.filter(e => e.type === "tool_calls");
    assert.equal(toolCallEvents.length, 1);
    assert.equal(toolCallEvents[0].calls[0].name, "Shell");
  });

  it("still detects standard <tool_calls> in sieve", () => {
    const sieve = createToolSieve(ALL_TOOLS);
    const events1 = sieve.push("<tool_calls>\n");
    const events2 = sieve.push("<tool_name>Shell</tool_name>\n");
    const events3 = sieve.push("<parameters>{\"command\":\"ls\"}</parameters>\n");
    const events4 = sieve.push("</tool_calls>");
    const flushEvents = sieve.flush();
    
    const allEvents = [...events1, ...events2, ...events3, ...events4, ...flushEvents];
    const toolCallEvents = allEvents.filter(e => e.type === "tool_calls");
    assert.equal(toolCallEvents.length, 1);
    assert.equal(toolCallEvents[0].calls[0].name, "Shell");
  });

  it("sieve.emittedText tracks emitted content", () => {
    const sieve = createToolSieve(ALL_TOOLS);
    sieve.push("Hello ");
    sieve.push("world");
    sieve.flush();
    assert.ok(sieve.emittedText.includes("Hello world"));
  });
});
