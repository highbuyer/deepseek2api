import { describe, it } from "node:test";
import assert from "node:assert";
import { stripLeakedMarkers, finalStrip, createStreamTextStripper } from "../src/utils/strip-markers.js";

/* ── stripLeakedMarkers ── */

describe("stripLeakedMarkers", () => {

  describe("block-level stripping", () => {
    it("strips <tool_result> block with content", () => {
      const input = 'some text\n<tool_result id="Read">\nfile contents here\n</tool_result>\nmore text';
      const result = stripLeakedMarkers(input);
      assert.ok(!result.includes("file contents"));
      assert.ok(result.includes("some text"));
      assert.ok(result.includes("more text"));
    });

    it("strips <think> block with content", () => {
      const input = 'hello\n<think>\nreasoning here\n</think>\nworld';
      const result = stripLeakedMarkers(input);
      assert.ok(!result.includes("reasoning"));
      assert.ok(result.includes("hello"));
      assert.ok(result.includes("world"));
    });

    it("strips <invoke> block entirely (not just tags)", () => {
      const input = 'before\n<invoke name="Read">\n<parameter name="path">/home/user/file.js</parameter>\n</invoke>\nafter';
      const result = stripLeakedMarkers(input);
      assert.ok(!result.includes("/home/user/file.js"), "file path should not leak");
      assert.ok(!result.includes("Read"), "tool name should not leak");
      assert.ok(result.includes("before"));
      assert.ok(result.includes("after"));
    });

    it("strips <tool_call> block entirely", () => {
      const input = 'text\n<tool_call>\n<tool_name>Grep</tool_name>\n<parameters>{"pattern":"secret"}</parameters>\n</tool_call>\nmore';
      const result = stripLeakedMarkers(input);
      assert.ok(!result.includes("Grep"), "tool name should not leak");
      assert.ok(!result.includes("secret"), "parameter values should not leak");
      assert.ok(result.includes("text"));
      assert.ok(result.includes("more"));
    });

    it("strips <function_calls> wrapper with nested <invoke>", () => {
      const input = 'result:\n<function_calls>\n<invoke name="Read">\n<parameter name="path">/tmp/log.txt</parameter>\n</invoke>\n</function_calls>\ndone';
      const result = stripLeakedMarkers(input);
      assert.ok(!result.includes("/tmp/log.txt"));
      assert.ok(!result.includes("Read"));
      assert.ok(result.includes("result:"));
      assert.ok(result.includes("done"));
    });
  });

  describe("malformed close tags", () => {
    it("strips </calls> (missing prefix + has >)", () => {
      const input = 'the sieve looks for </calls> in the stream';
      const result = stripLeakedMarkers(input);
      assert.ok(!result.includes("</calls>"), `got: "${result}"`);
      assert.ok(result.includes("the sieve looks for"));
      assert.ok(result.includes("in the stream"));
    });

    it("strips </calls without closing >", () => {
      const input = 'stuff...</calls comes after the block';
      const result = stripLeakedMarkers(input);
      assert.ok(!result.includes("</calls"), `got: "${result}"`);
    });

    it("strips </calls<parameter...> malformed fragment", () => {
      const input = '...</calls<parameter name="x">value';
      const result = stripLeakedMarkers(input);
      // Should strip the XML fragment, may or may not keep "value" depending
      // on greedy matching. The key invariant: </calls must be gone.
      assert.ok(!result.includes("</calls"), `got: "${result}"`);
    });

    it("strips </function_calls without >", () => {
      const input = 'block ends with </function_calls and then text';
      const result = stripLeakedMarkers(input);
      assert.ok(!result.includes("</function_calls"), `got: "${result}"`);
    });
  });

  describe("tag-only stripping (LEAK_TAG_REGEX)", () => {
    it("removes orphan <parameter> tags but keeps content", () => {
      const input = 'name="pattern"<parameter name="-n">true</parameter>rest';
      const result = stripLeakedMarkers(input);
      // Tags gone, text content between them survives
      assert.ok(!result.includes("<parameter"), `got: "${result}"`);
      assert.ok(result.includes('name="pattern"'));
      assert.ok(result.includes("true"));
    });
  });

  describe("role marker stripping", () => {
    it("removes TOOL: prefixes", () => {
      const input = "some text\nTOOL: this should be gone\nmore text";
      const result = stripLeakedMarkers(input);
      assert.ok(!result.includes("TOOL:"));
      assert.ok(result.includes("some text"));
      assert.ok(result.includes("more text"));
    });

    it("strips ASSISTANT + space + content (no colon, model used as heading)", () => {
      const input = "answer\n\nASSISTANT 好，结果是这样";
      const result = stripLeakedMarkers(input);
      assert.ok(!/^\s*(?:USER|ASSISTANT|TOOL)[\s:]/im.test(result), `marker not stripped: "${result}"`);
      assert.ok(result.includes("好，结果是这样"), `content lost: "${result}"`);
    });

    it("strips ASSISTANT + space + markdown backtick", () => {
      const input = "text\n\nASSISTANT `mcp__gitnexus__context` returned data";
      const result = stripLeakedMarkers(input);
      assert.ok(!/^\s*ASSISTANT\b/im.test(result), `banner not stripped: "${result}"`);
      assert.ok(result.includes("`mcp__gitnexus__context`"), `markdown lost: "${result}"`);
    });

    it("does NOT strip ASSISTANT adjacent to non-whitespace (legit text)", () => {
      const input = "the ASSISTANT-class helper does the work";
      const result = stripLeakedMarkers(input);
      assert.ok(result.includes("ASSISTANT-class"), `legit text destroyed: "${result}"`);
    });

    it("strips ASSISTANT alone on its own line (banner without colon or content)", () => {
      const input = "answer here\n\nASSISTANT\n\nfollow-up content";
      const result = stripLeakedMarkers(input);
      assert.ok(!/^\s*ASSISTANT\s*$/im.test(result), `bare banner not stripped: "${result}"`);
      assert.ok(result.includes("follow-up content"), `follow-up lost: "${result}"`);
    });
  });

  describe("fast path", () => {
    it("returns clean text unchanged", () => {
      const input = "hello world, how are you?";
      const result = stripLeakedMarkers(input);
      assert.equal(result, input);
    });
  });
});

/* ── finalStrip ── */

describe("finalStrip", () => {

  describe("malformed close tags", () => {
    it("strips </calls>", () => {
      const input = 'text </calls> more';
      const result = finalStrip(input);
      assert.ok(!result.includes("</calls>"), `got: "${result}"`);
    });

    it("strips </calls without >", () => {
      const input = 'text </calls more';
      const result = finalStrip(input);
      assert.ok(!result.includes("</calls"), `got: "${result}"`);
    });

    it("strips </calls<parameter...> fragment", () => {
      const input = '...</calls<parameter name="x">value';
      const result = finalStrip(input);
      assert.ok(!result.includes("</calls"), `got: "${result}"`);
      assert.ok(!result.includes("<parameter"), `got: "${result}"`);
    });
  });

  describe("unclosed open tags", () => {
    it("strips <tool_calls without >", () => {
      const input = 'some <tool_calls and then more text';
      const result = finalStrip(input);
      assert.ok(!result.includes("<tool_calls"), `got: "${result}"`);
    });

    it("strips <function_calls (unclosed, no >)", () => {
      const input = 'the <function_calls tag is broken';
      const result = finalStrip(input);
      assert.ok(!result.includes("<function_calls"), `got: "${result}"`);
    });
  });

  describe("CDATA wrappers", () => {
    it("strips <![CDATA[ and ]]>", () => {
      const input = "some <![CDATA[hidden]]> text";
      const result = finalStrip(input);
      assert.ok(!result.includes("CDATA"));
      assert.ok(result.includes("hidden"));
    });
  });

  describe("blank line collapse", () => {
    it("collapses 3+ blank lines", () => {
      const input = "line1\n\n\n\nline2";
      const result = finalStrip(input);
      assert.equal(result, "line1\n\nline2");
    });
  });
});

/* ── createStreamTextStripper ── */

describe("createStreamTextStripper", () => {

  describe("marker prefix holdback", () => {
    it("holds TO until OL: arrives then emits clean", () => {
      const s = createStreamTextStripper();
      const out1 = s.push("some TO");
      assert.ok(!out1.includes("TO"), `held "TO" but got: "${out1}"`);
      const out2 = s.push("OL: rest");
      assert.ok(!out2.includes("TOOL:"), `stripped TOOL: but got: "${out2}"`);
    });

    it("holds US until ER: arrives then emits clean", () => {
      const s = createStreamTextStripper();
      s.push("text US");
      const out = s.push("ER: more");
      assert.ok(!out.includes("USER:"), `got: "${out}"`);
    });

    it("holds ASS until ISTANT: arrives then emits clean", () => {
      const s = createStreamTextStripper();
      s.push("some ASS");
      const out = s.push("ISTANT: text");
      assert.ok(!out.includes("ASSISTANT:"), `got: "${out}"`);
    });

    it("strips [proxy]</think> artifact", () => {
      const s = createStreamTextStripper();
      s.push("text [proxy]</think> more");
      const out = s.flush();
      assert.ok(!out.includes("[proxy]</think>"), `got: "${out}"`);
    });
  });

  describe("leak tag stripping during stream", () => {
    it("strips </calls> mid-stream", () => {
      const s = createStreamTextStripper();
      s.push("before ");
      const out = s.push("</calls> after");
      assert.ok(!out.includes("</calls>"), `got: "${out}"`);
    });

    it("strips </calls when it arrives whole (not split)", () => {
      const s = createStreamTextStripper();
      const out = s.push("text </calls </calls> more");
      assert.ok(!out.includes("</calls"), `got: "${out}"`);
    });

    it("strips <invoke> block with parameter values mid-stream", () => {
      const s = createStreamTextStripper();
      s.push("start\n<invoke name=\"Read\">\n<parameter name=\"path\">/secret/path.js</parameter>\n</invoke>\nend");
      const out = s.flush();
      assert.ok(!out.includes("/secret/path.js"), `leaked path in: "${out}"`);
      assert.ok(!out.includes("Read"), `leaked name in: "${out}"`);
    });
  });

  describe("flush", () => {
    it("emits clean text on push when no markers at tail", () => {
      const s = createStreamTextStripper();
      const out = s.push("hello world");
      assert.ok(out.includes("hello world"));
    });

    it("emits held-back partial prefix on flush (no next chunk)", () => {
      const s = createStreamTextStripper();
      const out1 = s.push("text TO");  // "TO" held back as potential TOOL: prefix
      assert.ok(!out1.includes("TO"), `push should not emit held-back "TO": "${out1}"`);
      const out2 = s.flush();
      // "TO" alone is harmless — only dangerous when followed by "OL:" in next chunk
      assert.ok(out2.includes("TO"), `flush should release safe held-back text, got: "${out2}"`);
    });
  });
});
