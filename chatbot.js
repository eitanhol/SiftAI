#!/usr/bin/env node
"use strict";

/**
 * chatbot.js
 * - Always runs check_file.py on the provided CSV to (re)generate data_schema.json.
 * - Uses fixed filenames: data_plan.json, data_result.json, data_result.png, data_result.html.
 * - Runs Pyodide sandbox by default (use --no-sandbox to skip).
 * - Implements a retry-and-explain mechanism for code execution errors.
 *
 * Usage:
 * node chatbot.js <csv_path_any> "<question>"
 * [--no-sandbox]                     // opt-out; defaults to sandbox ON using <csv_path_any>
 * [--run-sandbox <csv_path>]         // optional override for which CSV to execute
 * [--out data_plan.json]             // optional; fixed default is data_plan.json
 * [--model gemini-2.5-flash]
 * [--output table|viz|both]          // merged into params.output
 * [--params '{"output":"both"}']     // extra params merged in
 */

require("dotenv").config();

const fs = require("fs");
const path = require("path");
const { execSync } = require("child_process");
const { GoogleGenerativeAI } = require("@google/generative-ai");

// ---------- CLI ----------
const args = process.argv.slice(2);
if (args.length < 2) {
  console.error(
    'Usage: node chatbot.js <csv_path_any> "<question>" [--no-sandbox] [--run-sandbox <csv_path>] [--out data_plan.json] [--model gemini-2.5-flash] [--output table|viz|both] [--params \'{"output":"both"}\']'
  );
  process.exit(1);
}

const csvPath = args[0];
const question = args[1];

function getFlag(name, defVal) {
  const i = args.indexOf(name);
  if (i === -1) return defVal;
  return i + 1 < args.length ? args[i + 1] : defVal;
}
function hasFlag(name) {
  return args.includes(name);
}

// Fixed filenames
const outFile = getFlag("--out", "data_plan.json");
const modelName = getFlag("--model", "gemini-2.5-flash");
const schemaPath = path.join(process.cwd(), "data_schema.json");

// Default: sandbox ON using the provided CSV; allow --no-sandbox to disable or --run-sandbox to override CSV path
const sandboxEnabled = !hasFlag("--no-sandbox");
const runCsvPath = sandboxEnabled ? getFlag("--run-sandbox", csvPath) : null;

// Optional JSON params
let params = {};
const outputMode = getFlag("--output", null); // table|viz|both
if (outputMode) params.output = outputMode;
if (hasFlag("--params")) {
  const idx = args.indexOf("--params");
  if (idx !== -1 && idx + 1 < args.length) {
    const raw = args[idx + 1];
    try {
      params = { ...params, ...JSON.parse(raw) };
    } catch {
      try {
        params = { ...params, ...JSON.parse(raw.replace(/`"/g, '"')) };
      } catch {
        console.warn("[warn] Could not parse --params JSON. Falling back to switches/defaults.");
      }
    }
  }
}
if (!("output" in params)) params.output = "both";

// ---------- Sanity ----------
if (!process.env.GOOGLE_API_KEY) {
  console.error("[ERROR] Missing GOOGLE_API_KEY environment variable.");
  process.exit(3);
}
if (!fs.existsSync(csvPath)) {
  console.error(`[ERROR] CSV not found at: ${csvPath}`);
  process.exit(4);
}

// ---------- Ensure schema by running check_file.py ----------
function findPython() {
  for (const cmd of ["python", "python3", "py -3", "py"]) {
    try {
      execSync(`${cmd} --version`, { stdio: "ignore", shell: true });
      return cmd;
    } catch (_) {}
  }
  return null;
}
const py = findPython();
if (!py) {
  console.error("[ERROR] Could not find a Python interpreter in PATH (python/python3/py).");
  process.exit(2);
}


// ---------- Read Schema (ONLY context we provide to the model) ----------
let schema;
try {
  const schemaText = fs.readFileSync("data_schema.json", "utf8");
  schema = JSON.parse(schemaText);
  if (!Array.isArray(schema)) {
    throw new Error("Schema must be a JSON array of { 'column-name', 'type' } objects.");
  }
} catch (e) {
  console.error("[ERROR] Failed to read/parse schema JSON:", e.message);
  process.exit(4);
}
const columns = schema.map((c) => c["column-name"]);
if (columns.length === 0) {
  console.error("[ERROR] Schema contains no columns.");
  process.exit(4);
}

// ---------- System rules (model prompt) ----------
const SYSTEM_RULES = `
You are a precise data coder. You will be given ONLY:
- the contents of data_schema.json (a JSON array of { "column-name": string, "type": SQL-like type }),
- and a user question.

Return STRICT JSON (no Markdown, no backticks) with EXACTLY these keys:
{
  "summary": "<3-5 sentences, in plain English, describing the logical process or methodology that will be used to arrive at the answer.>"
  "code": "<a single Python module that defines def solve(data, params=None)>"
}

### Python code requirements

1) Module API (backward-compatible):
   - Define ONE function:
       def solve(data, params=None):
           ...
     * "data" is an Array[Object] (rows) with string values; parse numbers/dates as needed.
     * "params" is optional. If provided, it will ALWAYS be a standard Python dict (parsed from JSON).
       Never assume a JS proxy/object. If None/omitted, default to {"output": "both"}.
     - * Environment: Pyodide (WebAssembly). Available: numpy, scipy, pandas, matplotlib.
     - * Allowed imports: stdlib, numpy, scipy, pandas, matplotlib.
     - * Do NOT use ortools. Avoid pulp in this environment.
     - * For assignment/optimization, prefer scipy.optimize.linear_sum_assignment or pure NumPy.
     * Do NOT read files or call the network (other than pip install for missing packages in native CPython).
     * No console output. Return a JSON-serializable Python dict.

2) Output control (must honor):
   - params["output"] ∈ {"table","viz","both"}; default "both".
   - If "table": return the table section (viz may be null).
   - If "viz": return the viz section (table may be null).
   - If "both": return both.

3) Headless plotting:
   - Force non-interactive backend:
       import matplotlib
       matplotlib.use("Agg")
       import matplotlib.pyplot as plt

4) Visualization behavior:
   - If params["chart_type"] is omitted or "auto", infer a reasonable chart from the question + data types.
   - Include base64 PNG if viz requested; close figures.

5) Filters / aggregation (optional but supported if present):
   - params may include: chart_type, x, y, agg ("count"|"sum"|"mean"|"median"|"min"|"max"),
     filters (list of {col, op, val}), top_n (int), bins (int), max_rows (int, default 50),
     include_png (bool, default true), figsize ([w,h]).
   - Be robust to "", "na", "n/a", null.

6) STRICT return shape (single dict):
   return {
     "ok": true,
     "spec": {
       "output": "<table|viz|both>",
       "chart_type": "<resolved type or 'auto'>",
       "x": <string or null>,
       "y": <string | [strings] | null>,
       "agg": <string or null>,
       "filters": <list>,
       "top_n": <int or null>,
       "bins": <int or null>,
       "max_rows": <int>,
       "include_png": <bool>
     },
     "data": { "columns": [ ... ], "rows": [ [ ... ], ... ] },
     "viz": { "type": "<bar|line|hist|scatter|pie|...>", "png_base64": "<...>", "width_px": <int>, "height_px": <int>, "description": "<...>" } or null,
     "errors": [ "<non-fatal warnings or empty>" ]
   }

7) Implementation notes:
   - Parse strings into numbers/dates as needed.
   - Implement explicit grouping/aggregation when implied.
   - Respect params["output"] exactly; if impossible, set ok=false with a helpful error.
   - Never print; only return the dict above.
   - Keep the module self-contained in a single code block assigned to "code".
   - **Unless the user explicitly requests otherwise, always limit results to the top 20 values (use params["top_n"] = 20). If there are fewer than 20 values, return them all.**

Return ONLY the JSON object—no extra text.
`.trim();

const USER_CONTEXT = {
  question,
  schema_info: {
    schema_file: path.basename(schemaPath),
    columns_with_types: schema,
  },
};

// ---------- Gemini Helper ----------
async function callGenerativeModel(systemInstruction, userText, modelName, responseSchema) {
    const genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY);
    const model = genAI.getGenerativeModel({
      model: modelName,
      systemInstruction,
      generationConfig: {
        responseMimeType: "application/json",
        responseSchema,
      },
    });
  
    const result = await model.generateContent({
      contents: [{ role: "user", parts: [{ text: userText }] }],
      generationConfig: {
        responseMimeType: "application/json",
        responseSchema,
      },
    });
  
    const response = result.response;
    const text = typeof response.text === "function" ? response.text() : String(response.text || "");
    return JSON.parse(text);
}


// ---------- Fixed result filenames ----------
function deriveResultFilesFixed() {
  return {
    resultJson: "data_result.json",
    resultPng: "data_result.png",
    resultHtml: "data_result.html",
  };
}

// ---------- Pyodide sandbox runner ----------
async function runInSandboxPyodide(codeString, csvPathLocal, paramsObj) {
  let loadPyodide;
  try {
    ({ loadPyodide } = await import("pyodide"));
  } catch (e) {
    console.error(
      "[ERROR] Pyodide not found. Install it with:\n  npm i pyodide\n\n" +
      "Then re-run (sandbox is default)."
    );
    process.exit(7);
  }

  const pyodide = await loadPyodide();

  // Always load matplotlib; detect and pre-load other packages used by the generated code
  const needs = new Set(["matplotlib"]);

  const wants = [
    { key: "pandas",       patterns: [/import\s+pandas\b/, /from\s+pandas\b/, /_ensure_pkg\(["']pandas["']\)/] },
    { key: "numpy",        patterns: [/import\s+numpy\b/, /from\s+numpy\b/, /\bnp\./, /_ensure_pkg\(["']numpy["']\)/] },
    { key: "scipy",        patterns: [/import\s+scipy\b/, /from\s+scipy\b/, /_ensure_pkg\(["']scipy["']\)/] },
    { key: "pulp",         patterns: [/import\s+pulp\b/, /from\s+pulp\b/, /_ensure_pkg\(["']pulp["']\)/] },
    { key: "scikit-learn", patterns: [/import\s+sklearn\b/, /from\s+sklearn\b/, /_ensure_pkg\(["']scikit-?learn["']\)/] },
  ];

  for (const w of wants) {
    if (w.patterns.some((rx) => rx.test(codeString))) needs.add(w.key);
  }

  // Explicitly block ortools in Pyodide (not supported)
  if (/(import|from)\s+ortools\b|_ensure_pkg\(["']ortools["']\)/.test(codeString)) {
    console.error(
      "[ERROR] The generated code requires 'ortools', which is not available in Pyodide. " +
      "Ask the model to use 'pulp' instead (LP/MIP) or a pure-NumPy/Scipy approach."
    );
    process.exit(8);
  }

  // Pre-load detected packages
  try {
    await pyodide.loadPackage(Array.from(needs));
  } catch (e) {
    console.error("[ERROR] Failed to pre-load one or more Pyodide packages:", e?.message || e);
    console.error("Tried to load:", Array.from(needs).join(", "));
    process.exit(9);
  }

  // 1) Put CSV into sandbox FS
  const csvText = fs.readFileSync(csvPathLocal, "utf8");
  const enc = new TextEncoder();
  pyodide.FS.writeFile("/data.csv", enc.encode(csvText));

  // 2) Inject code + params JSON string
  pyodide.globals.set("PLAN_CODE", codeString);
  pyodide.globals.set("CSV_VPATH", "/data.csv");
  pyodide.globals.set("JS_PARAMS_JSON", JSON.stringify(paramsObj || { output: "both" }));

  // 3) Execute in Python
  const jsonResult = await pyodide.runPythonAsync(
    `
import importlib.util, json, csv

with open("/plan_module.py", "w", encoding="utf-8") as f:
    f.write(PLAN_CODE)

spec = importlib.util.spec_from_file_location("plan_module", "/plan_module.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

if not hasattr(mod, "solve"):
    raise RuntimeError("Generated code has no solve(data, params=None)")

DATA_ROWS = []
with open(CSV_VPATH, "r", encoding="utf-8") as fh:
    reader = csv.DictReader(fh)
    for row in reader:
        DATA_ROWS.append({ k: ("" if v is None else str(v)) for k, v in row.items() })

try:
    P_PARAMS = json.loads(JS_PARAMS_JSON)
    if not isinstance(P_PARAMS, dict):
        P_PARAMS = {"output": "both"}
except Exception:
    P_PARAMS = {"output": "both"}

res = mod.solve(DATA_ROWS, P_PARAMS)
json.dumps(res, ensure_ascii=False)
`.trim()
  );

  return jsonResult; // stringified JSON
}

// ---------- Main ----------
(async () => {
  try {
    console.log(
      `[info] Asking ${modelName} for summary+code for: "${question}" (Using schema: data_schema.json)`
    );

    const initialUserPrompt =
      `Question:\n${USER_CONTEXT.question}\n\n` +
      `Schema (filename ${USER_CONTEXT.schema_info.schema_file}):\n` +
      JSON.stringify(USER_CONTEXT.schema_info.columns_with_types, null, 2) +
      `\n\nReturn ONLY the JSON with keys "summary" and "code", where "code" is a single Python module defining def solve(data, params=None).`;

    const codeResponseSchema = {
        type: "object",
        required: ["summary", "code"],
        properties: {
          summary: { type: "string" },
          code: { type: "string" }
        }
    };
    
    const initialGen = await callGenerativeModel(SYSTEM_RULES, initialUserPrompt, modelName, codeResponseSchema);

    fs.writeFileSync(outFile, JSON.stringify(initialGen, null, 2), "utf8");
    console.log(`[ok] Wrote ${outFile}`);
    console.log(`[summary]\n${initialGen.summary}\n`);

    if (runCsvPath) {
      let currentCode = initialGen.code;
      let lastError = null;
      let executionResult = null;
      const MAX_ATTEMPTS = 2;
      const originalCode = initialGen.code;
      let firstError = null;

      for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
        try {
          console.log(`[sandbox] Attempt ${attempt}: Executing generated code in Pyodide on: ${runCsvPath}`);
          const jsonStr = await runInSandboxPyodide(currentCode, runCsvPath, params);
          
          try {
            executionResult = JSON.parse(jsonStr);
          } catch (parseError) {
              throw new Error(`Sandbox returned non-JSON output:\n${jsonStr}`);
          }

          lastError = null; // Success, clear error
          console.log(`[ok] Attempt ${attempt} succeeded.`);
          break; // Exit loop on success
        } catch (err) {
          console.warn(`[warn] Attempt ${attempt} failed.`);
          console.error(err.message);
          lastError = err;
          
          if (attempt === 1) {
              firstError = err;
          }

          if (attempt < MAX_ATTEMPTS) {
            console.log(`[info] Asking ${modelName} to fix the code...`);
            
            const fixUserPrompt = `The previous code you generated failed with an error.
Please analyze the error and provide a fix.

Original Question:
${question}

Schema (filename ${USER_CONTEXT.schema_info.schema_file}):
${JSON.stringify(USER_CONTEXT.schema_info.columns_with_types, null, 2)}

--- FAILED CODE ---
${currentCode}

--- ERROR MESSAGE ---
${err.stack || err.message}

---
Return ONLY the corrected JSON object with "summary" and "code" keys. The new code should fix the error.`;

            const fixedGen = await callGenerativeModel(SYSTEM_RULES, fixUserPrompt, modelName, codeResponseSchema);
            
            currentCode = fixedGen.code;
            console.log(`[info] Received a potential fix. Retrying execution...`);
            console.log(`[summary for fix]\n${fixedGen.summary}\n`);
          }
        }
      }

      // After the loop, check the outcome
      if (lastError) {
        console.error(`[error] All ${MAX_ATTEMPTS} attempts failed. Asking model for an explanation.`);
        
        const EXPLANATION_SYSTEM_RULES = `You are a Python debugging expert.
A user's code failed, you provided a fix, but the fix also failed.
Your task is to provide a clear, concise explanation of the underlying problem.
Analyze the original code, the first error, the revised code, and the second error.
Explain why the code is failing and why it might be difficult for you to generate a correct solution in this context.

Return STRICT JSON (no Markdown, no backticks) with EXACTLY this key:
{
  "explanation": "<Your detailed analysis and explanation here>"
}`;
        
        const explanationUserPrompt = `The original code and your proposed fix both failed. Please explain the issue.

--- ORIGINAL CODE ---
${originalCode}

--- FIRST ERROR ---
${firstError.stack || firstError.message}

--- YOUR REVISED CODE (FIX ATTEMPT) ---
${currentCode}

--- SECOND ERROR ---
${lastError.stack || lastError.message}

---
Provide an explanation for the failure.`;

        const explanationSchema = {
            type: "object",
            required: ["explanation"],
            properties: {
              explanation: { type: "string" },
            }
        };

        const explanationObj = await callGenerativeModel(EXPLANATION_SYSTEM_RULES, explanationUserPrompt, modelName, explanationSchema);

        console.error("\n--- AI EXPLANATION OF FAILURE ---");
        console.error(explanationObj.explanation);
        console.error("-----------------------------------\n");
        process.exit(6);
      }

      if (executionResult) {
        // Success path
        const { resultJson, resultPng } = deriveResultFilesFixed();
        
        const parsed = executionResult;
        fs.writeFileSync(resultJson, JSON.stringify(parsed, null, 2), "utf8");
        console.log(`[ok] Wrote result JSON to ${resultJson}`);

        if (parsed.viz && parsed.viz.png_base64) {
          const buf = Buffer.from(parsed.viz.png_base64, "base64");
          fs.writeFileSync(resultPng, buf);
          console.log(`[ok] Wrote visualization PNG to ${resultPng}`);
        } else {
          console.log("[info] No visualization PNG in result (viz missing or no png_base64).");
        }
      }

    } else {
      console.log("[info] Sandbox disabled by --no-sandbox.");
    }
  } catch (err) {
    console.error("[error]", err && err.stack ? err.stack : err);
    process.exit(5);
  }
})();