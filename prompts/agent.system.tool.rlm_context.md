### rlm_context

run recursive long-context analysis against recent Agent Zero state when the default context window is too tight
use this tool when you want explicit RLM analysis instead of waiting for auto-routing
prefer source "recent_history" unless you need a narrower scope
always provide a concise "question"

usage:

~~~json
{
  "thoughts": [
    "The latest tool output is too large to reason over comfortably in the normal context window.",
    "I will run explicit recursive analysis over recent history."
  ],
  "headline": "Running recursive long-context analysis",
  "tool_name": "rlm_context",
  "tool_args": {
    "question": "Summarize the important findings and recommend the next step.",
    "source": "recent_history",
    "history_turns": 6
  }
}
~~~
