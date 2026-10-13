[.[] | select(.reason == "compiler-artifact" and .target.name == "workflow" and
  (.target.kind | index("test")) and .executable != null)] |
if length == 1 then .[0].executable
else error("Expected exactly one workflow test runner") end
