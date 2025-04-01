# test module

This is a module for TICO unittest.

## How to debug using vscode?

1. Add below configuration to `.vscode/launch.json`.
2. Make a breakpoint on the line you want.
3. Run the configuration with `Run and Debug`.

**TIP** Install this project in editable mode (pip install -e) for interactive test.

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug unit test (pt2_to_circle)",
            "type": "debugpy",
            "request": "launch",
            "module": "unittest",
            "args": ["test.pt2_to_circle_test.test_pt2_to_circle"],
            "cwd": "${workspaceFolder}",
            "console": "integratedTerminal",
            "justMyCode": false,
        },
    ]
}
```