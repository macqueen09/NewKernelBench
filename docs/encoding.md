# Encoding Policy

## Verified environment state

Remote Linux host:

- locale reports `LANG=en_US.UTF-8`
- the repaired summary markdown is detected as `charset=utf-8`

Local Windows shell:

- code page was set to `65001`
- PowerShell output encoding was set to `utf-8`

## Project policy

All human-readable files in `NewKernelBench` should use UTF-8 text encoding:

- `*.md`
- `*.py`
- `*.json`
- `*.toml`

The project-level `.editorconfig` enforces UTF-8 for editors that support it.

## Windows commands

Use these commands before manually generating or editing Chinese and English content from the terminal:

```powershell
chcp 65001
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$env:PYTHONUTF8 = '1'
```

## Linux commands

UTF-8 is already active on the remote host, but these are the safe defaults:

```bash
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export PYTHONUTF8=1
```

## File-writing rules

When scripts write text files, always pass an explicit UTF-8 encoding, for example:

```python
Path(output_path).write_text(text, encoding="utf-8")
```

## Validation

Run the built-in encoding check to verify that project files are valid UTF-8:

```bash
python scripts/check_encoding.py
```
