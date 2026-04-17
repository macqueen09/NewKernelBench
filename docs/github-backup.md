# GitHub Backup Setup

## Goal

Use `NewKernelBench` as a standalone git repository so that:

- GitHub can be used as a backup
- VSCode Remote can show file changes against the last committed version
- the repository stays isolated from any parent folder

## Current verified state

- `NewKernelBench` is not inside another git repository
- its parent directory is also not a git repository
- this means a nested-repo conflict does not exist right now

## Recommended repository name

Use:

- owner: `macqueen09`
- repo: `NewKernelBench`

## Local repository setup

The repository has already been initialized inside:

```text
/supercloud/llm-code/mkl/project/clang/KernelGen/NewKernelBench
```

Useful git commands from this folder:

```bash
git status
git diff
git add -A
git commit -m "Update NewKernelBench"
```

## SSH-based push flow

Recommended flow:

1. create or reuse an SSH key on the remote server
2. add the public key to GitHub SSH settings
3. create an empty GitHub repository named `NewKernelBench`
4. add the remote and push

## Commands after the GitHub repo exists and the SSH key is added

```bash
cd /supercloud/llm-code/mkl/project/clang/KernelGen/NewKernelBench
git remote add origin git@github.com:macqueen09/NewKernelBench.git
git push -u origin main
```

If `origin` already exists:

```bash
git remote set-url origin git@github.com:macqueen09/NewKernelBench.git
git push -u origin main
```

## Why this works for VSCode diff

Once the repository is initialized and committed, VSCode Remote can already show:

- modified files
- added files
- deleted files
- line-level diffs against the last commit

GitHub is only needed for remote backup and sync, not for local diff visibility.
