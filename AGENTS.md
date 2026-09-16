# Instructions for coding agents

- `src/app/` is user-owned. Do not create, edit, move, or delete files inside it unless the user explicitly authorizes that specific change.
- Place AI-generated tutorial/model code in `src/ai_guide/`.
- Do not overwrite the user's existing changes elsewhere in the repository.
- Treat `data/` as read-only input. Save generated checkpoints and predictions outside it.
- Use Conventional Commits for commit messages (for example, `feat: add supervised training`). Separate unrelated changes into independent commits to make code review easier.
