# Contract requests moved

Requests now live in one file per agent, so no two agents ever edit the same file:

- `docs/requests/from-a.md`, `from-b.md`, `from-c.md`, `from-d.md`: what that agent needs from others, and its replies.
- `docs/requests/from-pm.md`: decisions from Chiya and the PM chat.

Write only in your own file. To answer someone, add a `REPLY [you -> them]` line in your file naming their request; the asker moves their line to Done. If merging `main` gives you a conflict on this file, move your new lines into your own file and keep this one as is.
