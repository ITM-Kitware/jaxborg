# Studies

Give each study a directory named for its question. Its README should let a colleague find the question, the comparison, the finding and the evidence in one scan.

## Plan the comparison

Start with the decision the result could change. Cite the motivating paper, slide or earlier study with its version and page when available. Check whether saved measurements or policies can answer the question before scheduling training.

Record the conditions, controls, independent runs, evaluation cases, budget and selection rule before running. State how missing cases will be handled. Label comparisons chosen after seeing outcomes as exploratory.

## Write the study README

Use these headings. Put the important words first. Aim for one or two sentences per section, or use short bullets or a table when readers need to compare values.

| Heading | What the reader needs |
| --- | --- |
| Abstract | Answer the question and give its scope. For ongoing work, state the question and what is pending. |
| Background and motivation | Link the earlier claim and explain what this comparison resolves. |
| Methods | Say what changed, what stayed fixed and how performance was measured. Link the full protocol or recipe. |
| Results | Lead with the difference and comparator. Give completed versus planned independent runs when known. Explain meaningful exclusions and what varies in any interval. Link the selected measurements beside the claim. |
| Conclusions | State what the finding supports and what remains open for the decision. Put any qualification that changes the meaning of a number beside that number. |

## Check what actually ran

Read the resolved settings and the analyzer. In Jaxborg, check checkpoint and training budget, familiar or held-out layouts, opponents, episode seeds, action selection (sampling or argmax), score and aggregation. Count independently trained policies separately from retries and evaluation episodes. Explain consequential differences before combining scores.

## Keep results reproducible

- **In Git** keep the question, protocol, recipe or configuration, analysis code and rebuild command. Put lengthy execution instructions in a linked file when needed.
- **In shared storage** keep selected checkpoints, resolved settings and per-episode measurements. Retain traces or logs used to support a finding. Link a shared location or MLflow run ID and artifact path, with hashes for selected inputs.
- **For each reported version** record the producing and analysis revisions and exact input selection. Give a changed selection or calculation a new report, link the prior version and say what changed.
- **Before handoff** retrieve the selected inputs in a fresh directory and rebuild one reported value. Disclose any unavailable evidence or unverified result.
