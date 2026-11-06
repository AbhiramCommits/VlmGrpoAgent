# PERCEPTION example

Perception failure: the trajectory never surfaced the visual facts ['bar:Sun', 'bar:Fri', 'bar:Wed', 'bar:Sat']; no downstream arithmetic could be right.

- task: `test-chart_agg-0000-1234` (chart_agg)
- question: The bar chart shows orders shipped per day. What is the mean of the three lowest bars? Round to 2 decimals.
- gold answer: `16.00`
- predicted answer: `44.0`

## Trajectory

### turn 0 — answer

```text
<think>I will use the formula: <think>;</think>
<tool>(19,22,22,18,18,18)</tool>
<answer>44.0</answer>
<think>The value is 44.0</think>
<source>
```
