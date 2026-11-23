# Publishing Only Totals on Canton: Is It Private? What the Simulations Show

*Independent research for the Canton Community Research Grants (#CantonResearchNov26). Full version, code and every number: github.com/DeniKram/canton-dp-aggregates. Disclosure at the end.*

---

Canton hides each trade from everyone who is not part of it. That is the point of the network. But markets still need public numbers: daily volume, total supply, how many shares a private market has issued. The Dev Fund's own roadmap names the tension in its *Public verifiability* RFP: an aggregate is only useful if it is public, and it is only safe if it does not give away the private data behind it.

The common answer is "we only publish totals". I measured how much a total gives away, and how to fix it.

**Short version:**

- On a thin market, an exact daily total is not private. Two of three dealers who compare notes recover the third dealer's volume on **100% of days**.
- There are two fixes, and they fit different cases. **Differential privacy** (calibrated noise) when contributors are few and each knows its own share. A **threshold** (publish only after k changes) when contributors are many and the reader is an outsider.
- Both have traps that are easy to miss. I measured four of them below.
- On Canton the privacy budget can live in a **Daml contract**, so no operator can quietly overspend it.

## 1. How a total leaks

Take a tokenized bond traded by three dealers, A, B and C. The market publishes one number a day: total volume.

On Tuesday the total is 11.5M. Dealer A traded 5.0M and knows B did not trade that day. So C traded exactly 6.5M. Nobody disclosed anything, yet A now knows C's volume to the dollar.

With two dealers working together, no absence is even needed: total minus their own volumes is the third dealer's volume, every single day.

On a synthetic market (3 dealers, ~20 trades each per day, log-normal trade sizes, 200 random markets per number):

- **Two dealers together** recover the third dealer's daily volume on **100% of days**.
- **One dealer alone**, on a market where dealers trade 60% of days, recovers a competitor's volume exactly on **~49% of days**.

This is not exotic. Statistical agencies hit the same wall, which is why the 2020 US Census moved to differential privacy.

## 2. Fix one: differential privacy, and why "ε = 1" means little on its own

Differential privacy adds random noise sized so that any one participant barely changes what is published. The privacy parameter ε says how much is allowed to leak: smaller is more private and noisier.

The catch: **ε only means something once you say what it protects.** That choice is called the privacy unit.

At ε = 1, daily, three dealers:

- **Protect each trade:** 6.3% median error. Looks great. But a busy dealer's whole day is still guessable within ±10% on **42% of days**. A guarantee for one trade covers a dealer who makes 20 trades only as 20 × ε.
- **Protect each dealer's whole day:** exposure drops to **5.7%**, but the error jumps to **43%**.

Same ε, very different privacy.

[Image: figure1_privacy_vs_accuracy.png]

The good news is that dealer-level privacy gets cheap as the market grows, because the noise is sized to one dealer while the total grows with every dealer. With 20 dealers the same protection costs about **6%** error instead of 43%.

[Image: figure2_market_size.png]

**Trap 1: noise without a budget averages away.** If the same total can be requested again with fresh noise, an attacker just averages. After 100 requests the error falls about 8 times. A privacy budget, which refuses requests once the total ε is spent, closes this.

**Trap 2: noise on every daily snapshot of a running total.** A running total (say, shares outstanding) contains every past grant, so each grant appears in every later snapshot. Fresh noise on each snapshot looks cheap (2.4% error) but spends up to 64 times the budget it claims, and averaging the snapshots recovers **30% of grants**. Mechanisms that count the budget honestly cost more: 3–5% error at the same nominal ε.

## 3. Fix two: a threshold, and why it must sit on the level

The alternative is not to add noise but to wait: publish a value only once enough changes have built up, so that no single change can be read off.

As a real case, I modelled the design of the Canton Asset Data Feed (CADF): a Fairmint and Blockworks proposal, funded by the Canton Dev Fund after a committee vote on Nov 13, that publishes aggregates across many companies' cap tables.

**Trap 3: a rule on the change alone protects nothing.** If the daily *change* is suppressed but the *level* is still published every day, anyone can subtract two levels. With five companies, an outsider recovers **48% of grants** this way.

Put the threshold on the level instead ("the total moves only when five changes are pending") and the picture flips:

- with 20 companies it costs **0.4%** accuracy and an outsider recovers **0%**;
- noise at ε = 5 costs 9% for the same 20 companies.

So for many independent contributors and outside readers, a threshold wins. What it cannot do is protect against someone who already knows the other changes in a batch. Only noise bounds that party.

**Trap 4: count per metric, not in total.** A feed usually publishes several metrics: shares issued, shares outstanding per class, holders. If one shared counter collects every kind of change, a metric touched by only one of them still exposes that change. Four conversions plus one issuance reach k = 5, and the published change in shares issued is exactly that one issuance.

On one synthetic company over a year:

- shared counter, k = 5: **32%** of metric releases expose a single change (72% for rarely-changing preferred stock);
- one counter per metric and per share class: **0%**, by construction.

The cost is delay, mostly on rare metrics: at k = 5 preferred stock would update roughly every 80 days.

[Image: figure3_shared_vs_per_metric.png]

I shared an earlier version of this analysis in the public review of the CADF proposal (PR #890 in the canton-dev-fund repository), and the per-metric numbers there.

## 4. Which tool when

| | Few contributors, each knows its share | Many contributors, outside readers |
|---|---|---|
| Example | dealer volumes on a thin asset | one total across many cap tables |
| Use | differential privacy | threshold on the level |
| Watch out for | the privacy unit; a budget | count per metric; insiders |

## 5. Why Canton is a good place to do this right

Most differential privacy deployments have a weak spot: the privacy budget is a setting inside the operator's own system. Nothing stops the operator from quietly spending more.

On Canton the budget can be a contract.

[Image: figure4_daml_design.png]

- Each contributor sends a `Submission` visible only to itself and the aggregator (Canton's sub-transaction privacy, as is).
- A release is a choice on `AggregationRound` that must also draw on `PrivacyBudget`. If the total ε would pass the cap, the ledger rejects the transaction.
- The contributors sign `PrivacyBudget`, so the aggregator cannot archive it and start a fresh one.
- Daml is deterministic, so noise is generated off-ledger; the aggregator commits to a hash of the noise seed before submissions open.
- A threshold rule fits the same contracts: the release checks the number of pending changes instead of the budget.

What the ledger does not prove: that the aggregator summed honestly and added the noise it declared. That needs an auditor inside the trust boundary, or attestation (TEE or zero-knowledge), the stronger options the RFP already lists. Verifiable and private are two different properties, and a feed needs both.

## 6. Recommendations

1. **Protect the level, not only the change.**
2. **Many contributors, outside readers: threshold on the level**, counted per metric and per share class.
3. **Few contributors who know their share: differential privacy**, and always state the privacy unit next to ε.
4. **Dealer-level privacy needs roughly 20+ participants** before a daily figure stays useful.
5. **Avoid noise on figures that must be exact**, such as supply. Prefer thresholds or delay.
6. **Enforce the privacy budget on-ledger** and publish the mechanism, ε and clipping bounds with every release.

## Limits

All numbers come from synthetic data: real volumes on Canton's private markets are, by design, not public. Rates and sizes are assumptions; a robustness check across seven alternative markets is in the full version: four of the five main conclusions hold in all seven, and the fifth fails in two, as theory predicts. The attacks are lower bounds; the ε guarantee is what holds against all of them.

## Full version and code

The full article, with every table, the robustness checks and the Daml design in detail:

https://forum.canton.network/t/private-markets-public-numbers-thresholds-and-differential-privacy-for-aggregates-on-canton/9551

The code that reproduces every number in about 15 seconds:

github.com/DeniKram/canton-dp-aggregates

Questions and corrections are welcome, especially from teams publishing data on Canton.

---

**Disclosure.** Written for the Canton Community Research Grants. I hold no Canton Coin. I have no commercial relationship with Fairmint, Blockworks or any other team mentioned, and none of them sponsored this work. I am considering a separate open-source proposal to the Canton Development Fund on this topic. The analysis, code and text were prepared with the help of an AI assistant; I checked the results and am responsible for the conclusions.
