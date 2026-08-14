(() => {
  "use strict";
  const data = window.QNEURO_DATA;
  const colors = { architecture: "#73d6cd", training_law: "#d95e4b", halting: "#f1c75b" };
  const labels = { architecture: "architecture", training_law: "training law", halting: "inference law" };
  const formatName = (value) => value.replace(/::/g, " · ").replace(/_/g, " ");
  const pct = (value) => `${(value * 100).toFixed(1)}%`;

  function drawRobustness() {
    const chart = document.querySelector("#robustnessChart");
    const complex = data.robustness.find((item) => item.model === "complex_operator");
    const real = data.robustness.find((item) => item.model === "two_channel_operator");
    ["in_domain", "nuisance", "mild", "moderate", "severe"].forEach((severity) => {
      const group = document.createElement("div");
      group.className = "shift-group";
      group.innerHTML = `<div class="bar-area">
        <div class="bar complex" style="height:${complex[severity] * 100}%"><span>${complex[severity].toFixed(3)}</span></div>
        <div class="bar real" style="height:${real[severity] * 100}%"><span>${real[severity].toFixed(3)}</span></div>
      </div><span>${formatName(severity)}</span>`;
      chart.appendChild(group);
    });
  }

  function candidateCaveat(candidate) {
    if (candidate.context === "halting") return "Inference timing only; do not compare circle area with training-law timing.";
    if (candidate.candidate_id.includes("complex")) return "Synthetic robustness is the strength; ambiguity and calibration remain the boundary.";
    if (candidate.candidate_id === "gru") return "Strong source fit and readable factors coexist with a large unseen-world collapse.";
    return `Pareto status: ${candidate.pareto ? "non-dominated under declared objectives" : "dominated in this context"}.`;
  }

  function setReadout(candidate) {
    document.querySelector("#candidateName").textContent = formatName(candidate.candidate_id);
    const entries = [
      ["Context", labels[candidate.context]], ["Source top-1", pct(candidate.in_domain_top1)],
      ["Shifted top-1", pct(candidate.shifted_top1)], ["Chronology pairs", pct(candidate.counterfactual_pair_accuracy)],
      ["Shifted ECE", candidate.shifted_ece.toFixed(3)], ["CPU measure", `${candidate.training_seconds.toFixed(3)} s`],
    ];
    document.querySelector("#candidateMetrics").innerHTML = entries.map(([key, value]) => `<div><dt>${key}</dt><dd>${value}</dd></div>`).join("");
    document.querySelector("#candidateCaveat").textContent = candidateCaveat(candidate);
  }

  function drawField(context) {
    const svg = document.querySelector("#evidenceField");
    const records = data.candidates.filter((item) => item.context === context);
    const width = 760, height = 470, left = 66, right = 24, top = 26, bottom = 58;
    const x = (value) => left + value * (width - left - right);
    const y = (value) => height - bottom - value * (height - top - bottom);
    const maxTime = Math.max(...records.map((item) => item.training_seconds));
    const radius = (value) => 5 + 12 * Math.sqrt(value / Math.max(maxTime, .001));
    let markup = "";
    for (let tick = 0; tick <= 10; tick += 2) {
      const value = tick / 10;
      markup += `<line class="grid" x1="${x(value)}" y1="${top}" x2="${x(value)}" y2="${height - bottom}" />`;
      markup += `<line class="grid" x1="${left}" y1="${y(value)}" x2="${width - right}" y2="${y(value)}" />`;
      markup += `<text class="tick" x="${x(value)}" y="${height - bottom + 23}" text-anchor="middle">${value.toFixed(1)}</text>`;
      markup += `<text class="tick" x="${left - 13}" y="${y(value) + 4}" text-anchor="end">${value.toFixed(1)}</text>`;
    }
    markup += `<text class="axis-label" x="${(left + width - right) / 2}" y="${height - 8}" text-anchor="middle">unseen-world top-1 →</text>`;
    markup += `<text class="axis-label" transform="translate(16 ${(top + height - bottom) / 2}) rotate(-90)" text-anchor="middle">source top-1 →</text>`;
    records.forEach((record, index) => {
      markup += `<circle tabindex="0" role="button" aria-label="${formatName(record.candidate_id)}" class="candidate ${record.pareto ? "pareto" : ""}" data-index="${index}" cx="${x(record.shifted_top1)}" cy="${y(record.in_domain_top1)}" r="${radius(record.training_seconds)}" fill="${colors[context]}" fill-opacity="${record.pareto ? ".95" : ".38"}" />`;
    });
    svg.innerHTML = markup;
    const tooltip = document.querySelector("#fieldTooltip");
    svg.querySelectorAll(".candidate").forEach((node) => {
      const select = () => setReadout(records[Number(node.dataset.index)]);
      node.addEventListener("focus", select); node.addEventListener("click", select);
      node.addEventListener("mouseenter", (event) => {
        const record = records[Number(node.dataset.index)];
        tooltip.hidden = false;
        tooltip.textContent = `${formatName(record.candidate_id)} · ${pct(record.shifted_top1)} shifted`;
        tooltip.style.left = `${event.offsetX + 12}px`; tooltip.style.top = `${event.offsetY + 12}px`;
      });
      node.addEventListener("mouseleave", () => { tooltip.hidden = true; });
    });
    setReadout(records.find((item) => item.candidate_id === "complex_operator") || records.find((item) => item.pareto) || records[0]);
  }

  function buildClaims() {
    const filter = document.querySelector("#claimFilter");
    const list = document.querySelector("#claimList");
    const groups = ["All", "Replicated", "Preliminary", "Refuted", "Unsupported"];
    const render = (group) => {
      const claims = data.claims.filter((claim) => group === "All" || claim.status.toLowerCase().includes(group.toLowerCase()));
      list.innerHTML = claims.map((claim, index) => `<article class="claim-card"><button aria-expanded="false" aria-controls="claim-${index}"><span class="claim-status">${claim.status}</span><h3>${claim.claim}</h3><span class="toggle">+</span></button><div class="claim-detail" id="claim-${index}" hidden><p><strong>Evidence:</strong> ${claim.evidence}</p><p><strong>Boundary:</strong> ${claim.counterevidence}</p></div></article>`).join("") || "<p>No claims in this filter.</p>";
      list.querySelectorAll("button").forEach((button) => button.addEventListener("click", () => {
        const open = button.getAttribute("aria-expanded") === "true";
        button.setAttribute("aria-expanded", String(!open)); button.querySelector(".toggle").textContent = open ? "+" : "−";
        document.getElementById(button.getAttribute("aria-controls")).hidden = open;
      }));
    };
    groups.forEach((group) => {
      const button = document.createElement("button"); button.textContent = group;
      if (group === "All") button.classList.add("active");
      button.addEventListener("click", () => {
        filter.querySelectorAll("button").forEach((item) => item.classList.remove("active"));
        button.classList.add("active"); render(group);
      });
      filter.appendChild(button);
    });
    render("All");
  }

  document.querySelector("#failureList").innerHTML = data.failures.slice(-8).map((failure) => `<li>${failure}</li>`).join("");
  document.querySelector("#proposalList").innerHTML = data.proposals.map((proposal) => `<article class="proposal"><span class="priority">${proposal.priority} priority</span><h3>${formatName(proposal.id)}</h3><p>${proposal.mechanism}</p><p class="falsifier"><strong>Stop if:</strong> ${proposal.falsifier}</p></article>`).join("");
  document.querySelectorAll(".field-controls button").forEach((button) => button.addEventListener("click", () => {
    document.querySelectorAll(".field-controls button").forEach((item) => item.classList.remove("active"));
    button.classList.add("active"); drawField(button.dataset.context);
  }));
  document.querySelector("#provenanceNote").textContent = `${data.experiments.length} registered result directories; ${data.candidates.length} normalized candidates; ${data.surprises.length} audit flags. Cached artifacts reproduce the public evidence layer.`;
  drawRobustness(); drawField("architecture"); buildClaims();
})();
