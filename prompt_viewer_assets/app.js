const state = {
  search: "",
  model: "",
  depth: "",
  agreementCondition: "",
  status: "",
  offset: 0,
  limit: 150,
  records: [],
  selectedId: null,
  totalFiltered: 0,
};

const elements = {
  search: document.querySelector("#search"),
  model: document.querySelector("#model"),
  depth: document.querySelector("#depth"),
  agreementCondition: document.querySelector("#agreement-condition"),
  status: document.querySelector("#status"),
  stats: document.querySelector("#stats"),
  recordList: document.querySelector("#record-list"),
  loadMore: document.querySelector("#load-more"),
  emptyState: document.querySelector("#empty-state"),
  detail: document.querySelector("#detail"),
  detailTitle: document.querySelector("#detail-title"),
  detailSubtitle: document.querySelector("#detail-subtitle"),
  detailStatus: document.querySelector("#detail-status"),
  summaryMeta: document.querySelector("#summary-meta"),
  promptText: document.querySelector("#prompt-text"),
  grammarText: document.querySelector("#grammar-text"),
  responseText: document.querySelector("#response-text"),
  inputJson: document.querySelector("#input-json"),
  outputJson: document.querySelector("#output-json"),
};

function debounce(fn, wait) {
  let timeoutId = null;
  return (...args) => {
    window.clearTimeout(timeoutId);
    timeoutId = window.setTimeout(() => fn(...args), wait);
  };
}

function recordSubtitle(record) {
  return [
    record.fuzzy_model || "unknown model",
    record.grammar_name || "unknown grammar",
    record.depth === null ? null : `depth ${record.depth}`,
    record.agreement_condition_label || null,
  ]
    .filter(Boolean)
    .join(" | ");
}

function statusClass(status) {
  return `status-${status || "missing"}`;
}

function renderStats(batchDir, totalRecords, filteredRecords) {
  elements.stats.innerHTML = `
    <div class="stat">
      <span class="stat-label">Batch dir</span>
      <span class="stat-value">${batchDir}</span>
    </div>
    <div class="stat">
      <span class="stat-label">Loaded</span>
      <span class="stat-value">${filteredRecords} / ${totalRecords}</span>
    </div>
  `;
}

function renderList() {
  elements.recordList.innerHTML = "";
  if (!state.records.length) {
    const empty = document.createElement("div");
    empty.className = "list-empty";
    empty.textContent = "No records match the current filters.";
    elements.recordList.append(empty);
    return;
  }

  for (const record of state.records) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "record-item";
    if (record.custom_id === state.selectedId) {
      button.classList.add("selected");
    }
    button.innerHTML = `
      <div class="record-topline">
        <span class="pill ${statusClass(record.status)}">${record.status}</span>
        <span class="record-id">${record.custom_id}</span>
      </div>
      <p class="record-input">${record.input_sentence || "No extracted input sentence"}</p>
      <p class="record-answer">${record.final_answer || record.error_message || "No response text extracted"}</p>
      <p class="record-meta">${recordSubtitle(record)}</p>
    `;
    button.addEventListener("click", () => selectRecord(record.custom_id));
    elements.recordList.append(button);
  }
}

function renderMetaRows(record) {
  const items = [
    ["Requested model", record.model],
    ["Response model", record.response_model],
    ["Grammar", record.grammar_name],
    ["Sample id", record.sample_id],
    ["Depth", record.depth],
    ["Agreement condition", record.agreement_condition_label],
    ["Input sentence", record.input_sentence],
    ["Final answer", record.final_answer],
    ["Status code", record.status_code],
    ["Prompt tokens", record.prompt_tokens],
    ["Completion tokens", record.completion_tokens],
    ["Total tokens", record.total_tokens],
    ["Input file", record.input_file],
    ["Output file", record.output_file],
    ["Request id", record.request_id],
  ];

  elements.summaryMeta.innerHTML = items
    .map(
      ([label, value]) => `
        <dt>${label}</dt>
        <dd>${value ?? "-"}</dd>
      `
    )
    .join("");
}

async function fetchRecords({ append = false } = {}) {
  const params = new URLSearchParams({
    search: state.search,
    model: state.model,
    depth: state.depth,
    agreement_condition: state.agreementCondition,
    status: state.status,
    offset: String(state.offset),
    limit: String(state.limit),
  });
  const response = await fetch(`/api/records?${params.toString()}`);
  const payload = await response.json();

  renderStats(payload.batch_dir, payload.total_records, payload.filtered_records);
  populateSelect(elements.model, payload.models, state.model, "All models");
  populateSelect(elements.depth, payload.depths, state.depth, "All depths");
  populateAgreementSelect(
    elements.agreementCondition,
    payload.agreement_conditions,
    state.agreementCondition
  );

  state.totalFiltered = payload.filtered_records;
  state.records = append ? state.records.concat(payload.records) : payload.records;
  renderList();

  elements.loadMore.disabled = state.records.length >= state.totalFiltered;
  if (!state.selectedId && state.records.length) {
    selectRecord(state.records[0].custom_id);
  }
}

function populateSelect(element, options, currentValue, defaultLabel) {
  const currentOptions = [""].concat(options || []);
  const previousValue = currentValue;
  element.innerHTML = currentOptions
    .map((option) => {
      const label = option || defaultLabel;
      const selected = option === previousValue ? "selected" : "";
      return `<option value="${option}" ${selected}>${label}</option>`;
    })
    .join("");
}

function populateAgreementSelect(element, options, currentValue) {
  const currentOptions = [{ value: "", label: "All conditions" }].concat(
    options || []
  );
  element.innerHTML = currentOptions
    .map((option) => {
      const selected = option.value === currentValue ? "selected" : "";
      return `<option value="${option.value}" ${selected}>${option.label}</option>`;
    })
    .join("");
}

async function selectRecord(customId) {
  state.selectedId = customId;
  renderList();
  const response = await fetch(`/api/record?id=${encodeURIComponent(customId)}`);
  const record = await response.json();

  elements.emptyState.hidden = true;
  elements.detail.hidden = false;
  elements.detailTitle.textContent = record.input_sentence || record.custom_id;
  elements.detailSubtitle.textContent = [
    record.custom_id,
    record.fuzzy_model || "unknown model",
    record.grammar_name || "unknown grammar",
  ]
    .filter(Boolean)
    .join(" | ");
  elements.detailStatus.className = `pill ${statusClass(record.status)}`;
  elements.detailStatus.textContent = record.status;
  renderMetaRows(record);
  elements.promptText.textContent = record.prompt || "";
  elements.grammarText.textContent = record.grammar || "No grammar block extracted.";
  elements.responseText.textContent =
    record.response_text || record.error_message || "No response text extracted.";
  elements.inputJson.textContent = JSON.stringify(record.input_payload, null, 2);
  elements.outputJson.textContent = JSON.stringify(record.output_payload, null, 2);
}

function resetAndFetch() {
  state.offset = 0;
  state.records = [];
  state.selectedId = null;
  fetchRecords({ append: false });
}

elements.search.addEventListener(
  "input",
  debounce((event) => {
    state.search = event.target.value;
    resetAndFetch();
  }, 200)
);
elements.model.addEventListener("change", (event) => {
  state.model = event.target.value;
  resetAndFetch();
});
elements.depth.addEventListener("change", (event) => {
  state.depth = event.target.value;
  resetAndFetch();
});
elements.agreementCondition.addEventListener("change", (event) => {
  state.agreementCondition = event.target.value;
  resetAndFetch();
});
elements.status.addEventListener("change", (event) => {
  state.status = event.target.value;
  resetAndFetch();
});
elements.loadMore.addEventListener("click", () => {
  state.offset = state.records.length;
  fetchRecords({ append: true });
});

fetchRecords().catch((error) => {
  elements.recordList.innerHTML = `<div class="list-empty">${error.message}</div>`;
});
