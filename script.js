(() => {
  const state = {
    lastRetrievedChunks: [],
    selectedFiles: [],
  };

  const els = {
    healthBadge: document.getElementById("healthBadge"),
    docCountValue: document.getElementById("docCountValue"),
    chunkCountValue: document.getElementById("chunkCountValue"),
    docList: document.getElementById("docList"),
    refreshDocsBtn: document.getElementById("refreshDocsBtn"),
    uploadForm: document.getElementById("uploadForm"),
    fileInput: document.getElementById("fileInput"),
    dropzone: document.getElementById("dropzone"),
    selectedFilesLabel: document.getElementById("selectedFilesLabel"),
    uploadBtn: document.getElementById("uploadBtn"),
    reindexBtn: document.getElementById("reindexBtn"),
    uploadStatus: document.getElementById("uploadStatus"),
    tabs: document.querySelectorAll(".tab"),
    tabPanels: document.querySelectorAll(".tab-panel"),
    topKSelect: document.getElementById("topKSelect"),
    chatMessages: document.getElementById("chatMessages"),
    chatError: document.getElementById("chatError"),
    chatForm: document.getElementById("chatForm"),
    chatInput: document.getElementById("chatInput"),
    sendBtn: document.getElementById("sendBtn"),
    retrieveForm: document.getElementById("retrieveForm"),
    retrieveInput: document.getElementById("retrieveInput"),
    retrieveBtn: document.getElementById("retrieveBtn"),
    sampleChunksBtn: document.getElementById("sampleChunksBtn"),
    inspectorMeta: document.getElementById("inspectorMeta"),
    chunkResults: document.getElementById("chunkResults"),
    compareForm: document.getElementById("compareForm"),
    compareInput: document.getElementById("compareInput"),
    compareBtn: document.getElementById("compareBtn"),
    compareError: document.getElementById("compareError"),
    plainAnswerBody: document.getElementById("plainAnswerBody"),
    ragAnswerBody: document.getElementById("ragAnswerBody"),
    compareSources: document.getElementById("compareSources"),
    compareChunksSection: document.getElementById("compareChunksSection"),
    plainTimingBadge: document.getElementById("plainTimingBadge"),
    ragTimingBadge: document.getElementById("ragTimingBadge"),
  };

  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text == null ? "" : String(text);
    return div.innerHTML;
  }

  function showError(container, message) {
    container.textContent = message;
    container.classList.remove("hidden");
  }

  function hideError(container) {
    container.textContent = "";
    container.classList.add("hidden");
  }

  function switchTab(tabName) {
    els.tabs.forEach((btn) => {
      const active = btn.dataset.tab === tabName;
      btn.classList.toggle("active", active);
      btn.setAttribute("aria-selected", active ? "true" : "false");
    });
    els.tabPanels.forEach((panel) => {
      panel.classList.toggle("active", panel.id === `tab-${tabName}`);
    });
  }

  els.tabs.forEach((btn) => {
    btn.addEventListener("click", () => switchTab(btn.dataset.tab));
  });

  async function fetchJson(url, options = {}) {
    const response = await fetch(url, options);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(data.error || `Request failed with status ${response.status}`);
    }
    return data;
  }

  async function refreshHealthAndDocuments() {
    try {
      const [health, docs] = await Promise.all([
        fetchJson("/api/health"),
        fetchJson("/api/documents"),
      ]);

      const ok = health.status === "ok";
      els.healthBadge.textContent = ok ? "Online" : "Offline";
      els.healthBadge.className = `status-pill ${ok ? "ok" : "warn"}`;
      els.healthBadge.title = ok ? "System online and ready." : (health.status_message || "API key configuration required.");

      els.docCountValue.textContent = docs.total_documents ?? 0;
      els.chunkCountValue.textContent = docs.total_chunks ?? 0;

      renderDocumentsList(docs.documents || []);
    } catch (err) {
      els.healthBadge.textContent = "Offline";
      els.healthBadge.className = "status-pill warn";
      els.healthBadge.title = err.message;
      els.docList.innerHTML = `<p class="empty-state">${escapeHtml(err.message)}</p>`;
    }
  }

  function renderDocumentsList(documents) {
    if (!documents.length) {
      els.docList.innerHTML = `<p class="empty-state">No documents indexed yet. Upload PDF, MD, or TXT files above.</p>`;
      return;
    }

    els.docList.innerHTML = documents
      .map((doc) => {
        const pageMeta = doc.page_count ? ` • ${doc.page_count} pages` : "";
        return `
          <div class="doc-item">
            <div class="doc-item-top">
              <span class="doc-name" title="${escapeHtml(doc.source)}">${escapeHtml(doc.source)}</span>
              <span class="doc-type">${escapeHtml(doc.file_type)}</span>
            </div>
            <div class="doc-stats">${doc.chunk_count} chunks${pageMeta} • ~${doc.total_tokens} tokens</div>
          </div>
        `;
      })
      .join("");
  }

  // File Selection & Drag/Drop
  function updateSelectedFiles(files) {
    state.selectedFiles = Array.from(files || []);
    if (!state.selectedFiles.length) {
      els.selectedFilesLabel.textContent = "No files selected";
      return;
    }
    els.selectedFilesLabel.textContent = state.selectedFiles.map((f) => f.name).join(", ");
  }

  els.fileInput.addEventListener("change", (e) => updateSelectedFiles(e.target.files));

  ["dragenter", "dragover"].forEach((evtName) => {
    els.dropzone.addEventListener(evtName, (e) => {
      e.preventDefault();
      els.dropzone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((evtName) => {
    els.dropzone.addEventListener(evtName, (e) => {
      e.preventDefault();
      els.dropzone.classList.remove("dragover");
    });
  });

  els.dropzone.addEventListener("drop", (e) => {
    const files = e.dataTransfer?.files;
    if (files?.length) {
      els.fileInput.files = files;
      updateSelectedFiles(files);
    }
  });

  function setUploadStatus(message, type) {
    els.uploadStatus.textContent = message;
    els.uploadStatus.className = `upload-status ${type}`;
  }

  els.uploadForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!state.selectedFiles.length) {
      setUploadStatus("Please select at least one .pdf, .md, or .txt file.", "error");
      return;
    }

    const formData = new FormData();
    state.selectedFiles.forEach((file) => formData.append("file", file));

    els.uploadBtn.disabled = true;
    setUploadStatus("Uploading, chunking, and embedding...", "success");

    try {
      const res = await fetchJson("/api/upload", { method: "POST", body: formData });
      const ing = res.ingestion || {};
      const ingestedCount = (ing.files_ingested || []).length;
      const skippedCount = (ing.files_skipped || []).length;
      setUploadStatus(
        `Done: ${ingestedCount} ingested (${ing.chunks_added || 0} chunks added), ${skippedCount} skipped.`,
        "success"
      );
      els.fileInput.value = "";
      updateSelectedFiles([]);
      await refreshHealthAndDocuments();
    } catch (err) {
      setUploadStatus(err.message, "error");
    } finally {
      els.uploadBtn.disabled = false;
    }
  });

  els.reindexBtn.addEventListener("click", async () => {
    els.reindexBtn.disabled = true;
    setUploadStatus("Scanning data/documents/ for unindexed files...", "success");
    try {
      const res = await fetchJson("/api/ingest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ force: false }),
      });
      const ing = res.ingestion || {};
      setUploadStatus(
        `Scan complete: ${(ing.files_ingested || []).length} ingested, ${(ing.files_skipped || []).length} skipped (${ing.chunks_added || 0} chunks added).`,
        "success"
      );
      await refreshHealthAndDocuments();
    } catch (err) {
      setUploadStatus(err.message, "error");
    } finally {
      els.reindexBtn.disabled = false;
    }
  });

  els.refreshDocsBtn.addEventListener("click", refreshHealthAndDocuments);

  // Chat Handling
  function appendMessage(role, text, citations = [], retrievedChunks = [], timing = null) {
    const wrapper = document.createElement("div");
    wrapper.className = `message ${role}`;

    const roleLabel = document.createElement("div");
    roleLabel.className = "message-role";
    roleLabel.textContent = role === "user" ? "You" : "Valentius Assistant";
    wrapper.appendChild(roleLabel);

    const body = document.createElement("div");
    body.className = "message-content";
    body.textContent = text;
    wrapper.appendChild(body);

    if (role === "assistant" && citations.length) {
      const citationsBox = document.createElement("div");
      citationsBox.className = "citations-box";

      const title = document.createElement("div");
      title.className = "citations-title";
      title.textContent = "Source Citations (click to inspect retrieved chunks):";
      citationsBox.appendChild(title);

      const pills = document.createElement("div");
      pills.className = "citation-pills";

      citations.forEach((cit) => {
        const pill = document.createElement("button");
        pill.type = "button";
        pill.className = "citation-pill";
        const pageStr = cit.page != null ? ` · p.${cit.page}` : "";
        pill.textContent = `${cit.source}${pageStr} (${Math.round((cit.relevance_score || 0) * 100)}%)`;
        pill.title = cit.excerpt || "";
        pill.addEventListener("click", () => {
          state.lastRetrievedChunks = retrievedChunks;
          renderChunks(retrievedChunks, `Showing ${retrievedChunks.length} retrieved chunks from your latest question`);
          switchTab("inspector");
        });
        pills.appendChild(pill);
      });

      citationsBox.appendChild(pills);
      wrapper.appendChild(citationsBox);
    }

    if (role === "assistant" && timing && timing.total_time_ms != null) {
      const timingBox = document.createElement("div");
      timingBox.className = "timing-badge mono";
      timingBox.innerHTML = `
        <span class="highlight">⚡ Total: ${Math.round(timing.total_time_ms)}ms</span>
        <span>·</span>
        <span>Embed: ${Math.round(timing.embed_time_ms || 0)}ms</span>
        <span>·</span>
        <span>Retrieve: ${Math.round(timing.retrieval_time_ms || 0)}ms</span>
        <span>·</span>
        <span class="highlight">Groq: ${Math.round(timing.generation_time_ms || 0)}ms</span>
      `;
      wrapper.appendChild(timingBox);
    }

    els.chatMessages.appendChild(wrapper);
    els.chatMessages.scrollTop = els.chatMessages.scrollHeight;
    return wrapper;
  }

  function appendLoadingBubble() {
    const wrapper = document.createElement("div");
    wrapper.className = "message assistant";
    wrapper.innerHTML = `
      <div class="message-role">Valentius Assistant</div>
      <div class="loading-dots">
        <span></span><span></span><span></span>
        Retrieving context &amp; generating answer...
      </div>
    `;
    els.chatMessages.appendChild(wrapper);
    els.chatMessages.scrollTop = els.chatMessages.scrollHeight;
    return wrapper;
  }

  async function submitChatQuestion(question) {
    hideError(els.chatError);
    appendMessage("user", question);
    const loadingEl = appendLoadingBubble();
    els.sendBtn.disabled = true;

    try {
      const topK = Number(els.topKSelect.value || 5);
      const data = await fetchJson("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, top_k: topK }),
      });

      loadingEl.remove();
      state.lastRetrievedChunks = data.retrieved_chunks || [];
      appendMessage(
        "assistant",
        data.answer,
        data.citations || [],
        state.lastRetrievedChunks,
        data.timing
      );
      if (state.lastRetrievedChunks.length) {
        renderChunks(
          state.lastRetrievedChunks,
          `Retrieved ${state.lastRetrievedChunks.length} chunks for: "${question}"`
        );
      }
    } catch (err) {
      loadingEl.remove();
      showError(els.chatError, err.message);
    } finally {
      els.sendBtn.disabled = false;
    }
  }

  els.chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const question = els.chatInput.value.trim();
    if (!question) return;
    els.chatInput.value = "";
    await submitChatQuestion(question);
  });

  els.chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      els.chatForm.dispatchEvent(new Event("submit"));
    }
  });

  document.querySelectorAll(".chip[data-prompt]").forEach((chip) => {
    chip.addEventListener("click", () => {
      const prompt = chip.getAttribute("data-prompt");
      if (prompt) {
        submitChatQuestion(prompt);
      }
    });
  });

  // Chunk Inspector
  function renderChunks(chunks, summaryText, container = els.chunkResults) {
    if (container === els.chunkResults && summaryText) {
      els.inspectorMeta.textContent = summaryText;
    }
    if (!chunks.length) {
      container.innerHTML = `<p class="empty-state">No chunks found.</p>`;
      return;
    }

    container.innerHTML = chunks
      .map((chunk, idx) => {
        const pageBadge = chunk.page != null ? `<span class="badge badge-cyan">Page ${chunk.page}</span>` : "";
        const relBadge =
          chunk.relevance_score != null
            ? `<span class="badge badge-violet">Relevance: ${Number(chunk.relevance_score).toFixed(4)}</span>`
            : "";
        const distBadge =
          chunk.distance != null
            ? `<span class="badge badge-neutral">Dist: ${Number(chunk.distance).toFixed(4)}</span>`
            : "";

        return `
          <article class="chunk-card">
            <div class="chunk-card-header">
              <span class="chunk-source">#${idx + 1} · ${escapeHtml(chunk.source)}</span>
              <div class="chunk-badges">
                ${pageBadge}
                ${relBadge}
                ${distBadge}
                <span class="badge badge-neutral mono">chunk ${chunk.chunk_index} · ${chunk.token_count} tok</span>
              </div>
            </div>
            <pre class="chunk-text">${escapeHtml(chunk.text)}</pre>
          </article>
        `;
      })
      .join("");
  }

  els.retrieveForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const query = els.retrieveInput.value.trim();
    if (!query) return;

    els.retrieveBtn.disabled = true;
    els.inspectorMeta.textContent = "Running semantic search in ChromaDB...";
    try {
      const topK = Number(els.topKSelect.value || 5);
      const data = await fetchJson("/api/retrieve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, top_k: topK }),
      });
      renderChunks(
        data.chunks || [],
        `Semantic retrieval returned ${data.count} chunk(s) for "${data.query}" (top_k=${data.top_k}).`
      );
    } catch (err) {
      els.inspectorMeta.textContent = `Error: ${err.message}`;
      els.chunkResults.innerHTML = "";
    } finally {
      els.retrieveBtn.disabled = false;
    }
  });

  els.sampleChunksBtn.addEventListener("click", async () => {
    els.sampleChunksBtn.disabled = true;
    try {
      const data = await fetchJson("/api/chunks/sample?limit=5");
      renderChunks(
        data.samples || [],
        `Inspecting ${data.count} stored sample chunks from ChromaDB.`
      );
    } catch (err) {
      els.inspectorMeta.textContent = `Error: ${err.message}`;
    } finally {
      els.sampleChunksBtn.disabled = false;
    }
  });

  // RAG vs Plain LLM Comparison
  els.compareForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const question = els.compareInput.value.trim();
    if (!question) return;

    hideError(els.compareError);
    els.compareBtn.disabled = true;
    if (els.plainTimingBadge) {
      els.plainTimingBadge.textContent = "";
      els.plainTimingBadge.classList.add("hidden");
    }
    if (els.ragTimingBadge) {
      els.ragTimingBadge.textContent = "";
      els.ragTimingBadge.classList.add("hidden");
    }
    els.plainAnswerBody.innerHTML = `<div class="loading-dots"><span></span><span></span><span></span> Generating plain LLM answer...</div>`;
    els.ragAnswerBody.innerHTML = `<div class="loading-dots"><span></span><span></span><span></span> Retrieving context &amp; generating RAG answer...</div>`;
    els.compareSources.innerHTML = "";
    els.compareChunksSection.innerHTML = "";

    try {
      const topK = Number(els.topKSelect.value || 5);
      const data = await fetchJson("/api/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, top_k: topK }),
      });

      els.plainAnswerBody.textContent = data.plain_llm_answer || "";
      els.ragAnswerBody.textContent = data.rag_answer || "";

      if (els.plainTimingBadge && data.timing?.plain_llm) {
        const pGen = Math.round(data.timing.plain_llm.generation_time_ms || 0);
        els.plainTimingBadge.innerHTML = `<span class="highlight">⚡ Groq: ${pGen}ms</span>`;
        els.plainTimingBadge.classList.remove("hidden");
      }

      if (els.ragTimingBadge && data.timing?.rag) {
        const rTiming = data.timing.rag;
        els.ragTimingBadge.innerHTML = `
          <span class="highlight">⚡ Total: ${Math.round(rTiming.total_time_ms || 0)}ms</span>
          <span>·</span>
          <span>Embed: ${Math.round(rTiming.embed_time_ms || 0)}ms</span>
          <span>·</span>
          <span>Retrieve: ${Math.round(rTiming.retrieval_time_ms || 0)}ms</span>
          <span>·</span>
          <span class="highlight">Groq: ${Math.round(rTiming.generation_time_ms || 0)}ms</span>
        `;
        els.ragTimingBadge.classList.remove("hidden");
      }

      const sources = data.source_documents || [];
      els.compareSources.textContent = sources.length
        ? `Retrieved Sources: ${sources.join(", ")}`
        : "No source documents retrieved.";

      if ((data.retrieved_chunks || []).length) {
        const header = document.createElement("h3");
        header.textContent = `Retrieved Evidence Chunks (${data.retrieved_chunks.length})`;
        header.style.margin = "8px 0 4px";
        els.compareChunksSection.appendChild(header);

        const list = document.createElement("div");
        list.className = "chunk-results";
        renderChunks(data.retrieved_chunks, null, list);
        els.compareChunksSection.appendChild(list);
      }
    } catch (err) {
      showError(els.compareError, err.message);
      els.plainAnswerBody.innerHTML = `<p class="empty-state">Comparison failed.</p>`;
      els.ragAnswerBody.innerHTML = `<p class="empty-state">Comparison failed.</p>`;
    } finally {
      els.compareBtn.disabled = false;
    }
  });

  refreshHealthAndDocuments();
})();

