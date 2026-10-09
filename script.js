
// ==================================================
// CONFIGURATION
// ==================================================

const API_URL = "https://advanced-rag-document-intelligence-1.onrender.com";


// ==================================================
// CONVERSATION MEMORY
// ==================================================

let conversationHistory = [];
let isAsking = false;
let currentDocumentRequest = 0;


// ==================================================
// ELEMENTS
// ==================================================

const fileInput = document.getElementById("fileInput");
const uploadButton = document.getElementById("uploadButton");
const uploadStatus = document.getElementById("uploadStatus");
const documentList = document.getElementById("documentList");
const documentCount = document.getElementById("documentCount");
const chatMessages = document.getElementById("chatMessages");
const questionInput = document.getElementById("questionInput");
const askButton = document.getElementById("askButton");
const clearChatButton = document.getElementById("clearChatButton");
const newChatButton = document.getElementById("newChatButton");
const themeToggle = document.getElementById("themeToggle");
const themeIcon = document.getElementById("themeIcon");
const themeText = document.getElementById("themeText");

// In-app viewer elements are used for TXT files only.
const documentViewer = document.getElementById("documentViewer");
const documentViewerTitle = document.getElementById("documentViewerTitle");
const documentViewerContent = document.getElementById("documentViewerContent");
const closeDocumentViewerButton = document.getElementById("closeDocumentViewer");


// ==================================================
// INITIALIZE
// ==================================================

initializeTheme();
loadDocuments();

if (closeDocumentViewerButton) {
    closeDocumentViewerButton.addEventListener("click", closeDocumentViewer);
}


// ==================================================
// LOAD DOCUMENTS
// ==================================================

async function loadDocuments() {
    try {
        const response = await fetch(`${API_URL}/documents`);

        if (!response.ok) {
            throw new Error("Could not load documents.");
        }

        const data = await response.json();

        displayDocuments(data.documents);
        updateDocumentCount(data.documents);

    } catch (error) {
        console.error("Document loading error:", error);

        documentList.innerHTML = `
            <p class="empty-documents">
                Could not load documents.
            </p>
        `;

        updateDocumentCount([]);
    }
}


// ==================================================
// DISPLAY DOCUMENTS
// Includes View and Delete buttons.
// ==================================================

function displayDocuments(documents) {
    documentList.innerHTML = "";

    if (!documents || documents.length === 0) {
        documentList.innerHTML = `
            <p class="empty-documents">
                No documents uploaded
            </p>
        `;

        updateDocumentCount([]);
        return;
    }

    documents.forEach(function (filename) {
        const documentItem = document.createElement("div");
        documentItem.className = "document-item";

        const documentName = document.createElement("span");
        documentName.className = "document-name";
        documentName.textContent = filename;

        // ------------------------------
        // VIEW DOCUMENT BUTTON
        // ------------------------------

        const viewButton = document.createElement("button");
        viewButton.className = "view-document-button";
        viewButton.textContent = "View";
        viewButton.title = `View ${filename}`;
        viewButton.setAttribute("aria-label", `View ${filename}`);
        viewButton.type = "button";

        viewButton.style.flexShrink = "0";
        viewButton.style.cursor = "pointer";
        viewButton.style.padding = "5px 9px";
        viewButton.style.marginLeft = "8px";
        viewButton.style.border = "1px solid #7c5cff";
        viewButton.style.borderRadius = "6px";
        viewButton.style.backgroundColor = "#7c5cff";
        viewButton.style.color = "#ffffff";
        viewButton.style.fontSize = "12px";

        viewButton.addEventListener("click", function () {
            viewDocument(filename);
        });

        // ------------------------------
        // DELETE DOCUMENT BUTTON
        // ------------------------------

        const deleteButton = document.createElement("button");
        deleteButton.className = "delete-document-button";
        deleteButton.textContent = "🗑️";
        deleteButton.title = "Delete document";
        deleteButton.setAttribute("aria-label", `Delete ${filename}`);
        deleteButton.type = "button";

        deleteButton.addEventListener("click", function () {
            deleteDocument(filename, deleteButton);
        });

        // ------------------------------
        // ADD ELEMENTS TO DOCUMENT ROW
        // ------------------------------

        documentItem.appendChild(documentName);
        documentItem.appendChild(viewButton);
        documentItem.appendChild(deleteButton);

        documentList.appendChild(documentItem);
    });

    updateDocumentCount(documents);
}


// ==================================================
// UPDATE DOCUMENT COUNT
// ==================================================

function updateDocumentCount(documents) {
    if (!documentCount) {
        return;
    }

    documentCount.textContent = documents ? documents.length : 0;
}


// ==================================================
// VIEW DOCUMENT
//
// PDF: Open in a separate browser tab.
// TXT: Keep using the in-app text viewer.
// ==================================================

function viewDocument(filename) {
    const extension = filename.split(".").pop().toLowerCase();

    const encodedFilename = encodeURIComponent(filename);
    const viewUrl =
        `${API_URL}/documents/${encodedFilename}/view`;

    if (extension === "pdf") {
        // Open the browser's built-in PDF viewer.
        // Call window.open directly from the button click.
        const pdfTab = window.open(
            viewUrl,
            "_blank",
            "noopener,noreferrer"
        );

        if (!pdfTab) {
            alert(
                "The PDF tab may have been blocked. " +
                "Allow pop-ups for this local application and try again."
            );
        }

        return;
    }

    if (extension === "txt") {
        // Keep TXT files inside the existing application viewer.
        viewTextDocument(filename);
        return;
    }

    alert("Only PDF and TXT documents can be viewed.");
}


// ==================================================
// VIEW TXT DOCUMENT INSIDE THE APPLICATION
// ==================================================

async function viewTextDocument(filename) {
    if (
        !documentViewer ||
        !documentViewerTitle ||
        !documentViewerContent ||
        !chatMessages
    ) {
        console.error("TXT viewer elements are missing from index.html.");

        alert("The text viewer is not available. Please check index.html.");
        return;
    }

    const requestId = ++currentDocumentRequest;

    documentViewerTitle.textContent = filename;
    documentViewerContent.replaceChildren();

    const loadingMessage = document.createElement("p");
    loadingMessage.className = "document-viewer-loading";
    loadingMessage.textContent = "Loading document...";

    documentViewerContent.appendChild(loadingMessage);

    // Show the viewer and hide chat while the TXT file is open.
    chatMessages.hidden = true;

    const inputArea = document.querySelector(".input-area");

    if (inputArea) {
        inputArea.hidden = true;
    }

    documentViewer.hidden = false;

    try {
        const response = await fetch(
            `${API_URL}/documents/${encodeURIComponent(filename)}/view`
        );

        if (!response.ok) {
            throw new Error(`Server returned ${response.status}`);
        }

        const text = await response.text();

        // Ignore an old request if another document was selected.
        if (requestId !== currentDocumentRequest) {
            return;
        }

        const pre = document.createElement("pre");

        // textContent prevents the document contents from being
        // interpreted as HTML or executable JavaScript.
        pre.textContent = text;

        documentViewerContent.replaceChildren(pre);

    } catch (error) {
        console.error("TXT preview error:", error);

        if (requestId !== currentDocumentRequest) {
            return;
        }

        showDocumentViewerError(
            "Could not load this text document. " +
            "Check that the backend is running and the file still exists."
        );
    }
}


// ==================================================
// DOCUMENT VIEWER ERROR
// ==================================================

function showDocumentViewerError(message) {
    if (!documentViewerContent) {
        return;
    }

    const errorMessage = document.createElement("p");
    errorMessage.className = "document-viewer-message";
    errorMessage.textContent = message;

    documentViewerContent.replaceChildren(errorMessage);
}


// ==================================================
// CLOSE TXT VIEWER AND RETURN TO CHAT
// ==================================================

function closeDocumentViewer() {
    currentDocumentRequest++;

    if (documentViewer) {
        documentViewer.hidden = true;
    }

    if (documentViewerContent) {
        documentViewerContent.replaceChildren();
    }

    if (chatMessages) {
        chatMessages.hidden = false;
    }

    const inputArea = document.querySelector(".input-area");

    if (inputArea) {
        inputArea.hidden = false;
    }

    if (questionInput && !isAsking) {
        questionInput.focus();
    }
}


// ==================================================
// DELETE DOCUMENT
// ==================================================

async function deleteDocument(filename, deleteButton) {
    const confirmed = window.confirm(
        `Are you sure you want to delete "${filename}"?`
    );

    if (!confirmed) {
        return;
    }

    deleteButton.disabled = true;
    deleteButton.textContent = "Deleting...";

    try {
        const response = await fetch(
            `${API_URL}/documents/${encodeURIComponent(filename)}`,
            { method: "DELETE" }
        );

        if (!response.ok) {
            throw new Error(`Server returned ${response.status}`);
        }

        const data = await response.json();

        if (data.message !== "Document deleted successfully.") {
            throw new Error(data.message || "Delete failed.");
        }

        // Return to chat if the deleted TXT file is currently open.
        if (
            documentViewer &&
            !documentViewer.hidden &&
            documentViewerTitle &&
            documentViewerTitle.textContent === filename
        ) {
            closeDocumentViewer();
        }

        await loadDocuments();

    } catch (error) {
        console.error("Delete error:", error);

        alert(
            "Could not delete the document. " +
            "Please make sure the backend is running."
        );

        deleteButton.disabled = false;
        deleteButton.textContent = "🗑️";
    }
}


// ==================================================
// UPLOAD DOCUMENT
// ==================================================

if (uploadButton) {
    uploadButton.addEventListener("click", uploadDocument);
}

async function uploadDocument() {
    const file = fileInput.files[0];

    if (!file) {
        uploadStatus.textContent = "⚠ Please select a PDF or TXT file.";
        return;
    }

    const filename = file.name.toLowerCase();
    const allowedTypes = [".pdf", ".txt"];

    const isValidType = allowedTypes.some(function (extension) {
        return filename.endsWith(extension);
    });

    if (!isValidType) {
        uploadStatus.textContent =
            "⚠ Unsupported file type. Please select a PDF or TXT file.";

        fileInput.value = "";
        return;
    }

    uploadButton.disabled = true;
    uploadButton.textContent = "Uploading...";
    uploadStatus.textContent = "Uploading document...";

    try {
        const formData = new FormData();
        formData.append("file", file);

        const response = await fetch(`${API_URL}/upload`, {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Server returned ${response.status}`);
        }

        const data = await response.json();

        if (data.message !== "Document stored successfully") {
            uploadStatus.textContent =
                `✕ ${data.message || "Upload failed."}`;
            return;
        }

        uploadStatus.textContent =
            `✓ ${data.filename} uploaded successfully.`;

        fileInput.value = "";

        await loadDocuments();

    } catch (error) {
        console.error("Upload error:", error);

        uploadStatus.textContent =
            "✕ Upload failed. Please make sure the backend is running.";

    } finally {
        uploadButton.disabled = false;
        uploadButton.textContent = "📤 Upload Document";
    }
}


// ==================================================
// ASK QUESTION
// ==================================================

if (askButton) {
    askButton.addEventListener("click", askQuestion);
}

if (questionInput) {
    questionInput.addEventListener("keydown", function (event) {
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            askQuestion();
        }
    });
}

async function askQuestion() {
    if (isAsking) {
        return;
    }

    const question = questionInput.value.trim();

    if (!question) {
        questionInput.focus();
        return;
    }

    isAsking = true;

    const welcomeMessage = chatMessages.querySelector(".welcome-container");

    if (welcomeMessage) {
        welcomeMessage.remove();
    }

    addUserMessage(question);
    questionInput.value = "";

    askButton.disabled = true;
    askButton.textContent = "•••";
    questionInput.disabled = true;

    if (clearChatButton) {
        clearChatButton.disabled = true;
    }

    if (newChatButton) {
        newChatButton.disabled = true;
    }

    const thinkingMessage = addThinkingMessage();

    try {
        const url =
            `${API_URL}/ask?query=${encodeURIComponent(question)}` +
            `&top_k=3` +
            `&conversation_history=${encodeURIComponent(
                JSON.stringify(conversationHistory)
            )}`;

        const response = await fetch(url);

        if (!response.ok) {
            throw new Error(`Server returned ${response.status}`);
        }

        const data = await response.json();

        thinkingMessage.remove();

        if (!data.answer) {
            addAssistantMessage(
                "I could not generate an answer. Please try again.",
                []
            );
            return;
        }

        addAssistantMessage(data.answer, data.sources || []);

        conversationHistory.push({
            role: "user",
            content: question
        });

        conversationHistory.push({
            role: "assistant",
            content: data.answer
        });

        conversationHistory = conversationHistory.slice(-10);

    } catch (error) {
        console.error("Question error:", error);

        thinkingMessage.remove();

        addAssistantMessage(
            "✕ I could not connect to the backend. " +
            "Please make sure the FastAPI server is running.",
            []
        );

    } finally {
        isAsking = false;

        askButton.disabled = false;
        askButton.textContent = "➤";
        questionInput.disabled = false;

        if (clearChatButton) {
            clearChatButton.disabled = false;
        }

        if (newChatButton) {
            newChatButton.disabled = false;
        }

        if (
            (!documentViewer || documentViewer.hidden) &&
            questionInput
        ) {
            questionInput.focus();
        }
    }
}


// ==================================================
// ADD USER MESSAGE
// ==================================================

function addUserMessage(text) {
    const message = document.createElement("div");
    message.className = "message user-message";

    message.innerHTML = `
        <div class="message-label">You</div>

        <div class="message-content">
            ${escapeHtml(text)}
        </div>
    `;

    chatMessages.appendChild(message);
    scrollToBottom();
}


// ==================================================
// ADD THINKING MESSAGE
// ==================================================

function addThinkingMessage() {
    const message = document.createElement("div");
    message.className = "message assistant-message";

    message.innerHTML = `
        <div class="message-label">AI</div>

        <div class="message-content">
            <div class="thinking-container">
                <span class="thinking-text">Thinking</span>

                <div class="thinking-dots">
                    <span class="thinking-dot"></span>
                    <span class="thinking-dot"></span>
                    <span class="thinking-dot"></span>
                </div>
            </div>
        </div>
    `;

    chatMessages.appendChild(message);
    scrollToBottom();

    return message;
}


// ==================================================
// ADD ASSISTANT MESSAGE
// ==================================================

function addAssistantMessage(answer, sources = []) {
    const message = document.createElement("div");
    message.className = "message assistant-message";

    let sourcesHTML = "";

    if (sources && sources.length > 0) {
        const groupedSources = getGroupedSources(sources);

        sourcesHTML = `
            <div class="sources">
                <div class="sources-title">Sources</div>

                ${groupedSources.map(function (group) {
                    return createSourceCard(group);
                }).join("")}
            </div>
        `;
    }

    message.innerHTML = `
        <div class="message-label">AI</div>

        <div class="message-content">
            ${formatAnswer(answer)}
        </div>

        ${sourcesHTML}
    `;

    chatMessages.appendChild(message);
    scrollToBottom();
}


// ==================================================
// GROUP SOURCES BY DOCUMENT
// ==================================================

function getGroupedSources(sources) {
    const groups = new Map();

    sources.forEach(function (source) {
        const filename = source.filename || "Unknown document";

        if (!groups.has(filename)) {
            groups.set(filename, {
                filename: filename,
                references: [],
                seenReferences: new Set()
            });
        }

        const group = groups.get(filename);

        const location =
            source.page !== undefined && source.page !== null
                ? `Page ${source.page}`
                : source.chunk_id !== undefined && source.chunk_id !== null
                    ? `Chunk ${source.chunk_id}`
                    : "Document reference";

        const referenceKey =
            source.chunk_id !== undefined && source.chunk_id !== null
                ? `chunk-${source.chunk_id}`
                : `${location}-${source.text || ""}`;

        if (!group.seenReferences.has(referenceKey)) {
            group.seenReferences.add(referenceKey);

            group.references.push({
                ...source,
                filename: filename,
                displayLocation: location
            });
        }
    });

    return Array.from(groups.values());
}


// ==================================================
// SOURCE CARD
// Display only the source document name.
// ==================================================

function createSourceCard(group) {
    const filename = group.filename || "Unknown document";

    return `
        <div class="source-card">
            <div class="source-file">
                📄 ${escapeHtml(filename)}
            </div>
        </div>
    `;
}


// ==================================================
// FORMAT ANSWER USING MARKDOWN
// ==================================================

function formatAnswer(text) {
    if (!text) {
        return "No answer was returned.";
    }

    if (
        typeof marked === "undefined" ||
        typeof marked.parse !== "function"
    ) {
        return escapeHtml(text).replace(/\n/g, "<br>");
    }

    const html = marked.parse(text, {
        gfm: true,
        breaks: true
    });

    const template = document.createElement("template");
    template.innerHTML = html;

    template.content.querySelectorAll(
        "script, iframe, object, embed, form, input, button, style, link, meta, base"
    ).forEach(function (element) {
        element.remove();
    });

    template.content.querySelectorAll("*").forEach(function (element) {
        Array.from(element.attributes).forEach(function (attribute) {
            const name = attribute.name.toLowerCase();
            const value = attribute.value.trim().toLowerCase();

            if (name.startsWith("on")) {
                element.removeAttribute(attribute.name);
                return;
            }

            if (
                ["href", "src", "xlink:href", "formaction"].includes(name) &&
                (
                    value.startsWith("javascript:") ||
                    value.startsWith("data:") ||
                    value.startsWith("vbscript:")
                )
            ) {
                element.removeAttribute(attribute.name);
            }
        });
    });

    return template.innerHTML;
}


// ==================================================
// CLEAR CHAT AND NEW CHAT
// ==================================================

if (clearChatButton) {
    clearChatButton.addEventListener("click", clearChat);
}

if (newChatButton) {
    newChatButton.addEventListener("click", newChat);
}

function clearChat() {
    resetChat();
}

function newChat() {
    resetChat();
}

function resetChat() {
    if (isAsking) {
        return;
    }

    chatMessages.innerHTML = `
        <div class="welcome-container">
            <div class="welcome-icon">🤖</div>

            <h2>RAG AI Assistant</h2>

            <p>
                Ask questions about your documents
                and get intelligent answers.
            </p>
        </div>
    `;

    questionInput.value = "";
    conversationHistory = [];

    questionInput.focus();
    scrollToBottom();
}


// ==================================================
// THEME
// ==================================================

if (themeToggle) {
    themeToggle.addEventListener("click", toggleTheme);
}

function initializeTheme() {
    const savedTheme = localStorage.getItem("rag-theme");
    const isLight = savedTheme === "light";

    document.body.classList.toggle("light-theme", isLight);
    updateThemeButton(isLight);
}

function toggleTheme() {
    const isLight = document.body.classList.toggle("light-theme");

    localStorage.setItem(
        "rag-theme",
        isLight ? "light" : "dark"
    );

    updateThemeButton(isLight);
}

function updateThemeButton(isLight) {
    if (!themeIcon || !themeText) {
        return;
    }

    if (isLight) {
        themeIcon.textContent = "🌙";
        themeText.textContent = "Dark Mode";
    } else {
        themeIcon.textContent = "☀️";
        themeText.textContent = "Light Mode";
    }
}


// ==================================================
// ESCAPE HTML
// ==================================================

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = String(text ?? "");
    return div.innerHTML;
}


// ==================================================
// SCROLL TO BOTTOM
// ==================================================

function scrollToBottom() {
    if (!chatMessages) {
        return;
    }

    chatMessages.scrollTop = chatMessages.scrollHeight;
}
