class TranslationApp {
    constructor() {
        this.initializeElements();
        this.bindEvents();
        this.loadConfig();
        this.populateChapterDropdown();
        this.updateModelOptions(); // Initialize model options
        this.loadTranslationHistory(); // Load history on startup
    }

    initializeElements() {
        this.llmProvider = document.getElementById('llm-provider');
        this.modelSelect = document.getElementById('model-select');
        this.apiToken = document.getElementById('api-token');
        this.inputText = document.getElementById('input-text');
        this.outputText = document.getElementById('output-text');
        this.translateBtn = document.getElementById('translate-btn');
        this.copyBtn = document.getElementById('copy-btn');
        this.loading = document.getElementById('loading');
        this.errorMessage = document.getElementById('error-message');
        this.useContextCheckbox = document.getElementById('use-context-checkbox');
        this.chapterSelection = document.getElementById('chapter-selection');
        this.chapterNumber = document.getElementById('chapter-number');
        this.infoMessage = document.getElementById('info-message');

        // History elements
        this.historyList = document.getElementById('history-list');
        this.clearHistoryBtn = document.getElementById('clear-history-btn');

        // Output section elements
        this.outputSection = document.querySelector('.output-section');

        
        // Model options for each provider
        this.modelOptions = {
            'qwen': [
                { value: 'qwen-turbo', text: 'qwen-turbo' },
                { value: 'qwen-plus', text: 'qwen-plus' },
                { value: 'qwen-max', text: 'qwen-max' }
            ],
            'deepseek': [
                { value: 'deepseek-chat', text: 'deepseek-chat' }
            ],
            'chatgpt': [
                { value: 'gpt-4o-mini', text: 'gpt-4o-mini' },
                { value: 'gpt-4o', text: 'gpt-4o' }
            ],
            'chatgpt(azure)': [
                { value: 'gpt-4o-mini', text: 'gpt-4o-mini' },
                { value: 'gpt-4o', text: 'gpt-4o' }
            ]
        };
    }

    bindEvents() {
        this.translateBtn.addEventListener('click', () => this.translate());
        this.copyBtn.addEventListener('click', () => this.copyToClipboard());
        
        // Ctrl+Enter in input area
        this.inputText.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'Enter') {
                this.translate();
            }
        });

        // Update model options and API token when provider changes
        this.llmProvider.addEventListener('change', () => {
            this.updateModelOptions();
            this.updateApiToken();
        });
        
        // Context toggle event
        this.useContextCheckbox.addEventListener('change', () => this.toggleContextControls());

        // History event handlers
        this.clearHistoryBtn.addEventListener('click', () => this.clearTranslationHistory());
    }

    async populateChapterDropdown() {
        try {
            const response = await fetch('/chapters');
            if (!response.ok) {
                throw new Error('Failed to fetch chapters');
            }
            const chapters = await response.json();
            
            if (chapters.length === 0) {
                this.useContextCheckbox.disabled = true;
                const label = document.querySelector('label[for="use-context-checkbox"]');
                if (label) {
                    label.textContent = 'Context (Not Available)';
                    label.title = 'Run load_pdfs.py to generate context files.';
                }
                return;
            }

            this.chapterNumber.innerHTML = ''; // Clear existing options
            chapters.forEach(chapter => {
                const option = document.createElement('option');
                option.value = chapter;
                option.textContent = `Chapter ${chapter}`;
                this.chapterNumber.appendChild(option);
            });
        } catch (error) {
            console.error('Failed to populate chapters:', error);
            this.useContextCheckbox.disabled = true;
            const label = document.querySelector('label[for="use-context-checkbox"]');
            if (label) {
                label.textContent = 'Context (Error)';
            }
        }
    }

    async loadConfig() {
        try {
            const response = await fetch('/config');
            const config = await response.json();
            this.tokensAvailable = config.tokens_available;
            await this.updateApiToken();
            console.log('Configuration loaded successfully');
        } catch (error) {
            console.error('Failed to load config:', error);
            this.tokensAvailable = {};
        }
    }

    async updateApiToken() {
        const provider = this.llmProvider.value;

        // Check if token is available for this provider
        if (this.tokensAvailable && this.tokensAvailable[provider]) {
            try {
                const response = await fetch(`/token/${provider}`);
                if (response.ok) {
                    const data = await response.json();
                    this.apiToken.value = data.token;
                } else {
                    // Token not available, clear the field
                    this.apiToken.value = '';
                }
            } catch (error) {
                console.error('Failed to fetch API token:', error);
                this.apiToken.value = '';
            }
        } else {
            // Token not available, clear the field
            this.apiToken.value = '';
        }
    }

    updateModelOptions() {
        const provider = this.llmProvider.value;
        const options = this.modelOptions[provider] || [];
        
        // Clear existing options
        this.modelSelect.innerHTML = '';
        
        // Add new options
        options.forEach(option => {
            const optionElement = document.createElement('option');
            optionElement.value = option.value;
            optionElement.textContent = option.text;
            this.modelSelect.appendChild(optionElement);
        });
        
        // Select the first option by default (which should be the default model)
        if (options.length > 0) {
            this.modelSelect.value = options[0].value;
        }
    }

    toggleContextControls() {
        const isContextEnabled = this.useContextCheckbox.checked;
        
        if (isContextEnabled) {
            this.chapterSelection.classList.remove('hidden');
            this.chapterNumber.disabled = false;
            this.chapterNumber.required = true;
        } else {
            this.chapterSelection.classList.add('hidden');
            this.chapterNumber.disabled = true;
            this.chapterNumber.required = false;
        }
    }

    showLoading() {
        this.loading.classList.remove('hidden');
        this.translateBtn.disabled = true;
    }

    hideLoading() {
        this.loading.classList.add('hidden');
        this.translateBtn.disabled = false;
    }

    showError(message) {
        this.errorMessage.textContent = message;
        this.errorMessage.classList.remove('hidden');
        setTimeout(() => {
            this.errorMessage.classList.add('hidden');
        }, 5000);
    }

    showInfo(message) {
        this.infoMessage.textContent = message;
        this.infoMessage.classList.remove('hidden');
        setTimeout(() => {
            this.infoMessage.classList.add('hidden');
        }, 3000);
    }

    hideInfoMessage() {
        this.infoMessage.classList.add('hidden');
    }

    async translate() {
        if (!this.validateTranslationInputs()) {
            return;
        }

        const requestData = this.buildTranslationRequest();

        this.hideInfoMessage(); // Clear any previous info messages
        this.showLoading();

        try {
            // Always use streaming translation
            await this.handleStreamingTranslation(requestData);
            // History refresh is handled in finishStreamingTranslation after completion

        } catch (error) {
            this.showError(`Translation error: ${error.message}`);
        } finally {
            this.hideLoading();
        }
    }

    validateTranslationInputs() {
        const text = this.inputText.value.trim();
        const token = this.apiToken.value.trim();
        const useContext = this.useContextCheckbox.checked;
        const chapterNumber = this.chapterNumber.value;

        if (!text) {
            this.showError('Please enter text to translate');
            return false;
        }

        if (!token) {
            this.showError('Please enter API token');
            return false;
        }

        if (useContext && !chapterNumber) {
            this.showError('Please enter a chapter number when context is enabled');
            return false;
        }

        return true;
    }

    buildTranslationRequest() {
        const text = this.inputText.value.trim();
        const provider = this.llmProvider.value;
        const model = this.modelSelect.value;
        const token = this.apiToken.value.trim();
        const useContext = this.useContextCheckbox.checked;
        const chapterNumber = this.chapterNumber.value;

        const requestBody = {
            text: text,
            llm_provider: provider,
            api_token: token,
            use_context: useContext,
            stream: true  // Always enable streaming
        };

        if (model) {
            requestBody.model = model;
        }

        if (useContext && chapterNumber) {
            requestBody.chapter_number = parseInt(chapterNumber);
        }

        return requestBody;
    }

    async makeTranslationRequest(requestData) {
        const response = await fetch('/translate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestData)
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Translation failed');
        }

        return await response.json();
    }

    addContextInfoToOutput(context) {
        const score = (context.score * 100).toFixed(2);
        const contextInfo = document.createElement('div');
        contextInfo.className = 'context-info-output';
        contextInfo.innerHTML = `
            <div class="context-info-header">
                📚 <strong>Context Retrieved:</strong> A relevant text block was found with a similarity score of <strong>${score}%</strong> and used to improve the translation.
            </div>
        `;

        // Insert before the label (first child of output-section)
        const label = this.outputSection.querySelector('label');
        this.outputSection.insertBefore(contextInfo, label);
    }

    clearContextInfoFromOutput() {
        // Remove any existing context info from output section
        const existingContextInfo = this.outputSection.querySelector('.context-info-output');
        if (existingContextInfo) {
            existingContextInfo.remove();
        }
    }

    displayTranslation(translatedText, dictionaryMatches, context) {
        // Clear previous context info
        this.clearContextInfoFromOutput();

        // Clear previous translation content
        this.outputText.innerHTML = '';

        // Show context information above the Translation label if available
        if (context && context.text) {
            this.addContextInfoToOutput(context);
        }

        // Create translation text container
        const translationContainer = document.createElement('div');

        if (!dictionaryMatches || dictionaryMatches.length === 0) {
            translationContainer.textContent = translatedText;
            this.outputText.appendChild(translationContainer);
            return;
        }

        // Create a list of positions to highlight
        const highlightPositions = [];
        
        dictionaryMatches.forEach(match => {
            const [originalTerm, translatedTerm] = match;
            
            // Find all occurrences of the translated term in the output
            const isChineseOutput = /[\u4e00-\u9fff]/.test(translatedText);
            const escapedTerm = translatedTerm.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            
            let regex;
            if (isChineseOutput && /[\u4e00-\u9fff]/.test(translatedTerm)) {
                // Chinese term in Chinese output - no word boundaries needed
                regex = new RegExp(escapedTerm, 'g');
            } else {
                // English term or mixed - use word boundaries
                regex = new RegExp(`\\b${escapedTerm}\\b`, 'gi');
            }
            
            let regexMatch;
            while ((regexMatch = regex.exec(translatedText)) !== null) {
                highlightPositions.push({
                    start: regexMatch.index,
                    end: regexMatch.index + translatedTerm.length,
                    originalTerm: originalTerm,
                    translatedTerm: translatedTerm
                });
            }
        });

        // Sort by start position and merge overlapping positions
        highlightPositions.sort((a, b) => a.start - b.start);
        
        // Remove overlaps - keep the longest match when there's overlap
        const mergedPositions = [];
        for (let i = 0; i < highlightPositions.length; i++) {
            const current = highlightPositions[i];
            let shouldAdd = true;
            
            // Check if this overlaps with any existing position
            for (let j = 0; j < mergedPositions.length; j++) {
                const existing = mergedPositions[j];
                if (current.start < existing.end && current.end > existing.start) {
                    // There's an overlap
                    if (current.end - current.start > existing.end - existing.start) {
                        // Current is longer, replace existing
                        mergedPositions[j] = current;
                    }
                    shouldAdd = false;
                    break;
                }
            }
            
            if (shouldAdd) {
                mergedPositions.push(current);
            }
        }
        
        // Sort again and build the highlighted text
        mergedPositions.sort((a, b) => a.start - b.start);
        
        let highlightedText = '';
        let lastEnd = 0;
        
        mergedPositions.forEach(pos => {
            // Add text before this highlight
            highlightedText += translatedText.substring(lastEnd, pos.start);
            
            // Add the highlighted term
            highlightedText += `<span class="dictionary-match" title="${pos.originalTerm} → ${pos.translatedTerm}">${pos.translatedTerm}</span>`;
            
            lastEnd = pos.end;
        });
        
        // Add remaining text
        highlightedText += translatedText.substring(lastEnd);

        translationContainer.innerHTML = highlightedText;
        this.outputText.appendChild(translationContainer);
    }

    async copyToClipboard() {
        const text = this.outputText.textContent;
        
        if (!text) {
            this.showError('No text to copy');
            return;
        }

        try {
            await navigator.clipboard.writeText(text);
            
            // Visual feedback
            const originalText = this.copyBtn.textContent;
            this.copyBtn.textContent = 'Copied!';
            this.copyBtn.style.backgroundColor = '#27ae60';
            
            setTimeout(() => {
                this.copyBtn.textContent = originalText;
                this.copyBtn.style.backgroundColor = '';
            }, 2000);
            
        } catch (error) {
            // Fallback for older browsers
            const textArea = document.createElement('textarea');
            textArea.value = text;
            document.body.appendChild(textArea);
            textArea.focus();
            textArea.select();
            
            try {
                document.execCommand('copy');
                this.showError('Text copied to clipboard');
            } catch (fallbackError) {
                this.showError('Failed to copy text to clipboard');
            }
            
            document.body.removeChild(textArea);
        }
    }

    async loadTranslationHistory() {
        try {
            const response = await fetch('/history');
            if (!response.ok) {
                throw new Error('Failed to fetch translation history');
            }

            const history = await response.json();
            this.displayHistory(history);

        } catch (error) {
            console.error('Failed to load translation history:', error);
            this.historyList.innerHTML = '<p class="no-history">Failed to load history.</p>';
        }
    }

    displayHistory(history) {
        if (!history || history.length === 0) {
            this.historyList.innerHTML = '<p class="no-history">No translation history yet.</p>';
            return;
        }

        const historyHtml = history.map(item => {
            const timestamp = new Date(item.timestamp).toLocaleString();
            const direction = `${item.source_language.toUpperCase()} → ${item.target_language.toUpperCase()}`;

            // Truncate long text for display
            const originalText = item.original_text.length > 50
                ? item.original_text.substring(0, 50) + '...'
                : item.original_text;
            const translatedText = item.translated_text.length > 50
                ? item.translated_text.substring(0, 50) + '...'
                : item.translated_text;

            const contextUsed = item.use_context && item.chapter_number
                ? `<span class="history-context-used">Ch.${item.chapter_number}</span>`
                : '';

            return `
                <div class="history-item" data-original="${this.escapeHtml(item.original_text)}"
                     data-translated="${this.escapeHtml(item.translated_text)}">
                    <div class="history-item-header">
                        <span class="history-timestamp">${timestamp}</span>
                        <span class="history-direction">${direction}</span>
                    </div>
                    <div class="history-text">
                        <div class="history-original">${this.escapeHtml(originalText)}</div>
                        <div class="history-translated">${this.escapeHtml(translatedText)}</div>
                    </div>
                    <div class="history-meta">
                        <span class="history-provider">${item.llm_provider}</span>
                        ${contextUsed}
                        ${item.model ? `<span>${item.model}</span>` : ''}
                    </div>
                </div>
            `;
        }).join('');

        this.historyList.innerHTML = historyHtml;

        // Add click handlers for history items
        this.historyList.querySelectorAll('.history-item').forEach((item, index) => {
            item.addEventListener('click', () => {
                const historyItem = history[index];
                const original = item.dataset.original;
                const translated = item.dataset.translated;

                // Fill the input with original text
                this.inputText.value = original;

                // Restore translation with proper highlighting using stored dictionary matches
                this.displayTranslation(translated, historyItem.dictionary_matches, historyItem.context);

                // Scroll to translation area
                this.inputText.scrollIntoView({ behavior: 'smooth' });
            });
        });
    }

    async clearTranslationHistory() {
        if (!confirm('Are you sure you want to clear the translation history?')) {
            return;
        }

        try {
            const response = await fetch('/history', {
                method: 'DELETE'
            });

            if (!response.ok) {
                throw new Error('Failed to clear history');
            }

            this.historyList.innerHTML = '<p class="no-history">No translation history yet.</p>';
            this.showInfo('Translation history cleared successfully.');

        } catch (error) {
            console.error('Failed to clear history:', error);
            this.showError('Failed to clear translation history');
        }
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    async handleStreamingTranslation(requestData) {
        /**Handle streaming translation using Server-Sent Events.**/
        this.prepareStreamingDisplay();

        try {
            const response = await fetch('/translate-stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestData)
            });

            if (!response.ok) {
                throw new Error('Failed to start streaming translation');
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            let completeTranslation = '';
            let dictionaryMatches = [];
            let contextInfo = null;

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop(); // Keep incomplete line in buffer

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));

                            if (data.type === 'chunk') {
                                completeTranslation += data.content;
                                this.updateStreamingText(completeTranslation);
                            } else if (data.type === 'metadata') {
                                dictionaryMatches = data.dictionary_matches;
                            } else if (data.type === 'context') {
                                contextInfo = data.context;
                                this.addContextInfoToOutput(data.context);
                            } else if (data.type === 'complete') {
                                await this.finishStreamingTranslation(completeTranslation, dictionaryMatches, contextInfo);
                                return;
                            } else if (data.type === 'error') {
                                throw new Error(data.message);
                            }
                        } catch (e) {
                            console.error('Error parsing streaming data:', e);
                        }
                    }
                }
            }

        } catch (error) {
            console.error('Streaming error:', error);
            throw error;
        }
    }

    prepareStreamingDisplay() {
        /**Prepare the output area for streaming display.**/
        this.clearContextInfoFromOutput();
        this.outputText.innerHTML = '';

        // Add streaming container
        const streamingContainer = document.createElement('div');
        streamingContainer.className = 'streaming-container';
        this.outputText.appendChild(streamingContainer);
    }

    updateStreamingText(text) {
        /**Update the streaming text display.**/
        const container = this.outputText.querySelector('.streaming-container');
        if (container) {
            container.innerHTML = text + '<span class="streaming-cursor">▊</span>';
        }
    }

    async finishStreamingTranslation(fullText, dictionaryMatches, contextInfo) {
        /**Finish streaming translation and apply highlighting.**/
        this.displayTranslation(fullText, dictionaryMatches, contextInfo);

        // Refresh translation history after streaming is complete
        await this.loadTranslationHistory();
    }
}

// Initialize the app when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new TranslationApp();
});