class TranslationApp {
    constructor() {
        this.initializeElements();
        this.bindEvents();
        this.loadConfig();
    }

    initializeElements() {
        this.llmProvider = document.getElementById('llm-provider');
        this.apiToken = document.getElementById('api-token');
        this.inputText = document.getElementById('input-text');
        this.outputText = document.getElementById('output-text');
        this.translateBtn = document.getElementById('translate-btn');
        this.copyBtn = document.getElementById('copy-btn');
        this.loading = document.getElementById('loading');
        this.errorMessage = document.getElementById('error-message');
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

        // Auto-fill API token when provider changes
        this.llmProvider.addEventListener('change', () => this.updateApiToken());
    }

    async loadConfig() {
        try {
            const response = await fetch('/config');
            const config = await response.json();
            this.defaultTokens = config.default_tokens;
            this.updateApiToken();
        } catch (error) {
            console.error('Failed to load config:', error);
            this.defaultTokens = {};
        }
    }

    updateApiToken() {
        const provider = this.llmProvider.value;
        if (this.defaultTokens && this.defaultTokens[provider]) {
            this.apiToken.value = this.defaultTokens[provider];
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

    async translate() {
        const text = this.inputText.value.trim();
        const provider = this.llmProvider.value;
        const token = this.apiToken.value.trim();

        if (!text) {
            this.showError('Please enter text to translate');
            return;
        }

        if (!token) {
            this.showError('Please enter API token');
            return;
        }

        this.showLoading();

        try {
            const response = await fetch('/translate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: text,
                    llm_provider: provider,
                    api_token: token
                })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Translation failed');
            }

            const result = await response.json();
            this.displayTranslation(result.translated_text, result.dictionary_matches);

        } catch (error) {
            this.showError(`Translation error: ${error.message}`);
        } finally {
            this.hideLoading();
        }
    }

    displayTranslation(translatedText, dictionaryMatches) {
        // Clear previous content
        this.outputText.innerHTML = '';

        if (!dictionaryMatches || dictionaryMatches.length === 0) {
            this.outputText.textContent = translatedText;
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

        this.outputText.innerHTML = highlightedText;
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
}

// Initialize the app when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new TranslationApp();
});