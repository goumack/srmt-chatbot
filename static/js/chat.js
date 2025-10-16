// SRMT ChatBot UI - enhanced interactions, theme toggle, smooth UX

function addMessageEffects() {
    document.querySelectorAll('.message-container:not(.processed)').forEach(container => {
        container.classList.add('processed');
    });
    // Auto-scroll to bottom after effects
    scrollToBottom();
}

function addChatButtons() {
    const buttons = document.querySelectorAll('.chat-button:not(.processed)');
    buttons.forEach((button, index) => {
        button.classList.add('processed');
        setTimeout(() => {
            button.style.transform = 'translateY(0)';
            button.style.opacity = '1';
        }, index * 100);
    });
}

// Document reference highlight
function showDocReference(id) {
    const reference = document.getElementById(id);
    if (reference) {
        reference.classList.add('highlight');
        setTimeout(() => reference.classList.remove('highlight'), 1200);
        reference.style.display = 'block';
        reference.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
}

// Scroll helpers
function scrollToBottom() {
    const cm = document.getElementById('chat-messages');
    if (!cm) return;
    cm.scrollTop = cm.scrollHeight;
}

function toggleScrollButton() {
    const cm = document.getElementById('chat-messages');
    const btn = document.getElementById('scroll-bottom');
    if (!cm || !btn) return;
    const nearBottom = cm.scrollHeight - cm.scrollTop - cm.clientHeight < 120;
    btn.classList.toggle('show', !nearBottom);
}

// Theme toggle with persistence
function setupThemeToggle() {
    const btn = document.getElementById('theme-toggle');
    if (!btn) return;
    const saved = localStorage.getItem('srmt-theme');
    if (saved) document.documentElement.setAttribute('data-theme', saved);
    btn.addEventListener('click', () => {
        const current = document.documentElement.getAttribute('data-theme');
        const next = current === 'dark' ? 'light' : 'dark';
        if (next === 'light') {
            document.documentElement.removeAttribute('data-theme');
            localStorage.removeItem('srmt-theme');
        } else {
            document.documentElement.setAttribute('data-theme', 'dark');
            localStorage.setItem('srmt-theme', 'dark');
        }
    });
}

// Initialize UI
document.addEventListener('DOMContentLoaded', function() {
    setupThemeToggle();
    setTimeout(addChatButtons, 300);
    setTimeout(addMessageEffects, 300);

    const cm = document.getElementById('chat-messages');
    const sb = document.getElementById('scroll-bottom');
    if (cm) {
        cm.addEventListener('scroll', toggleScrollButton);
        // Initial state
        scrollToBottom();
        toggleScrollButton();
    }
    if (sb) sb.addEventListener('click', scrollToBottom);

    const messageInput = document.getElementById('message-input');
    if (messageInput) messageInput.focus();
});

// Handle message submission: keep original server flow but improve UX
function submitMessage() {
    const messageInput = document.getElementById('message-input');
    const message = messageInput.value.trim();
    if (!message) return false;

    const loading = document.getElementById('loading-indicator');
    loading.style.display = 'block';

    // Disable during submit
    messageInput.disabled = true;
    const sendBtn = document.getElementById('send-button');
    sendBtn.disabled = true;

    // Optimistic UI: append the user message locally for fluidity
    const chatMessages = document.getElementById('chat-messages');
    const container = document.createElement('div');
    container.className = 'message-container clearfix processed';
    const bubble = document.createElement('div');
    bubble.className = 'user-message';
    bubble.textContent = message;
    container.appendChild(bubble);
    chatMessages.appendChild(container);
    scrollToBottom();

    // Let the form submit to server to get bot response
    return true;
}