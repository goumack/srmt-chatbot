// SRMT ChatBot Production UI functions - enhanced

// Animation effect for messages
function addMessageEffects() {
    document.querySelectorAll('.message-container:not(.processed)').forEach((container, index) => {
        container.classList.add('processed');
        setTimeout(() => {
            container.style.opacity = '1';
        }, index * 100); // Staggered animation
    });
    
    // Auto-scroll to latest message
    const chatMessages = document.getElementById('chat-messages');
    if (chatMessages) {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
}

// Add animation to suggestion buttons
function addChatButtons() {
    const buttons = document.querySelectorAll('.chat-button:not(.processed)');
    buttons.forEach((button, index) => {
        button.classList.add('processed');
        setTimeout(() => {
            button.style.transform = 'translateY(0)';
            button.style.opacity = '1';
        }, index * 150); // Staggered animation for buttons
    });
}

// Function to handle document references display
function showDocReference(id) {
    const reference = document.getElementById(id);
    
    // Hide all other references first
    document.querySelectorAll('.reference-details').forEach(ref => {
        if (ref.id !== id) {
            ref.style.display = 'none';
        }
    });
    
    // Toggle this reference
    if (reference) {
        const isVisible = reference.style.display === 'block';
        reference.style.display = isVisible ? 'none' : 'block';
        
        if (!isVisible) {
            reference.classList.add('highlight');
            setTimeout(() => {
                reference.classList.remove('highlight');
            }, 1000);
            
            // Scroll to make visible
            setTimeout(() => {
                reference.scrollIntoView({behavior: 'smooth', block: 'nearest'});
            }, 100);
        }
    }
}

// Form submission handling
function submitMessage() {
    const messageInput = document.getElementById('message-input');
    const message = messageInput.value.trim();
    
    if (message) {
        // Show loading indicator with smooth fade-in
        const loadingIndicator = document.getElementById('loading-indicator');
        loadingIndicator.style.opacity = '0';
        loadingIndicator.style.display = 'block';
        
        setTimeout(() => {
            loadingIndicator.style.opacity = '1';
        }, 10);
        
        // Disable input during processing
        messageInput.disabled = true;
        document.getElementById('send-button').disabled = true;
        
        // Add user message to UI immediately for better UX
        const chatMessages = document.getElementById('chat-messages');
        const messageContainer = document.createElement('div');
        messageContainer.className = 'message-container clearfix';
        
        const userMessage = document.createElement('div');
        userMessage.className = 'user-message';
        userMessage.textContent = message;
        
        messageContainer.appendChild(userMessage);
        chatMessages.appendChild(messageContainer);
        
        // Add animation effect
        setTimeout(() => {
            messageContainer.style.opacity = '1';
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }, 10);
    }
    
    return message.length > 0;
}

// Initialize UI
document.addEventListener('DOMContentLoaded', function() {
    // Initialize chat buttons with animation
    setTimeout(addChatButtons, 500);
    
    // Add message effects to any existing messages
    setTimeout(() => addMessageEffects(), 300);
    
    // Focus on input field
    const messageInput = document.getElementById('message-input');
    if (messageInput) {
        messageInput.focus();
    }
    
    // Ensure chat container takes appropriate height
    adjustChatContainerHeight();
    window.addEventListener('resize', adjustChatContainerHeight);
    
    // Theme toggle persistence
    const btn = document.getElementById('theme-toggle');
    if (btn) {
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
    
    // Scroll-to-bottom button
    const cm = document.getElementById('chat-messages');
    const sb = document.getElementById('scroll-bottom');
    const toggleScrollBtn = () => {
        if (!cm || !sb) return;
        const nearBottom = cm.scrollHeight - cm.scrollTop - cm.clientHeight < 120;
        sb.classList.toggle('show', !nearBottom);
    };
    if (cm) cm.addEventListener('scroll', toggleScrollBtn);
    if (sb) sb.addEventListener('click', () => { cm.scrollTop = cm.scrollHeight; toggleScrollBtn(); });
    toggleScrollBtn();
});

// Adjust chat container height
function adjustChatContainerHeight() {
    const chatContainer = document.querySelector('.chat-container');
    if (chatContainer) {
        const viewportHeight = window.innerHeight;
        chatContainer.style.height = `${viewportHeight - 60}px`;
    }
}