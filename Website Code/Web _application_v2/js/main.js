// Function to handle prompt submission (if applicable)
document.getElementById('submitBtn')?.addEventListener('click', async () => {
    const prompt = document.getElementById('promptInput')?.value;
    if (prompt) {
        // Post the prompt to your server or handle it as needed
        console.log('Prompt submitted:', prompt);
        postPrompt(prompt); // Ensure this function is defined or adjust according to your needs
    } else {
        alert('Please enter a prompt.');
    }
});

// Function to post the user prompt to the server (if applicable)
async function postPrompt(prompt) {
    // Adjust '/api/post-prompt' to your actual API endpoint
    const response = await fetch('/api/post-prompt', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt })
    });

    if (response.ok) {
        const result = await response.json();
        console.log('Prompt posted successfully:', result);
        // Additional actions based on the response, e.g., fetching an image
    } else {
        console.error('Failed to post prompt:', await response.text());
    }
}

// Function to fetch an image based on a prompt ID (if applicable)
async function getImage(promptId) {
    // Adjust '/api/get-image/' to your actual API endpoint
    const response = await fetch(`/api/get-image/${promptId}`);

    if (response.ok) {
        const imageBlob = await response.blob();
        const imageUrl = URL.createObjectURL(imageBlob);
        document.getElementById('generatedImage').src = imageUrl; // Ensure this ID matches your image element
    } else {
        console.error('Failed to fetch image:', await response.text());
    }
}
